import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image, ImageOps
import streamlit as st


st.set_page_config(page_title="Object Counting", layout="wide")
st.title("🔍 Object Counting")
st.caption("사진을 올리고 사이드바에서 전처리·임계값·모폴로지 파라미터를 조정해 보세요.")

# ---------------- Sidebar: 파라미터 ----------------
st.sidebar.header("1. 이미지")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
# [변경] 기본값 1500 → 1000. 모폴로지 크기 같은 px 단위 파라미터는 해상도가 바뀌면 의미가 달라지므로
#        기본 해상도를 고정해 두고 웬만하면 건드리지 않는 게 안정적
max_side = st.sidebar.slider("처리용 최대 변 길이 (px, 속도용 리사이즈)", 256, 4000, 1000, step=64,
                             help="바꾸면 아래 모폴로지 크기도 같이 조정해야 합니다.")

st.sidebar.header("2. 노이즈 제거")
denoise = st.sidebar.selectbox("필터", ["Median", "Gaussian", "없음"])
if denoise == "Median":
    med_k = st.sidebar.slider("Median kernel 크기 (홀수)", 1, 21, 5, step=2)
elif denoise == "Gaussian":
    sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 2.0, step=0.5)

st.sidebar.header("3. Thresholding")
# [추가] 배경 보정: 조명 불균일 + 배경보다 밝은/어두운 물체를 한 번에 처리
bg_corr = st.sidebar.checkbox("배경 보정 (|영상 − 배경|)", value=True,
                              help="배경 밝기를 추정해서 뺀 뒤 차이로 thresholding. "
                                   "조명이 고르지 않거나 밝은 물체·어두운 물체가 섞여 있을 때 필수.")
th_mode = st.sidebar.radio("임계값 방식", ["자동", "직접 지정"], horizontal=True)  # [변경] 기본을 자동으로
if th_mode == "직접 지정":
    th_val = st.sidebar.slider("임계값", 0, 255, 30 if bg_corr else 150)
else:
    th_scale = st.sidebar.slider("자동 임계값 배율", 0.3, 2.0, 1.0, 0.05,
                                 help="낮추면 흐린 물체까지, 높이면 배경 잡티가 줄어듦")
if not bg_corr:
    direction = st.sidebar.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)"])

st.sidebar.header("4. 모폴로지")
# [변경] 정사각형 → 원형 구조요소, 순서 closing → fill → opening
close_k = st.sidebar.slider("Closing 크기 (조각 잇기)", 0, 51, 9, step=2,
                            help="한 물체가 여러 조각이면 키우고, 물체끼리 붙으면 줄이세요.")
fill = st.sidebar.checkbox("Hole filling", value=True)
open_k = st.sidebar.slider("Opening 크기 (얇은 연결 끊기)", 0, 51, 5, step=2,
                           help="물체끼리 그림자로 이어져 있으면 키우세요.")

st.sidebar.header("5. 컴포넌트 필터")
# [변경] 절대 픽셀 수 → 영상 대비 %. 해상도가 달라도 같은 의미
min_pct = st.sidebar.slider("최소 면적 (영상 대비 %)", 0.0, 2.0, 0.05, 0.01)
connectivity = st.sidebar.radio("연결성", ["8-연결", "4-연결"], horizontal=True)  # [변경] 기본 8-연결

# [추가] 붙은 물체 분리
st.sidebar.header("6. 붙은 물체 분리")
do_split = st.sidebar.checkbox("Watershed로 분리", value=True)
solidity_th = st.sidebar.slider("분리 대상 기준 (solidity)", 0.5, 1.0, 0.85, 0.01, disabled=not do_split,
                                help="덩어리가 이 값보다 오목하면 여러 물체가 붙은 것으로 보고 나눔. 높일수록 적극적.")
min_dist = st.sidebar.slider("물체 중심 최소 간격 (px)", 3, 100, 15, disabled=not do_split,
                             help="잘게 쪼개지면 키우고, 덜 나뉘면 줄이세요.")


# ---------------- 처리 함수 ----------------
def otsu(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.nanargmax(sigma_b))


def disk(size):
    # [추가] 원형 구조요소. 정사각형은 모서리 방향으로 더 많이 붙여서 대각선 이웃끼리 잘 붙음
    r = size // 2
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


@st.cache_data(show_spinner=False)
def load_gray(file_bytes, max_side):
    I = Image.open(io.BytesIO(file_bytes))
    I = ImageOps.exif_transpose(I)  # 휴대폰 사진 회전 보정
    I = I.convert("L")
    I.thumbnail((max_side, max_side))
    return np.array(I)


@st.cache_data(show_spinner=False)
def do_denoise(f, mode, param):
    if mode == "Median" and param > 1:
        return signal.medfilt2d(f.astype(float), param).astype(np.uint8)
    if mode == "Gaussian" and param > 0:
        return nd.gaussian_filter(f.astype(float), param).clip(0, 255).astype(np.uint8)
    return f


@st.cache_data(show_spinner=False)
def estimate_background(f):
    # [추가] 작게 줄여서 큰 median → 물체는 지워지고 배경 밝기만 남음 → 원래 크기로
    h, w = f.shape
    sh, sw = 120, max(1, round(w * 120 / h))
    small = np.array(Image.fromarray(f).resize((sw, sh), Image.BILINEAR))
    k = max(3, int(min(sh, sw) * 0.25) | 1)
    bg_small = nd.median_filter(small, size=k, mode="nearest")
    return np.array(Image.fromarray(bg_small).resize((w, h), Image.BILINEAR))


def split_touching(labels, solidity_th, min_dist):
    # [추가] 오목한 덩어리(= 여러 물체가 붙은 것)만 distance transform + watershed로 분리.
    #        립스틱처럼 길쭉해도 볼록한 단일 물체는 solidity가 높아서 그대로 둠
    out = np.zeros_like(labels)
    nxt = 1
    for p in measure.regionprops(labels):
        r0, c0, r1, c1 = p.bbox
        sub = np.pad(p.image, 1)
        parts = None
        if p.solidity < solidity_th:
            dist = nd.distance_transform_edt(sub)
            peaks = (dist == nd.maximum_filter(dist, size=2 * min_dist + 1)) & (dist > 0.3 * dist.max())
            markers, nm = nd.label(peaks)
            if nm > 1:
                ws = segmentation.watershed(-dist, markers, mask=sub)
                areas = np.bincount(ws.ravel())[1:]
                if areas.min() >= 0.1 * p.area:  # 너무 작은 조각이 생기면 과분할로 보고 취소
                    parts = [ws == k for k in range(1, nm + 1)]
        if parts is None:
            parts = [sub]
        for part in parts:
            out[r0:r1, c0:c1][part[1:-1, 1:-1]] = nxt
            nxt += 1
    return out, nxt - 1


def show(img, cmap="gray", title=None, colorbar=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


# ---------------- 메인 ----------------
if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드하세요.")
    st.stop()

f = load_gray(uploaded.getvalue(), max_side)
param = med_k if denoise == "Median" else (sigma if denoise == "Gaussian" else 0)
with st.spinner("노이즈 제거 중..."):
    f_denoised = do_denoise(f, denoise, param)

# [추가] thresholding 대상 영상: 배경 보정을 켜면 |f - 배경|
if bg_corr:
    bg = estimate_background(f_denoised)
    f_th = np.abs(f_denoised.astype(int) - bg.astype(int)).astype(np.uint8)
else:
    f_th = f_denoised

if th_mode == "자동":
    # [변경] 배경 보정 영상은 0 근처에 큰 봉우리 + 긴 꼬리 모양이라 Otsu(봉우리 2개 가정)는 너무 높게 잡음
    #        → 이런 분포에 강한 Li 방법 사용. 배경 보정을 끄면 기존 Otsu
    if bg_corr:
        th_val = int(filters.threshold_li(f_th) * th_scale)
        st.sidebar.success(f"Li 자동 임계값: {th_val}")
    else:
        th_val = int(otsu(f_th) * th_scale)
        st.sidebar.success(f"Otsu 임계값: {th_val}")

# 1~2. 원본 / 노이즈 제거
st.subheader("1–2. 입력 영상 & 노이즈 제거")
c1, c2 = st.columns(2)
with c1:
    show(f, title=f"input (grayscale) {f.shape[1]}×{f.shape[0]}", colorbar=True)
with c2:
    show(f_denoised, title=f"denoised ({denoise})", colorbar=True)

# 3. 히스토그램 + 마스크
st.subheader("3. Histogram & Binary mask")
if bg_corr:
    c0, c1, c2 = st.columns(3)
    with c0:
        show(f_th, title="|f − background|", colorbar=True)
else:
    c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(f_th.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
    ax.set_yscale("log")
    ax.set_xlabel("|f − background|" if bg_corr else "Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

if bg_corr:
    f_mask = f_th > th_val
else:
    f_mask = f_denoised < th_val if direction.startswith("어둡") else f_denoised > th_val
# [변경] 순서: closing(조각 잇기) → fill → opening(얇은 연결 끊기)
#        기존처럼 opening 다음 closing을 하면, 끊어 놓은 물체 사이를 closing이 다시 붙여 버림
if close_k > 0:
    f_mask = nd.binary_closing(f_mask, structure=disk(close_k))
if fill:
    f_mask = nd.binary_fill_holes(f_mask)
if open_k > 0:
    f_mask = nd.binary_opening(f_mask, structure=disk(open_k))
with c2:
    show(f_mask, title="binary mask")

# 4. Connected components
structure = None if connectivity == "4-연결" else np.ones((3, 3))
labels, nums = nd.label(f_mask, structure=structure)

if nums == 0:
    st.warning("검출된 객체가 없습니다. 임계값이나 방향을 조정해 보세요.")
    st.stop()


def filter_small(labels, nums, min_area):
    areas = np.bincount(labels.ravel(), minlength=nums + 1)[1:]
    keep = areas >= min_area
    lut = np.zeros(nums + 1, dtype=labels.dtype)
    lut[1:][keep] = np.arange(1, keep.sum() + 1)
    return lut[labels], int(keep.sum())


min_area = int(min_pct / 100 * f.size)
labels, nums = filter_small(labels, nums, min_area)       # 잡티 먼저 제거
if do_split and nums > 0:
    labels, nums = split_touching(labels, solidity_th, min_dist)
    labels, nums = filter_small(labels, nums, min_area)   # 분리 후 생긴 작은 조각 제거

if nums == 0:
    st.warning("최소 면적 조건을 만족하는 객체가 없습니다.")
    st.stop()

areas = np.bincount(labels.ravel())[1:]
idx = np.arange(1, nums + 1)
cc = pd.DataFrame({"index": idx, "number of pixels": areas})

st.subheader(f"4. Connected components — 객체 수: **{nums}**")
c1, c2 = st.columns([3, 2])
with c1:
    # [변경] Spectral_r는 번호가 가까운 객체끼리 색이 비슷해서 구분이 안 됨 → 랜덤 색 + 번호 표시
    rng = np.random.RandomState(0)
    colors = rng.rand(nums + 1, 3) * 0.8 + 0.2
    colors[0] = 0
    fig, ax = plt.subplots(figsize=(8, 8 * f.shape[0] / f.shape[1]))
    ax.imshow(f, cmap="gray")
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap=plt.matplotlib.colors.ListedColormap(colors),
              alpha=0.6, interpolation="nearest", vmin=0, vmax=nums)
    for p in measure.regionprops(labels):
        y, x = p.centroid
        ax.text(x, y, str(p.label), color="white", fontsize=9, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
with c2:
    st.dataframe(cc, hide_index=True, use_container_width=True, height=420)
    st.download_button("CSV 다운로드", cc.to_csv(index=False).encode("utf-8-sig"),
                       "components.csv", "text/csv")

# 5. 최대 / 최소 컴포넌트
st.subheader("5. 가장 큰 / 가장 작은 component")
i_max = int(np.argmax(areas))
i_min = int(np.argmin(areas))
st.write(f"- **the largest component**: index {idx[i_max]}, # of pixels: {areas[i_max]}")
st.write(f"- **the smallest component**: index {idx[i_min]}, # of pixels: {areas[i_min]}")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(f, cmap="gray"); axes[0].set_title("input image")
axes[1].imshow(labels == idx[i_max], cmap="Spectral_r"); axes[1].set_title("the largest component")
axes[2].imshow(labels == idx[i_min], cmap="Spectral_r"); axes[2].set_title("the smallest component")
for a in axes:
    a.axis("off")
st.pyplot(fig)
plt.close(fig)
