import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("🔍 Object Counting (Connected Components)")
st.caption("사진을 올리고 사이드바에서 전처리·임계값·모폴로지 파라미터를 조정해 보세요.")

# ---------------- Sidebar: 파라미터 ----------------
st.sidebar.header("1. 이미지")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
max_side = st.sidebar.slider("처리용 최대 변 길이 (px, 속도용 리사이즈)", 256, 4000, 1500, step=64,
                             help="휴대폰 사진은 매우 커서 느릴 수 있습니다. 원본 크기로 하려면 최대로 올리세요.")

st.sidebar.header("2. 노이즈 제거")
denoise = st.sidebar.selectbox("필터", ["Median", "Gaussian", "없음"])
if denoise == "Median":
    med_k = st.sidebar.slider("Median kernel 크기 (홀수)", 1, 21, 5, step=2)
elif denoise == "Gaussian":
    sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 2.0, step=0.5)

st.sidebar.header("3. Thresholding")
th_mode = st.sidebar.radio("임계값 방식", ["직접 지정", "Otsu 자동"], horizontal=True)
if th_mode == "직접 지정":
    th_val = st.sidebar.slider("임계값", 0, 255, 150)
direction = st.sidebar.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)"])

st.sidebar.header("4. 모폴로지")
close_k = st.sidebar.slider("Closing 구조요소 크기", 0, 51, 15, help="0이면 closing을 하지 않습니다.")
open_k = st.sidebar.slider("Opening 구조요소 크기", 0, 51, 0, help="작은 점 노이즈 제거용. 0이면 생략.")
fill = st.sidebar.checkbox("Hole filling", value=True)

st.sidebar.header("5. 컴포넌트 필터")
min_area = st.sidebar.number_input("최소 픽셀 수 (이보다 작으면 제외)", 0, 10_000_000, 0, step=50)
connectivity = st.sidebar.radio("연결성", ["4-연결", "8-연결"], horizontal=True)


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


def show(img, cmap="gray", title=None, colorbar=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(img, cmap=cmap)
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

if th_mode == "Otsu 자동":
    th_val = otsu(f_denoised)
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
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
    ax.set_yscale("log")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

f_mask = f_denoised < th_val if direction.startswith("어둡") else f_denoised > th_val
if open_k > 0:
    f_mask = nd.binary_opening(f_mask, structure=np.ones((open_k, open_k)))
if close_k > 0:
    f_mask = nd.binary_closing(f_mask, structure=np.ones((close_k, close_k)))
if fill:
    f_mask = nd.binary_fill_holes(f_mask)
with c2:
    show(f_mask, title="binary mask")

# 4. Connected components
structure = None if connectivity == "4-연결" else np.ones((3, 3))
labels, nums = nd.label(f_mask, structure=structure)

if nums == 0:
    st.warning("검출된 객체가 없습니다. 임계값이나 방향을 조정해 보세요.")
    st.stop()

areas = np.bincount(labels.ravel())[1:]  # label 1..nums 의 픽셀 수
idx = np.arange(1, nums + 1)
keep = areas >= min_area
if not keep.all():
    lut = np.zeros(nums + 1, dtype=labels.dtype)
    lut[idx[keep]] = np.arange(1, keep.sum() + 1)
    labels = lut[labels]
    areas = areas[keep]
    nums = int(keep.sum())
    idx = np.arange(1, nums + 1)

if nums == 0:
    st.warning("최소 픽셀 수 조건을 만족하는 객체가 없습니다.")
    st.stop()

cc = pd.DataFrame({"index": idx, "number of pixels": areas})

st.subheader(f"4. Connected components — 객체 수: **{nums}**")
c1, c2 = st.columns([3, 2])
with c1:
    lab_show = np.ma.masked_where(labels == 0, labels)
    show(lab_show, cmap="Spectral_r", title="labels")
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
