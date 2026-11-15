import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image, ImageOps
import streamlit as st
from scipy.spatial import ConvexHull  # [추가] 오목한 정도(solidity) 계산용 (scipy에 포함)

st.set_page_config(page_title="Object Counting", layout="wide")
st.title("🔍 Object Counting (Connected Components)")
st.caption("사진을 올리고 사이드바에서 전처리·임계값·모폴로지 파라미터를 조정해 보세요.")

# ---------------- Sidebar: 파라미터 ----------------
# [추가] 영상 유형 프리셋: 고르면 아래 슬라이더 값이 한 번에 바뀜. 그 뒤에 슬라이더로 미세 조정 가능
DEFAULTS = dict(max_side=1000, denoise="Median", med_k=5, sigma=2.0,
                bg_corr=True, polarity="둘 다", th_mode="자동", th_val=30, th_scale=1.0,
                direction="어둡다 (f < th)", close_k=9, fill=True, open_k=5,
                min_pct=0.05, connectivity="8-연결",
                do_split=True, solidity_th=0.85, area_est=False, min_dist=15)
PRESETS = {
    "기본 (흰 배경 + 크기가 제각각인 물체, 예: 화장품)": {},
    "흰 배경 + 빽빽하게 붙은 물체 (예: 동전)": dict(
        bg_corr=False, direction="어둡다 (f < th)", close_k=3, open_k=9,
        solidity_th=0.9, min_dist=20),
    "무늬 있는 바닥 + 밝은 물체 (예: 보도블록 위 낙엽)": dict(
        bg_corr=True, polarity="밝다", close_k=5, open_k=5, solidity_th=0.9, min_dist=10),
    "어두운 배경 + 같은 크기 물체 (예: 면봉)": dict(
        bg_corr=True, polarity="밝다", close_k=0, open_k=5, area_est=True),
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    st.session_state.pop("last_change", None)
    for k, v in {**DEFAULTS, **PRESETS[st.session_state["preset"]]}.items():
        if k != "max_side":
            st.session_state[k] = v


cap = st.sidebar.caption  # [추가] 설명을 물음표 대신 화면에 바로 표시

st.sidebar.header("0. 영상 유형")
cap("가장 비슷한 유형을 고르면 아래 값들이 그 유형에 맞게 한 번에 바뀝니다. "
    "그다음 결과 아래의 **지금 상황 고르기**로 미세 조정하세요.")
st.sidebar.selectbox("영상 유형", list(PRESETS), key="preset", on_change=apply_preset)

st.sidebar.header("1. 이미지")
cap("**기능** 사진을 grayscale로 불러오고, 빠른 처리를 위해 긴 변을 정해진 크기로 줄여요.")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
# [변경] 기본값 1500 → 1000. 모폴로지 크기 같은 px 단위 파라미터는 해상도가 바뀌면 의미가 달라지므로
#        기본 해상도를 고정해 두고 웬만하면 건드리지 않는 게 안정적
max_side = st.sidebar.slider("처리용 최대 변 길이 (px)", 256, 4000, step=64, key="max_side")
cap("**언제** 보통은 그대로 두세요. 바꾸면 아래 크기 값들의 의미도 같이 바뀌어요.")

st.sidebar.header("2. 노이즈 제거")
cap("**기능** 사진의 자글자글한 잡티를 흐리게 지워요.\n\n"
    "**언제** 배경 표면이 오돌토돌해서 잡티가 물체로 많이 잡히면 크기를 키우세요. "
    "너무 키우면 작은 물체와 경계가 뭉개져요.")
denoise = st.sidebar.selectbox("필터", ["Median", "Gaussian", "없음"], key="denoise")
if denoise == "Median":
    med_k = st.sidebar.slider("Median kernel 크기 (홀수)", 1, 21, step=2, key="med_k")
elif denoise == "Gaussian":
    sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, step=0.5, key="sigma")

st.sidebar.header("3. Thresholding")
cap("**기능** 밝기 기준(임계값)으로 픽셀을 물체 / 배경으로 나눠 binary mask를 만들어요.")
# [추가] 배경 보정: 조명 불균일 + 배경보다 밝은/어두운 물체를 한 번에 처리
bg_corr = st.sidebar.checkbox("배경 보정 (|영상 − 배경|)", key="bg_corr")
cap("**언제** 조명이 고르지 않을 때(한쪽이 어두운 사진) 켜세요. "
    "물체가 화면을 빽빽하게 채우면 배경을 잘못 추정하므로 끄세요.")
if bg_corr:
    # [추가] 물체 밝기 방향. 예) 보도블록 위 은행잎: 잎은 밝고 블록 틈은 어두움 → "밝음"만 골라야 틈이 안 잡힘
    polarity = st.sidebar.radio("물체가 배경보다…", ["밝다", "어둡다", "둘 다"], horizontal=True, key="polarity")
    cap("바닥의 틈이나 그림자처럼 물체가 아닌 무늬가 같이 잡히면, 물체 쪽 방향만 고르세요.")
else:
    direction = st.sidebar.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)"], key="direction")
th_mode = st.sidebar.radio("임계값 방식", ["자동", "직접 지정"], horizontal=True, key="th_mode")
if th_mode == "직접 지정":
    th_val = st.sidebar.slider("임계값", 0, 255, key="th_val")
    cap("히스토그램의 빨간 선 위치예요. 물체가 덜 잡히면 내리고, 배경이 잡히면 올리세요.")
else:
    th_scale = st.sidebar.slider("자동 임계값 배율", 0.3, 2.0, step=0.05, key="th_scale")
    cap("자동으로 구한 값에 곱해요. 물체가 덜 잡히면 내리고, 배경 잡티가 잡히면 올리세요.")

st.sidebar.header("4. 모폴로지")
cap("**기능** binary mask의 모양을 다듬어요. Closing → 구멍 채우기 → Opening 순서로 적용해요.")
# [변경] 정사각형 → 원형 구조요소, 순서 closing → fill → opening
close_k = st.sidebar.slider("Closing 크기 (조각 잇기)", 0, 51, step=2, key="close_k")
cap("**언제** 한 물체가 여러 조각으로 세어지면 키우세요. 옆 물체와 붙어 버리면 줄이세요.")
fill = st.sidebar.checkbox("Hole filling", key="fill")
cap("**언제** 물체 안쪽에 구멍(글씨, 반사광)이 뚫려 있으면 켜 두세요.")
open_k = st.sidebar.slider("Opening 크기 (얇은 연결 끊기)", 0, 51, step=2, key="open_k")
cap("**언제** 물체끼리 그림자나 가는 부분으로 이어져 있으면 키우세요. "
    "가장 얇은 물체보다 크게 잡으면 그 물체가 사라져요.")

st.sidebar.header("5. 컴포넌트 필터")
cap("**기능** 이어진 덩어리마다 번호를 매기고(connected component), 너무 작은 덩어리는 버려요.")
# [변경] 절대 픽셀 수 → 영상 대비 %. 해상도가 달라도 같은 의미
min_pct = st.sidebar.slider("최소 면적 (영상 대비 %)", 0.0, 2.0, step=0.01, key="min_pct")
cap("**언제** 작은 잡티가 객체로 세어지면 올리세요. 진짜 작은 물체가 사라지면 내리세요.")
connectivity = st.sidebar.radio("연결성", ["8-연결", "4-연결"], horizontal=True, key="connectivity")
cap("8-연결은 대각선으로 닿은 픽셀도 같은 덩어리로 봐요. 보통 8-연결을 쓰세요.")

# [추가] 붙은 물체 분리
st.sidebar.header("6. 붙은 물체 분리")
cap("**기능** 서로 맞닿아서 하나로 이어진 덩어리를 여러 물체로 나눠요.")
do_split = st.sidebar.checkbox("Watershed로 분리", key="do_split")
cap("**언제** 물체끼리 붙어 있는 사진이면 켜세요. 과제용으로 순수한 connected component만 볼 때는 끄세요.")
solidity_th = st.sidebar.slider("분리 대상 기준 (solidity)", 0.5, 1.0, step=0.01, disabled=not do_split,
                                key="solidity_th")
cap("덩어리가 이 값보다 오목하면 붙은 물체로 보고 나눠요. 붙은 덩어리가 안 나뉘면 올리고, "
    "멀쩡한 물체가 쪼개지면 내리세요.")
min_dist = st.sidebar.slider("물체 중심 최소 간격 (px)", 3, 100, disabled=not do_split, key="min_dist")
cap("물체 반지름보다 조금 작게 잡는 게 기준이에요. 한 물체가 여러 개로 쪼개지면 키우고, "
    "붙은 물체가 덜 나뉘면 줄이세요.")
# [추가] 같은 종류 물체(면봉, 동전 등)는 크기가 거의 같음 → 덩어리 면적 / 물체 1개 면적으로 몇 개가 붙었는지 추정.
#        X자로 교차하거나 나란히 딱 붙어서 모양으로는 못 나누는 경우를 해결
area_est = st.sidebar.checkbox("면적으로 개수 보정 (같은 종류 물체일 때)", key="area_est")
cap("**언제** 면봉처럼 크기가 비슷한 물체만 있을 때 켜세요. 보통 크기의 2배인 덩어리를 2개로 셉니다. "
    "X자로 교차하거나 나란히 딱 붙은 물체도 셀 수 있어요. 크기가 제각각인 물체에는 끄세요.")


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


def li_threshold(img, tol=0.5):
    # [추가] Li 임계값: 물체/배경 평균을 반복해서 갱신하며 cross-entropy를 최소화
    x = img.astype(float).ravel()
    t = x.mean()
    for _ in range(100):
        fg, bg = x[x > t], x[x <= t]
        if fg.size == 0 or bg.size == 0:
            break
        m_f, m_b = max(fg.mean(), 1e-6), max(bg.mean(), 1e-6)
        t_new = (m_b - m_f) / (np.log(m_b) - np.log(m_f)) if m_f != m_b else t
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new
    return t


def solidity(region):
    # [추가] 면적 / 볼록껍질 면적. 1에 가까우면 볼록한 단일 물체, 작으면 여러 물체가 붙은 오목한 덩어리
    ys, xs = np.nonzero(region)
    if len(ys) < 3:
        return 1.0
    pts = np.concatenate([np.c_[xs + dx, ys + dy] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
    try:
        return len(ys) / ConvexHull(pts).volume  # 2D에서 volume = 넓이
    except Exception:
        return 1.0


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
    # [추가] 오목한 덩어리(= 여러 물체가 붙은 것)만 distance transform 기반으로 분리.
    #        립스틱처럼 길쭉해도 볼록한 단일 물체는 solidity가 높아서 그대로 둠
    out = np.zeros_like(labels)
    nxt = 1
    for i, sl in enumerate(nd.find_objects(labels), 1):
        if sl is None:
            continue
        sub = np.pad(labels[sl] == i, 1)
        area = int(sub.sum())
        parts = None
        if solidity(sub) < solidity_th:
            dist = nd.distance_transform_edt(sub)
            peaks = (dist == nd.maximum_filter(dist, size=2 * min_dist + 1)) & (dist > 0.3 * dist.max())
            markers, nm = nd.label(peaks)
            if nm > 1:
                # 물체 픽셀마다 가장 가까운 중심(marker)에 배정 → 중심끼리의 경계에서 덩어리가 나뉨
                _, (iy, ix) = nd.distance_transform_edt(markers == 0, return_indices=True)
                ws = markers[iy, ix] * sub
                areas = np.bincount(ws[ws > 0].ravel(), minlength=nm + 1)[1:]
                if areas.min() >= 0.1 * area:            # 너무 작은 조각이 생기면 과분할로 보고 취소
                    parts = [ws == k for k in range(1, nm + 1)]
        if parts is None:
            parts = [sub]
        for part in parts:
            out[sl][part[1:-1, 1:-1]] = nxt
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
    d = f_denoised.astype(int) - bg.astype(int)
    if polarity == "밝다":
        d = np.clip(d, 0, None)
    elif polarity == "어둡다":
        d = np.clip(-d, 0, None)
    f_th = np.abs(d).clip(0, 255).astype(np.uint8)
else:
    f_th = f_denoised

if th_mode == "자동":
    # [변경] 배경 보정 영상은 0 근처에 큰 봉우리 + 긴 꼬리 모양이라 Otsu(봉우리 2개 가정)는 너무 높게 잡음
    #        → 이런 분포에 강한 Li 방법 사용. 배경 보정을 끄면 기존 Otsu
    if bg_corr:
        th_val = int(li_threshold(f_th) * th_scale)
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
    # [변경] 서버에 한글 폰트가 없어 그림 제목의 한글이 □로 깨짐 → 영어로 표시
    show(f_denoised, title=f"denoised ({ {'Median': 'median', 'Gaussian': 'gaussian'}.get(denoise, 'none') })",
         colorbar=True)

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
if area_est:
    unit = np.median(areas)                                 # 물체 1개의 대표 면적
    est = np.maximum(1, np.round(areas / unit)).astype(int)  # 덩어리마다 몇 개인지
else:
    est = np.ones(nums, dtype=int)
total = int(est.sum())

cc = pd.DataFrame({"index": idx, "number of pixels": areas})
if area_est:
    cc["추정 개수"] = est

if area_est and total != nums:
    st.subheader(f"4. Connected components — 객체 수: **{total}** (component {nums}개, 면적 보정)")
else:
    st.subheader(f"4. Connected components — 객체 수: **{total}**")
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
    for lab, k, (y, x) in zip(idx, est, nd.center_of_mass(labels > 0, labels, idx)):
        ax.text(x, y, f"{lab} (x{k})" if k > 1 else str(lab), color="white", fontsize=9, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
with c2:
    st.dataframe(cc, hide_index=True, use_container_width=True, height=420)
    st.download_button("CSV 다운로드", cc.to_csv(index=False).encode("utf-8-sig"),
                       "components.csv", "text/csv")

# [추가] 지금 상황 고르기: 결과를 보고 해당하는 상황을 누르면 관련 값이 한 단계씩 자동 조정됨
st.markdown("#### 지금 상황 고르기")
st.caption("원본과 결과를 비교해서 해당하는 상황을 누르세요. 누를 때마다 관련 값이 한 단계씩 조정돼요. "
           "여러 번 눌러도 되고, 여러 상황을 차례로 눌러도 돼요.")
SITUATIONS = {
    "여러 물체가 하나로 붙어 있어요": "merged",
    "한 물체가 여러 조각으로 쪼개졌어요": "split",
    "배경이나 바닥 무늬까지 물체로 잡혀요": "too_much",
    "물체가 덜 잡혀요 (빠지거나 일부만 잡힘)": "too_little",
    "작은 잡티가 객체로 세어져요": "specks",
    "얇은 물체가 사라졌어요": "thin_lost",
}


def _clip(key, value, lo, hi):
    st.session_state[key] = type(DEFAULTS[key])(min(hi, max(lo, value)))


def adjust(kind):
    ss, before = st.session_state, dict(st.session_state)
    if kind == "merged":
        _clip("open_k", ss["open_k"] + 4, 0, 51)
        _clip("close_k", ss["close_k"] - 4, 0, 51)
        ss["do_split"] = True
        _clip("solidity_th", round(ss["solidity_th"] + 0.03, 2), 0.5, 1.0)
        _clip("min_dist", ss["min_dist"] - 3, 3, 100)
    elif kind == "split":
        _clip("close_k", ss["close_k"] + 4, 0, 51)
        _clip("solidity_th", round(ss["solidity_th"] - 0.03, 2), 0.5, 1.0)
        _clip("min_dist", ss["min_dist"] + 5, 3, 100)
    elif kind == "too_much":
        if ss["th_mode"] == "자동":
            _clip("th_scale", round(ss["th_scale"] + 0.15, 2), 0.3, 2.0)
        else:
            _clip("th_val", ss["th_val"] + 5, 0, 255)
        if ss["denoise"] == "Median":
            _clip("med_k", ss["med_k"] + 2, 1, 21)
    elif kind == "too_little":
        if ss["th_mode"] == "자동":
            _clip("th_scale", round(ss["th_scale"] - 0.15, 2), 0.3, 2.0)
        else:
            _clip("th_val", ss["th_val"] - 5, 0, 255)
        _clip("close_k", ss["close_k"] + 2, 0, 51)
    elif kind == "specks":
        _clip("min_pct", round(ss["min_pct"] + 0.05, 2), 0.0, 2.0)
        _clip("open_k", ss["open_k"] + 2, 0, 51)
    elif kind == "thin_lost":
        _clip("open_k", ss["open_k"] - 2, 0, 51)
        _clip("min_pct", round(ss["min_pct"] / 2, 2), 0.0, 2.0)
    names = {"open_k": "Opening", "close_k": "Closing", "do_split": "분리", "solidity_th": "분리 기준",
             "min_dist": "중심 간격", "th_scale": "임계값 배율", "th_val": "임계값", "med_k": "Median 크기",
             "min_pct": "최소 면적"}
    changed = [f"{names[k]} {before.get(k)} → {ss[k]}" for k in names if before.get(k) != ss.get(k)]
    ss["last_change"] = ", ".join(changed) if changed else "더 조정할 수 있는 값이 없어요 (이미 한계값이에요)."


cols = st.columns(3)
for i, (label, kind) in enumerate(SITUATIONS.items()):
    cols[i % 3].button(label, on_click=adjust, args=(kind,), use_container_width=True)
if kind_msg := st.session_state.get("last_change"):
    st.info(f"**조정됨** {kind_msg}")
st.caption("배경·무늬까지 잡히면 먼저 사이드바 3번의 **물체가 배경보다…** 방향이 맞는지 확인하세요. "
           "물체가 빽빽한 사진이면 **배경 보정**을 끄는 것도 방법이에요.")
st.button("처음 값으로 되돌리기", on_click=apply_preset)

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
