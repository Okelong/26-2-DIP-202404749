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
st.caption("사진을 올리고 왼쪽 질문에 답한 뒤, 결과를 보고 '지금 상황 고르기'로 조정해 보세요.")

# ---------------- 기본값 / 사진에 대한 질문 ----------------
DEFAULTS = dict(max_side=1000, denoise="Median", med_k=5, sigma=2.0,
                direction="어둡다 (f < th)", th_mode="Otsu 자동", th_val=150, th_offset=0, bg_margin=20,
                close_k=9, fill=True, open_k=5, min_pct=0.05, connectivity="8-연결",
                area_est=False)
# 사진에 대한 질문: 일반 사용자도 답할 수 있는 질문으로 슬라이더 값을 정함
QUESTIONS = {
    "q_bg": ("배경은 어떤 색인가요?", ["밝다 (흰색, 밝은 회색)", "어둡다 (검정, 짙은 회색)"]),
    "q_texture": ("배경에 무늬나 결이 있나요? (바닥 틈, 천 질감 등)", ["없다 (매끈함)", "있다"]),
    "q_similar": ("배경과 밝기가 비슷한 물체가 섞여 있나요? (흰 배경 위 흰 물체 등)", ["없다", "있다"]),
    "q_small": ("물체 크기는 어떤가요?", ["보통 (손가락 마디 이상)", "아주 작다 (점처럼 흩어짐)"]),
    "q_touch": ("물체끼리 붙어 있나요?", ["대부분 떨어져 있다", "붙어 있는 곳이 많다"]),
    "q_same": ("물체들의 크기가 모두 비슷한가요? (동전, 레고, 면봉 등)", ["제각각이다", "거의 같다"]),
}
for k, (_, opts) in QUESTIONS.items():
    st.session_state.setdefault(k, opts[0])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_answers():
    """질문 답 → 슬라이더 값. 지금까지 테스트한 사진들(화장품, 은행잎, 레고 등)로 정한 규칙"""
    ss = st.session_state
    bright_bg = ss["q_bg"].startswith("밝")
    texture = ss["q_texture"].startswith("있")
    similar = ss["q_similar"].startswith("있")
    small = ss["q_small"].startswith("아주")
    touch = ss["q_touch"].startswith("붙")
    same = ss["q_same"].startswith("거의")

    cfg = dict(DEFAULTS)
    # 밝은 배경이면 물체가 더 어둡고, 어두운 배경이면 물체가 더 밝음
    cfg["direction"] = "어둡다 (f < th)" if bright_bg else "밝다 (f > th)"
    # 임계값: 배경과 비슷한 물체가 있으면 배경 바로 옆에서 자름, 아주 작은 밝은 물체는 높게 직접 지정
    if similar:
        cfg["th_mode"] = "배경 기준 자동"
    elif small and not bright_bg:
        cfg.update(th_mode="직접 지정", th_val=170)
    # 무늬가 있으면 노이즈 제거를 세게 (단, 아주 작은 물체는 같이 지워지므로 그대로)
    if texture and not small:
        cfg["med_k"] = 7
    # 모폴로지
    cfg["close_k"] = 3 if (small or touch or texture) else 5
    cfg["open_k"] = 0 if small else (3 if texture else 5)
    if touch and not small:
        cfg["open_k"] += 4
    # 최소 면적
    cfg["min_pct"] = 0.004 if small else (0.02 if texture else 0.05)
    # 크기가 같은 물체가 붙어 있으면 면적으로 개수 보정
    cfg["area_est"] = same and touch
    for k, v in cfg.items():
        if k != "max_side":
            ss[k] = v
    ss.pop("last_change", None)


if "answers_applied" not in st.session_state:  # 처음 열었을 때 기본 답 기준 값으로 시작
    apply_answers()
    st.session_state["answers_applied"] = True


# ---------------- Sidebar: 파라미터 ----------------
cap = st.sidebar.caption  # 설명을 물음표 대신 화면에 바로 표시

st.sidebar.header("0. 사진에 대해 알려주세요")
cap("사진을 보고 아래 질문에 답하면, 1~5번 값이 그 사진에 맞게 **자동으로 설정**돼요. "
    "보통은 여기만 답하고 결과를 확인하면 돼요.")
for key, (question, opts) in QUESTIONS.items():
    st.sidebar.radio(question, opts, key=key, on_change=apply_answers)
st.sidebar.divider()
cap("아래 1~5번은 자동으로 정해진 값이에요. 결과가 이상할 때만 직접 바꾸거나, "
    "결과 아래 **지금 상황 고르기** 버튼을 쓰세요.")

st.sidebar.header("1. 이미지")
cap("**기능** 사진을 grayscale로 불러오고, 빠른 처리를 위해 긴 변을 정해진 크기로 줄여요.")
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
max_side = st.sidebar.slider("처리용 최대 변 길이 (px)", 256, 4000, step=64, key="max_side")
cap("**언제** 보통은 그대로 두세요. 바꾸면 아래 크기 값들의 의미도 같이 바뀌어요.")

st.sidebar.header("2. 노이즈 제거")
cap("**기능** 사진의 자글자글한 잡티를 흐리게 지워요.\n\n"
    "**언제** 바닥 표면이 오돌토돌해서 잡티가 많이 잡히면 크기를 키우세요. "
    "너무 키우면 작은 물체와 경계가 뭉개져요.")
denoise = st.sidebar.selectbox("필터", ["Median", "Gaussian", "없음"], key="denoise")
if denoise == "Median":
    med_k = st.sidebar.slider("Median kernel 크기 (홀수)", 1, 21, step=2, key="med_k")
elif denoise == "Gaussian":
    sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, step=0.5, key="sigma")

st.sidebar.header("3. Thresholding")
cap("**기능** 밝기 기준(임계값)으로 픽셀을 물체 / 배경으로 나눠 binary mask를 만들어요.")
direction = st.sidebar.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)"], key="direction")
cap("**언제** 흰 배경 위 물체면 '어둡다', 어두운 배경 위 밝은 물체면 '밝다'를 고르세요. "
    "바닥의 틈이나 그림자가 같이 잡히면 방향이 맞는지 먼저 확인하세요.")
th_mode = st.sidebar.radio("임계값 방식", ["Otsu 자동", "배경 기준 자동", "직접 지정"], key="th_mode")
if th_mode == "배경 기준 자동":
    bg_margin = st.sidebar.slider("배경에서 떨어진 정도", 3, 80, step=1, key="bg_margin")
    cap("히스토그램에서 가장 많은 밝기(= 배경)를 찾고, 거기서 이만큼 떨어진 곳을 임계값으로 써요. "
        "배경과 비슷한 밝은 물체까지 잡을 때 써요. 값이 작을수록 배경에 가까운 물체까지 잡혀요.")
elif th_mode == "직접 지정":
    th_val = st.sidebar.slider("임계값", 0, 255, key="th_val")
    cap("히스토그램의 빨간 선 위치예요. 선 기준으로 물체 쪽과 배경 쪽이 나뉘어요. "
        "배경과 비슷한 밝은 물체까지 잡으려면 배경 봉우리 바로 옆까지 선을 옮기세요.")
else:
    th_offset = st.sidebar.slider("Otsu 값 보정 (±)", -60, 60, step=1, key="th_offset")
    cap("Otsu가 구한 값에 더하거나 빼요. 물체가 덜 잡히거나 배경이 잡히면 조금씩 움직여 보세요.")

st.sidebar.header("4. 모폴로지")
cap("**기능** binary mask의 모양을 다듬어요. Closing → 구멍 채우기 → Opening 순서로 적용해요.")
close_k = st.sidebar.slider("Closing 크기 (조각 잇기)", 0, 51, step=2, key="close_k")
cap("**언제** 한 물체가 여러 조각으로 세어지면 키우세요. 옆 물체와 붙어 버리면 줄이세요.")
fill = st.sidebar.checkbox("Hole filling", key="fill")
cap("**언제** 물체 안쪽에 구멍(글씨, 반사광)이 뚫려 있으면 켜 두세요.")
open_k = st.sidebar.slider("Opening 크기 (얇은 연결 끊기)", 0, 51, step=2, key="open_k")
cap("**언제** 물체끼리 그림자나 가는 부분으로 이어져 있으면 키우세요. "
    "가장 얇은 물체보다 크게 잡으면 그 물체가 사라져요.")

st.sidebar.header("5. Connected component")
cap("**기능** 이어진 덩어리마다 번호를 매기고 픽셀 수를 세요. 너무 작은 덩어리는 잡티로 보고 버려요.")
min_pct = st.sidebar.slider("최소 면적 (영상 대비 %)", 0.0, 1.0, step=0.001, format="%.3f", key="min_pct")
cap("**언제** 작은 잡티가 객체로 세어지면 올리세요. 작은 물체가 사라지면 내리세요.")
connectivity = st.sidebar.radio("연결성", ["8-연결", "4-연결"], horizontal=True, key="connectivity")
cap("8-연결은 대각선으로 닿은 픽셀도 같은 덩어리로 봐요. 보통 8-연결을 쓰세요.")
area_est = st.sidebar.checkbox("면적으로 개수 보정 (같은 종류 물체일 때)", key="area_est")
cap("**언제** 면봉처럼 크기가 같은 물체만 있을 때 켜세요. 덩어리 픽셀 수 ÷ 물체 1개 픽셀 수(중앙값)로 "
    "붙은 물체 개수를 셉니다. 크기가 제각각인 물체에는 끄세요.")


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
    im = ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
    plt.close(fig)


# ---------------- 메인 ----------------
with st.expander("📖 사용 방법 (처음이면 읽어 주세요)", expanded=uploaded is None):
    st.markdown(
        "1. **왼쪽에서 사진을 올리세요.**\n"
        "2. **'0. 사진에 대해 알려주세요'의 질문에 답하세요.** 사진을 보고 고르기만 하면 "
        "나머지 값은 자동으로 정해져요.\n"
        "3. **'4. Connected components'에서 결과를 확인하세요.** 물체마다 색과 번호가 붙어요. "
        "원본과 비교해서 빠진 물체, 하나로 붙은 물체, 쪼개진 물체가 있는지 보세요.\n"
        "4. **이상한 곳이 있으면 결과 아래 '지금 상황 고르기'에서 해당 버튼을 누르세요.** "
        "누를 때마다 값이 한 단계씩 조정돼요. 결과가 나빠지면 '처음 값으로 되돌리기'를 누르세요.\n\n"
        "**결과 보는 요령**\n"
        "- **3번 binary mask**에서 흰색이 물체로 인식된 부분이에요. 물체가 하얗게, 배경이 까맣게 나와야 해요.\n"
        "- **히스토그램의 빨간 선**이 임계값이에요. 선 한쪽이 물체, 반대쪽이 배경으로 나뉘어요.\n"
        "- 번호 옆에 **(x2)**가 붙으면 붙어 있는 물체를 면적으로 계산해서 2개로 센 거예요."
    )

if uploaded is None:
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드하세요.")
    st.stop()

f = load_gray(uploaded.getvalue(), max_side)
param = med_k if denoise == "Median" else (sigma if denoise == "Gaussian" else 0)
with st.spinner("노이즈 제거 중..."):
    f_denoised = do_denoise(f, denoise, param)

if th_mode == "Otsu 자동":
    th_val = int(np.clip(otsu(f_denoised) + th_offset, 0, 255))
    st.sidebar.success(f"Otsu 임계값: {th_val}")
elif th_mode == "배경 기준 자동":
    bg_level = int(np.argmax(np.bincount(f_denoised.ravel(), minlength=256)))  # 히스토그램 최빈값 = 배경 밝기
    th_val = int(np.clip(bg_level - bg_margin if direction.startswith("어둡") else bg_level + bg_margin, 0, 255))
    st.sidebar.success(f"배경 밝기 {bg_level} → 임계값 {th_val}")

# 1~2. 원본 / 노이즈 제거
st.subheader("1–2. 입력 영상 & 노이즈 제거")
c1, c2 = st.columns(2)
with c1:
    show(f, title=f"input (grayscale) {f.shape[1]}×{f.shape[0]}", colorbar=True)
with c2:
    # 서버에 한글 폰트가 없어 그림 제목의 한글이 □로 깨지므로 영어로 표시
    show(f_denoised, title=f"denoised ({ {'Median': 'median', 'Gaussian': 'gaussian'}.get(denoise, 'none') })",
         colorbar=True)

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
# 순서: closing(조각 잇기) → fill → opening(얇은 연결 끊기)
# opening 다음 closing을 하면, 끊어 놓은 물체 사이를 closing이 다시 붙여 버림
if close_k > 0:
    f_mask = nd.binary_closing(f_mask, structure=np.ones((close_k, close_k)))
if fill:
    f_mask = nd.binary_fill_holes(f_mask)
if open_k > 0:
    f_mask = nd.binary_opening(f_mask, structure=np.ones((open_k, open_k)))
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
min_area = int(min_pct / 100 * f.size)
keep = areas >= min_area
if not keep.all():
    lut = np.zeros(nums + 1, dtype=labels.dtype)
    lut[idx[keep]] = np.arange(1, keep.sum() + 1)
    labels = lut[labels]
    areas = areas[keep]
    nums = int(keep.sum())
    idx = np.arange(1, nums + 1)

if nums == 0:
    st.warning("최소 면적 조건을 만족하는 객체가 없습니다.")
    st.stop()

# 면적으로 개수 보정: 덩어리 픽셀 수 ÷ 물체 1개 픽셀 수(중앙값)
if area_est:
    est = np.maximum(1, np.round(areas / np.median(areas))).astype(int)
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
    # 객체마다 랜덤 색 + 번호 표시 (Spectral_r는 이웃 번호끼리 색이 비슷해서 구분이 어려움)
    rng = np.random.RandomState(0)
    colors = rng.rand(nums + 1, 3) * 0.8 + 0.2
    colors[0] = 0
    fig, ax = plt.subplots(figsize=(8, 8 * f.shape[0] / f.shape[1]))
    ax.imshow(f, cmap="gray")
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap=plt.matplotlib.colors.ListedColormap(colors),
              alpha=0.6, interpolation="nearest", vmin=0, vmax=nums)
    for lab, k, (y, x) in zip(idx, est, nd.center_of_mass(labels > 0, labels, idx)):
        ax.text(x, y, f"{lab} (x{k})" if k > 1 else str(lab), color="white",
                fontsize=9 if nums <= 50 else 6, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
with c2:
    st.dataframe(cc, hide_index=True, use_container_width=True, height=420)
    st.download_button("CSV 다운로드", cc.to_csv(index=False).encode("utf-8-sig"),
                       "components.csv", "text/csv")

# ---------------- 지금 상황 고르기 ----------------
# 결과를 보고 해당하는 상황을 누르면 관련 슬라이더 값이 한 단계씩 바뀜 (새 알고리즘 없음)
st.markdown("#### 지금 상황 고르기")
st.caption("원본과 결과를 비교해서 해당하는 상황을 누르세요. 누를 때마다 관련 값이 한 단계씩 조정돼요.")
SITUATIONS = {
    "여러 물체가 하나로 붙어 있어요": "merged",
    "한 물체가 여러 조각으로 쪼개졌어요": "split",
    "배경이나 바닥 무늬까지 물체로 잡혀요": "too_much",
    "물체가 덜 잡혀요 (빠지거나 일부만 잡힘)": "too_little",
    "작은 잡티가 객체로 세어져요": "specks",
    "작거나 얇은 물체가 사라졌어요": "thin_lost",
}
NAMES = {"open_k": "Opening", "close_k": "Closing", "th_offset": "Otsu 보정", "th_val": "임계값",
         "bg_margin": "배경에서 떨어진 정도",
         "med_k": "Median 크기", "min_pct": "최소 면적"}


def _set(key, value, lo, hi):
    st.session_state[key] = type(DEFAULTS[key])(min(hi, max(lo, value)))


def adjust(kind):
    ss = st.session_state
    before = {k: ss.get(k) for k in NAMES}
    # 임계값을 '물체가 더/덜 잡히는 쪽'으로 움직이는 방향 (어두운 물체면 값을 올려야 더 잡힘)
    more = 1 if ss["direction"].startswith("어둡") else -1
    th_key, th_lo, th_hi = {"Otsu 자동": ("th_offset", -60, 60), "직접 지정": ("th_val", 0, 255),
                            "배경 기준 자동": ("bg_margin", 3, 80)}[ss["th_mode"]]
    if ss["th_mode"] == "배경 기준 자동":
        more = -1  # 배경 기준: 간격을 줄일수록 물체가 더 잡힘
    if kind == "merged":
        _set("open_k", ss["open_k"] + 4, 0, 51)
        _set("close_k", ss["close_k"] - 4, 0, 51)
    elif kind == "split":
        _set("close_k", ss["close_k"] + 4, 0, 51)
    elif kind == "too_much":
        _set(th_key, ss[th_key] - 8 * more, th_lo, th_hi)
        if ss["denoise"] == "Median":
            _set("med_k", ss["med_k"] + 2, 1, 21)
    elif kind == "too_little":
        _set(th_key, ss[th_key] + 8 * more, th_lo, th_hi)
        _set("close_k", ss["close_k"] + 2, 0, 51)
    elif kind == "specks":
        _set("min_pct", round(ss["min_pct"] * 2 + 0.002, 3), 0.0, 1.0)
        _set("open_k", ss["open_k"] + 2, 0, 51)
    elif kind == "thin_lost":
        _set("open_k", ss["open_k"] - 2, 0, 51)
        _set("min_pct", round(ss["min_pct"] / 2, 3), 0.0, 1.0)
    changed = [f"{NAMES[k]} {before[k]} → {ss[k]}" for k in NAMES if before[k] != ss.get(k)]
    ss["last_change"] = ", ".join(changed) if changed else "더 조정할 수 있는 값이 없어요 (이미 한계값이에요)."


cols = st.columns(3)
for i, (label, kind) in enumerate(SITUATIONS.items()):
    cols[i % 3].button(label, on_click=adjust, args=(kind,), use_container_width=True)
if msg := st.session_state.get("last_change"):
    st.info(f"**조정됨** {msg}")
st.caption("배경·무늬까지 잡히면 먼저 사이드바 3번의 **물체가 배경보다…** 방향이 맞는지 확인하세요.")
st.button("처음 값으로 되돌리기 (질문 답 기준)", on_click=apply_answers)

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
