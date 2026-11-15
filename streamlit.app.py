import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Object Counting", layout="wide")

# =====================================================================
# 기본값 / 사진에 대한 질문 → 세부 값
# =====================================================================
DEFAULTS = dict(max_side=1000, denoise="Median", med_k=5, sigma=2.0,
                direction="어둡다 (f < th)", th_mode="Otsu 자동", th_val=150, th_offset=0, bg_margin=20,
                tophat=False, tophat_k=51,
                close_k=5, fill=True, open_k=5, erode_mode="끄기", erode_k=21,
                min_pct=0.05, connectivity="8-연결", area_est=False)

QUESTIONS = {
    "q_bg": ("배경은 어떤 색인가요?", ["밝다 (흰색, 밝은 회색)", "어둡다 (검정, 짙은 회색)"]),
    "q_texture": ("배경에 무늬나 결이 있나요? (바닥 틈, 천 질감 등)", ["없다 (매끈함)", "있다"]),
    "q_light": ("조명이 고르지 않나요? (한쪽만 햇빛이 비치거나 그림자가 짐)", ["고르다", "고르지 않다"]),
    "q_similar": ("물체 전체가 배경과 거의 같은 색인 것이 섞여 있나요? (흰 종이 위 흰 블록 등)", ["없다", "있다"]),
    "q_small": ("물체 크기는 어떤가요?", ["보통", "아주 작다 (점처럼 흩어짐)"]),
    "q_touch": ("물체끼리 붙어 있나요?", ["대부분 떨어져 있다", "붙어 있는 곳이 많다"]),
    "q_same": ("물체들의 크기가 모두 비슷한가요? (레고, 면봉 등)", ["제각각이다", "거의 같다"]),
}
for k, (_, opts) in QUESTIONS.items():
    st.session_state.setdefault(k, opts[0])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_answers():
    """질문 답 → 세부 값. 지금까지 테스트한 사진들(화장품, 은행잎, 레고, 동전)로 정한 규칙"""
    ss = st.session_state
    bright_bg = ss["q_bg"].startswith("밝")
    texture = ss["q_texture"].startswith("있")
    uneven = ss["q_light"].startswith("고르지")
    similar = ss["q_similar"].startswith("있")
    small = ss["q_small"].startswith("아주")
    touch = ss["q_touch"].startswith("붙")
    same = ss["q_same"].startswith("거의")

    cfg = dict(DEFAULTS)
    cfg["direction"] = "어둡다 (f < th)" if bright_bg else "밝다 (f > th)"
    if similar:                      # 배경과 비슷한 물체 → 배경 바로 옆에서 자르고 테두리를 hole filling으로 채움
        cfg["th_mode"] = "배경 기준 자동"
    elif small and not bright_bg:    # 어두운 바닥 위 아주 작은 밝은 물체 → 높게 직접 지정
        cfg.update(th_mode="직접 지정", th_val=170)
    if texture and not small:
        cfg["med_k"] = 7
    if uneven:                       # 조명이 고르지 않으면 top-hat으로 배경 밝기 차이를 없앤 뒤 Otsu
        cfg.update(tophat=True, tophat_k=31 if small else 51, th_mode="Otsu 자동")
    cfg["close_k"] = 3 if (small or touch or texture) else 5
    cfg["open_k"] = 0 if small else (3 if texture else 5)
    cfg["min_pct"] = 0.004 if small else (0.02 if texture else 0.05)
    if touch and not similar and not small:
        cfg["erode_mode"] = "자동"   # 붙은 물체는 erosion으로 떼어냄 (동전 등)
    cfg["area_est"] = same and touch  # 크기가 같은 물체가 붙어 있으면 면적으로 개수 보정 (레고 등)
    for k, v in cfg.items():
        if k != "max_side":
            ss[k] = v
    ss.pop("last_change", None)


if "answers_applied" not in st.session_state:
    apply_answers()
    st.session_state["answers_applied"] = True

# =====================================================================
# 지금 상황 고르기: 결과를 보고 버튼을 누르면 관련 값이 한 단계씩 조정됨
# =====================================================================
SITUATIONS = {
    "여러 물체가 하나로 붙어 있어요": "merged",
    "한 물체가 여러 조각으로 쪼개졌어요": "split",
    "배경이나 바닥 무늬까지 잡혀요": "too_much",
    "물체가 덜 잡혀요": "too_little",
    "작은 잡티가 세어져요": "specks",
    "작거나 얇은 물체가 사라졌어요": "thin_lost",
}
NAMES = {"open_k": "Opening", "close_k": "Closing", "erode_mode": "물체 떼기", "erode_k": "Erosion 크기",
         "th_offset": "Otsu 보정", "th_val": "임계값", "bg_margin": "배경 간격", "med_k": "Median 크기",
         "min_pct": "최소 면적"}


def _set(key, value, lo, hi):
    st.session_state[key] = type(DEFAULTS[key])(min(hi, max(lo, value)))


def adjust(kind):
    ss = st.session_state
    before = {k: ss.get(k) for k in NAMES}
    more = 1 if (ss["direction"].startswith("어둡") and not ss["tophat"]) else -1  # 임계값을 '물체가 더 잡히는 쪽'으로
    th_key, lo, hi = {"Otsu 자동": ("th_offset", -60, 60), "직접 지정": ("th_val", 0, 255),
                      "배경 기준 자동": ("bg_margin", 3, 80)}[ss["th_mode"]]
    if ss["th_mode"] == "배경 기준 자동":
        more = -1                                              # 간격을 줄일수록 물체가 더 잡힘
    if kind == "merged":
        if ss["erode_mode"] == "끄기":
            ss["erode_mode"] = "자동"
        elif ss["erode_mode"] == "직접 지정":
            _set("erode_k", ss["erode_k"] + 6, 0, 81)
        else:
            _set("open_k", ss["open_k"] + 4, 0, 51)
        _set("close_k", ss["close_k"] - 2, 0, 51)
    elif kind == "split":
        if ss["erode_mode"] == "직접 지정":
            _set("erode_k", ss["erode_k"] - 6, 0, 81)
        _set("close_k", ss["close_k"] + 4, 0, 51)
    elif kind == "too_much":
        _set(th_key, ss[th_key] - 8 * more, lo, hi)
        if ss["denoise"] == "Median":
            _set("med_k", ss["med_k"] + 2, 1, 21)
    elif kind == "too_little":
        _set(th_key, ss[th_key] + 8 * more, lo, hi)
        _set("close_k", ss["close_k"] + 2, 0, 51)
    elif kind == "specks":
        _set("min_pct", round(ss["min_pct"] * 2 + 0.002, 3), 0.0, 1.0)
        _set("open_k", ss["open_k"] + 2, 0, 51)
    elif kind == "thin_lost":
        _set("open_k", ss["open_k"] - 2, 0, 51)
        _set("min_pct", round(ss["min_pct"] / 2, 3), 0.0, 1.0)
    changed = [f"{NAMES[k]} {before[k]} → {ss[k]}" for k in NAMES if before[k] != ss.get(k)]
    ss["last_change"] = ", ".join(changed) if changed else "더 조정할 수 있는 값이 없어요."


# =====================================================================
# 왼쪽: 조작 (업로드 → 질문 → 상황 고르기 → 세부 설정)
# =====================================================================
sb = st.sidebar
sb.title("🔍 Object Counting")
with sb.expander("📖 사용 방법"):
    st.markdown(
        "1. **사진을 올리세요.**\n"
        "2. **사진을 보면서 질문에 답하세요.** 답을 고르면 세부 값이 자동으로 정해지고 "
        "오른쪽 결과가 바로 바뀌어요.\n"
        "3. **오른쪽 결과를 원본과 비교하세요.** 빠진 물체, 하나로 붙은 물체, 쪼개진 물체가 있는지 보세요.\n"
        "4. **이상하면 '결과가 이상하면'에서 해당 버튼을 누르세요.** 누를 때마다 값이 한 단계씩 조정돼요.\n\n"
        "**결과 보는 요령**\n"
        "- **binary mask**: 흰색이 물체로 인식된 부분이에요.\n"
        "- **히스토그램의 빨간 선**: 임계값이에요. 선 한쪽이 물체, 반대쪽이 배경이에요.\n"
        "- 번호 옆 **(x2)**: 붙은 물체를 면적으로 계산해서 2개로 셌다는 뜻이에요."
    )

sb.header("① 사진 올리기")
uploaded = sb.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                            label_visibility="collapsed")

if uploaded is not None:
    sb.header("② 사진에 대해 알려주세요")
    sb.caption("답을 고르면 세부 값이 자동으로 정해져요.")
    for key, (question, opts) in QUESTIONS.items():
        sb.radio(question, opts, key=key, on_change=apply_answers)

    sb.header("③ 결과가 이상하면")
    sb.caption("오른쪽 결과를 보고 해당하는 상황을 누르세요. 누를 때마다 한 단계씩 조정돼요.")
    for label, kind in SITUATIONS.items():
        sb.button(label, on_click=adjust, args=(kind,), use_container_width=True)
    if msg := st.session_state.get("last_change"):
        sb.info(f"**조정됨** {msg}")
    sb.button("↺ 처음 값으로 되돌리기", on_click=apply_answers, use_container_width=True)

# 세부 설정: 질문에 답하면 자동으로 정해짐. 직접 바꿀 때만 펼침
with sb.expander("⚙️ 세부 설정 (자동으로 정해져요)"):
    cap = st.caption
    st.markdown("**1. 이미지**")
    max_side = st.slider("처리용 최대 변 길이 (px)", 256, 4000, step=64, key="max_side")
    cap("빠른 처리를 위해 사진을 줄여요. 보통은 그대로 두세요.")

    st.markdown("**2. 노이즈 제거**")
    denoise = st.selectbox("필터", ["Median", "Gaussian", "없음"], key="denoise")
    if denoise == "Median":
        med_k = st.slider("Median kernel 크기 (홀수)", 1, 21, step=2, key="med_k")
    elif denoise == "Gaussian":
        sigma = st.slider("Gaussian sigma", 0.0, 10.0, step=0.5, key="sigma")
    cap("바닥 무늬나 잡티가 많이 잡히면 키우세요. 너무 키우면 작은 물체가 뭉개져요.")
    tophat = st.checkbox("조명 보정 (top-hat)", key="tophat")
    if tophat:
        tophat_k = st.slider("조명 보정 크기 (물체보다 크게)", 11, 151, step=2, key="tophat_k")
    cap("조명이 고르지 않을 때 켜세요. 물체보다 큰 크기로 opening(밝은 물체) 또는 closing(어두운 물체)을 해서 "
        "배경 밝기를 구하고, 원본과의 차이만 남겨요. 그러면 물체만 밝게 남아요.")

    st.markdown("**3. Thresholding**")
    direction = st.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)"], key="direction")
    th_mode = st.radio("임계값 방식", ["Otsu 자동", "배경 기준 자동", "직접 지정"], key="th_mode")
    if th_mode == "Otsu 자동":
        th_offset = st.slider("Otsu 값 보정 (±)", -60, 60, step=1, key="th_offset")
        cap("Otsu가 구한 값에 더하거나 빼요.")
    elif th_mode == "배경 기준 자동":
        bg_margin = st.slider("배경에서 떨어진 정도", 3, 80, step=1, key="bg_margin")
        cap("히스토그램에서 가장 많은 밝기(= 배경)에서 이만큼 떨어진 곳을 임계값으로 써요. "
            "작을수록 배경에 가까운 물체까지 잡혀요.")
    else:
        th_val = st.slider("임계값", 0, 255, key="th_val")
        cap("히스토그램의 빨간 선 위치예요.")

    st.markdown("**4. 모폴로지**")
    close_k = st.slider("Closing 크기 (조각 잇기)", 0, 51, step=2, key="close_k")
    cap("한 물체가 여러 조각이면 키우고, 옆 물체와 붙으면 줄이세요.")
    open_k = st.slider("Opening 크기 (얇은 연결 끊기)", 0, 51, step=2, key="open_k")
    cap("물체끼리 가늘게 이어져 있으면 키우세요. 가장 얇은 물체보다 크면 그 물체가 사라져요.")
    fill = st.checkbox("Hole filling (물체 안 구멍 채우기)", key="fill")
    erode_mode = st.radio("붙은 물체 떼기 (erosion)", ["끄기", "자동", "직접 지정"], horizontal=True,
                          key="erode_mode")
    if erode_mode == "직접 지정":
        erode_k = st.slider("Erosion 크기", 3, 81, step=2, key="erode_k")
    cap("동전처럼 빽빽하게 붙은 물체를 깎아서 떼어낸 뒤 셉니다. 자동은 물체 수가 가장 많이 나오는 크기를 찾아요. "
        "켜면 hole filling은 깎은 뒤에 적용돼요 (물체 사이 빈틈이 채워지지 않게).")

    st.markdown("**5. Connected component**")
    min_pct = st.slider("최소 면적 (영상 대비 %)", 0.0, 1.0, step=0.001, format="%.3f", key="min_pct")
    cap("작은 잡티가 세어지면 올리고, 작은 물체가 사라지면 내리세요.")
    connectivity = st.radio("연결성", ["8-연결", "4-연결"], horizontal=True, key="connectivity")
    area_est = st.checkbox("면적으로 개수 보정 (크기가 같은 물체일 때)", key="area_est")
    cap("덩어리 픽셀 수 ÷ 물체 1개 픽셀 수(중앙값)로 붙은 물체 개수를 셉니다.")


# =====================================================================
# 처리 함수
# =====================================================================
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


def label_cores(mask, ek, fill, structure, min_area):
    """erosion으로 깎은 뒤(ek=0이면 그대로) 구멍 채우고 labeling. 작은 조각은 제외하고 개수 반환"""
    core = nd.binary_erosion(mask, np.ones((ek, ek))) if ek > 0 else mask
    if fill:
        core = nd.binary_fill_holes(core)
    lab, n = nd.label(core, structure=structure)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return lab, core, int((areas >= max(1, min_area * 0.2)).sum())


def fig_img(img, title, cmap="gray"):
    fig, ax = plt.subplots(figsize=(5, 5 * img.shape[0] / img.shape[1]))
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def put(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# 오른쪽: 각 과정의 결과만
# =====================================================================
if uploaded is None:
    st.info("👈 왼쪽에서 사진을 올리세요.")
    st.stop()

f = load_gray(uploaded.getvalue(), max_side)
param = med_k if denoise == "Median" else (sigma if denoise == "Gaussian" else 0)
f_denoised = do_denoise(f, denoise, param)

# --- 조명 보정 (top-hat) ---
dark_obj = direction.startswith("어둡")
if tophat:
    se = (tophat_k, tophat_k)
    if dark_obj:   # black-hat: closing(배경) − 영상 → 어두운 물체가 밝게 남음
        f_th = (nd.grey_closing(f_denoised, size=se).astype(int) - f_denoised).clip(0, 255).astype(np.uint8)
    else:          # white top-hat: 영상 − opening(배경) → 밝은 물체만 남음
        f_th = (f_denoised.astype(int) - nd.grey_opening(f_denoised, size=se)).clip(0, 255).astype(np.uint8)
    obj_is_dark = False   # 보정 영상에서는 물체가 항상 밝음
else:
    f_th = f_denoised
    obj_is_dark = dark_obj

# --- 3. 임계값 ---
if th_mode == "Otsu 자동":
    th_val = int(np.clip(otsu(f_th) + th_offset, 0, 255))
elif th_mode == "배경 기준 자동":
    bg_level = int(np.argmax(np.bincount(f_th.ravel(), minlength=256)))  # 최빈값 = 배경 밝기
    th_val = int(np.clip(bg_level - bg_margin if obj_is_dark else bg_level + bg_margin, 0, 255))

f_mask = f_th < th_val if obj_is_dark else f_th > th_val
# 순서: closing(조각 잇기) → opening(얇은 연결 끊기) → [erosion] → hole filling
if close_k > 0:
    f_mask = nd.binary_closing(f_mask, structure=np.ones((close_k, close_k)))
if open_k > 0:
    f_mask = nd.binary_opening(f_mask, structure=np.ones((open_k, open_k)))

# --- 4. Connected components ---
structure = None if connectivity == "4-연결" else np.ones((3, 3))
min_area = int(min_pct / 100 * f.size)
ek = 0
if erode_mode == "자동":
    # erosion 크기를 바꿔 가며 물체 수가 가장 많이 나오는 크기를 고름
    counts = [(k, label_cores(f_mask, k, fill, structure, min_area)[2]) for k in range(3, 82, 4)]
    best = max(c for _, c in counts)
    ek = next(k for k, c in counts if c >= 0.97 * best)
elif erode_mode == "직접 지정":
    ek = erode_k
labels, core, _ = label_cores(f_mask, ek, fill, structure, min_area)
if ek > 0:
    # 깎인 부분을 다시 키워서 원래 물체 영역에 번호를 돌려줌 (dilation)
    region = f_mask | nd.binary_dilation(core, np.ones((ek, ek)))
    labels = nd.grey_dilation(labels, footprint=np.ones((ek + 2, ek + 2))) * region
    mask_shown = region
else:
    mask_shown = core
nums = int(labels.max())

# 작은 조각 제거 후 번호 다시 매기기
areas = np.bincount(labels.ravel(), minlength=nums + 1)[1:]
keep = areas >= min_area
lut = np.zeros(nums + 1, dtype=labels.dtype)
lut[1:][keep] = np.arange(1, keep.sum() + 1)
labels = lut[labels]
areas = areas[keep]
nums = int(keep.sum())
idx = np.arange(1, nums + 1)

if area_est and nums > 0:
    est = np.maximum(1, np.round(areas / np.median(areas))).astype(int)
else:
    est = np.ones(nums, dtype=int)
total = int(est.sum())

# --- 화면: 요약 ---
m1, m2, m3 = st.columns(3)
m1.metric("검출된 객체 수", total)
m2.metric("Connected component 수", nums)
m3.metric("임계값", th_val, help=f"방식: {th_mode}")

# --- 1~3단계 ---
st.subheader("1–3. 전처리 & Thresholding")
c = st.columns(5 if tophat else 4)
with c[0]:
    put(fig_img(f, f"1. input (gray) {f.shape[1]}x{f.shape[0]}"))
with c[1]:
    put(fig_img(f_denoised, f"2. denoised ({ {'Median': 'median', 'Gaussian': 'gaussian'}.get(denoise, 'none') })"))
if tophat:
    with c[2]:
        put(fig_img(f_th, f"2. {'black-hat' if dark_obj else 'top-hat'} ({tophat_k})"))
    c = c[1:]
with c[2]:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(f_th.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
    ax.set_yscale("log")
    ax.set_title("3. histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.legend()
    fig.tight_layout()
    put(fig)
with c[3]:
    put(fig_img(mask_shown, "3. binary mask" + (f" (erosion {ek})" if ek else "")))

# --- 4단계 ---
st.subheader(f"4. Connected components — {total}개" + (f" (component {nums}개 + 면적 보정)" if total != nums else ""))
if nums == 0:
    st.warning("검출된 객체가 없어요. 왼쪽 질문의 답(특히 배경 색)을 확인해 보세요.")
    st.stop()
if areas.max() > 0.3 * f.size:
    st.warning("하나의 덩어리가 화면의 30% 이상이에요. 여러 물체가 하나로 뭉쳤을 수 있어요 → "
               "왼쪽 ③에서 **'여러 물체가 하나로 붙어 있어요'**를 눌러 보세요.")
c1, c2 = st.columns([3, 1])
with c1:
    colors = np.random.RandomState(0).rand(nums + 1, 3) * 0.8 + 0.2
    colors[0] = 0
    fig, ax = plt.subplots(figsize=(10, 10 * f.shape[0] / f.shape[1]))
    ax.imshow(f, cmap="gray")
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap=plt.matplotlib.colors.ListedColormap(colors),
              alpha=0.6, interpolation="nearest", vmin=0, vmax=nums)
    for lab, k, (y, x) in zip(idx, est, nd.center_of_mass(labels > 0, labels, idx)):
        ax.text(x, y, f"{lab} (x{k})" if k > 1 else str(lab), color="white",
                fontsize=9 if nums <= 50 else 6, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, lw=0))
    ax.axis("off")
    fig.tight_layout()
    put(fig)
with c2:
    cc = pd.DataFrame({"index": idx, "number of pixels": areas})
    if area_est:
        cc["추정 개수"] = est
    st.dataframe(cc, hide_index=True, use_container_width=True, height=480)
    st.download_button("CSV 다운로드", cc.to_csv(index=False).encode("utf-8-sig"), "components.csv", "text/csv")

# --- 5단계 ---
i_max, i_min = int(np.argmax(areas)), int(np.argmin(areas))
st.subheader("5. 가장 큰 / 가장 작은 component")
st.write(f"- **the largest component**: index {idx[i_max]}, # of pixels: {areas[i_max]}\n"
         f"- **the smallest component**: index {idx[i_min]}, # of pixels: {areas[i_min]}")
c = st.columns(3)
with c[0]:
    put(fig_img(f, "input image"))
with c[1]:
    put(fig_img(labels == idx[i_max], "the largest component", cmap="Spectral_r"))
with c[2]:
    put(fig_img(labels == idx[i_min], "the smallest component", cmap="Spectral_r"))
