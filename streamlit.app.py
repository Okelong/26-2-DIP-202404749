import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # 컬러맵 에러 수정을 위해 추가
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Object Counting", layout="wide")

# =====================================================================
# 기본값 / 사진에 대한 질문 → 세부 값
# =====================================================================
DEFAULTS = dict(max_side=1000, denoise="Median", med_k=5, sigma=2.0,
                bg_sub=False, bg_k=51,
                direction="어둡다 (f < th)", th_mode="Otsu 자동", th_val=150, th_offset=0, bg_margin=20,
                use_edge=False, edge_scale=0.7,
                close_k=5, fill=True, open_k=5, erode_mode="끄기", erode_k=21,
                min_pct=0.05, connectivity="8-연결", area_est=False)

QUESTIONS = {
    "q_bg": ("배경은 어떤 색인가요?", ["밝다 (흰색, 밝은 회색)", "어둡다 (검정, 짙은 회색)"]),
    "q_texture": ("배경에 무늬나 결이 있나요? (바닥 틈, 천 질감 등)", ["없다 (매끈함)", "있다"]),
    "q_both": ("배경보다 밝은 물체와 어두운 물체가 섞여 있나요? (회색 바닥 위 흰 블록과 검은 블록 등)", ["아니다", "섞여 있다"]),
    "q_light": ("조명이 고르지 않나요? (한쪽만 햇빛이 비치거나 그림자가 짐)", ["고르다", "고르지 않다"]),
    "q_similar": ("물체 전체가 배경과 거의 같은 색인 것이 섞여 있나요? (흰 종이 위 흰 블록 등)", ["없다", "있다"]),
    "q_small": ("물체 크기는 어떤가요?", ["보통", "아주 작다 (점처럼 흩어짐)"]),
    "q_touch": ("물체끼리 붙어 있나요?", ["대부분 떨어져 있다", "붙어 있는 곳이 많다"]),
    "q_same": ("물체들의 크기가 모두 비슷한가요?", ["제각각이다", "거의 같다"]),
}
for k, (_, opts) in QUESTIONS.items():
    st.session_state.setdefault(k, opts[0])
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def answers_to_cfg(ans):
    """질문 답 → 세부 값. 테스트한 사진들(화장품, 은행잎, 레고, 동전, 아스팔트)로 정한 규칙"""
    bright_bg = ans["q_bg"].startswith("밝")
    texture = ans["q_texture"].startswith("있")
    uneven = ans["q_light"].startswith("고르지")
    both = ans["q_both"].startswith("섞여")
    similar = ans["q_similar"].startswith("있")
    small = ans["q_small"].startswith("아주")
    touch = ans["q_touch"].startswith("붙")
    same = ans["q_same"].startswith("거의")

    cfg = dict(DEFAULTS)
    cfg["direction"] = "어둡다 (f < th)" if bright_bg else "밝다 (f > th)"
    if both:                         # 배경보다 밝은 물체도, 어두운 물체도 있음 → 배경과의 차이 |f − 배경|로 자름
        cfg["direction"] = "둘 다 (|f − 배경| > th)"
    
    if similar:                      # 배경과 비슷한 물체 → 배경 바로 옆에서 자르고, 윤곽선(edge)도 함께 써서 채움
        cfg.update(th_mode="배경 기준 자동",
                   use_edge=not (same and touch) and not st.session_state.get("_giant", False))
    elif small and not bright_bg:
        uneven = True
        
    if texture and not small:
        tex_level = st.session_state.get("_tex_level", 3.0)
        cfg["med_k"] = 9 if tex_level > 8 else 7
        cfg["open_k"] = 5 if tex_level > 8 else 3
        
    if uneven:                       # 조명 불균일 & 빛 반사 방어 로직 (개선됨)
        obj = st.session_state.get("_obj_size", 0)
        # 빛 반사가 심한 중앙부 키보드 등은 Otsu보다 일정 차이값 이상을 객체로 잡는 것이 정확함
        k = int(np.clip(2.5 * obj, 41, 151)) | 1 if obj else (31 if small else 51)
        cfg.update(bg_sub=True, bg_k=k, th_mode="배경 기준 자동", bg_margin=20)
        
    cfg["close_k"] = 3 if (small or touch or texture) else 5
    cfg["open_k"] = 0 if small else (cfg["open_k"] if texture else 5)
    cfg["min_pct"] = 0.004 if small else (0.02 if texture else 0.05)
    
    obj = st.session_state.get("_obj_size", 0)
    if small and obj:
        cfg["min_pct"] = float(np.clip(0.07 * obj ** 2 / st.session_state.get("_img_size", 1e6) * 100, 0.002, 0.05))
    if touch and not small and not (similar and same):
        cfg["erode_mode"] = "자동"   
    cfg["area_est"] = same and touch 
    
    if st.session_state.get("_plain", False):
        cfg.update(bg_sub=False, th_mode="Otsu 자동", use_edge=False, area_est=False)
    return cfg


def apply_answers():
    ss = st.session_state
    for k, v in answers_to_cfg({k: ss[k] for k in QUESTIONS}).items():
        if k != "max_side":
            ss[k] = v
    ss.pop("last_change", None)


if "answers_applied" not in st.session_state:
    apply_answers()
    st.session_state["answers_applied"] = True


# =====================================================================
# 지금 상황 고르기
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
    more = 1 if (ss["direction"].startswith("어둡") and not ss["bg_sub"]) else -1
    th_key, lo, hi = {"Otsu 자동": ("th_offset", -60, 60), "직접 지정": ("th_val", 0, 255),
                      "배경 기준 자동": ("bg_margin", 3, 80)}[ss["th_mode"]]
    if ss["th_mode"] == "배경 기준 자동":
        more = -1                                              
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
    
    # 0나누기 방어 (앱 크래시 방지)
    if np.isnan(sigma_b).all():
        return 0
    return int(np.nanargmax(sigma_b))


@st.cache_data(show_spinner=False)
def load_gray(file_bytes, max_side):
    I = Image.open(io.BytesIO(file_bytes))
    I = ImageOps.exif_transpose(I)  
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


def estimate_background(fd, k):
    h, w = fd.shape
    small = np.array(Image.fromarray(fd).resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR))
    bg = nd.median_filter(small, size=max(3, (k // 4) | 1), mode="nearest")
    return np.array(Image.fromarray(bg).resize((w, h), Image.BILINEAR))


def label_cores(mask, ek, fill, structure, min_area):
    core = nd.binary_erosion(mask, np.ones((ek, ek))) if ek > 0 else mask
    if fill:
        core = nd.binary_fill_holes(core)
    lab, n = nd.label(core, structure=structure)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return lab, core, int((areas >= max(1, min_area * 0.2)).sum())


def split_each(labels, min_area, kmax=41):
    out = np.zeros_like(labels); nxt = 1
    areas = np.bincount(labels.ravel())[1:]
    med = np.median(areas) if len(areas) else 0
    for i, sl in enumerate(nd.find_objects(labels), 1):
        if sl is None: continue
        sl = tuple(slice(max(s.start-2,0), s.stop+2) for s in sl)
        comp = labels[sl] == i
        area = comp.sum()
        best = (1, None, 0)
        if area > 1.5 * med:
            for k in range(3, kmax + 1, 2):
                core = nd.binary_erosion(comp, np.ones((k, k)))
                if not core.any(): break
                lab, n = nd.label(core)
                a = np.bincount(lab.ravel())[1:]
                good = (a >= max(min_area * 0.1, 0.02 * area)).sum()
                if good > best[0]:
                    keep = np.zeros(n + 1, bool); keep[1:] = a >= max(min_area * 0.1, 0.02 * area)
                    best = (good, np.where(keep[lab], lab, 0), k)
        if best[1] is None:
            out[sl][comp] = nxt; nxt += 1; continue
        lab = best[1]
        for _ in range(best[2] + 2):           
            grown = nd.maximum_filter(lab, size=3)   
            lab = np.where((lab == 0) & comp, grown, lab)
        lab[~comp] = 0
        for j in range(1, lab.max() + 1):
            m = lab == j
            if m.any(): out[sl][m] = nxt; nxt += 1
    return out


def process(f, P):
    param = P["med_k"] if P["denoise"] == "Median" else (P["sigma"] if P["denoise"] == "Gaussian" else 0)
    f_denoised = do_denoise(f, P["denoise"], param)

    dark_obj = P["direction"].startswith("어둡")
    both_dir = P["direction"].startswith("둘")
    if P["bg_sub"]:
        bg = estimate_background(f_denoised, P["bg_k"]).astype(int)
        diff = f_denoised.astype(int) - bg
        if both_dir:
            diff = np.abs(diff)          
        elif dark_obj:
            diff = -diff                 
        f_th = diff.clip(0, 255).astype(np.uint8)
        obj_is_dark = False              
    elif both_dir:
        bg_level = int(np.argmax(np.bincount(f_denoised.ravel(), minlength=256)))
        f_th = np.abs(f_denoised.astype(int) - bg_level).clip(0, 255).astype(np.uint8)
        obj_is_dark = False
    else:
        f_th = f_denoised
        obj_is_dark = dark_obj

    if P["th_mode"] == "Otsu 자동":
        th_val = int(np.clip(otsu(f_th) + P["th_offset"], 0, 255))
    elif P["th_mode"] == "배경 기준 자동":
        bg_level = int(np.argmax(np.bincount(f_th.ravel(), minlength=256)))  
        th_val = int(np.clip(bg_level - P["bg_margin"] if obj_is_dark else bg_level + P["bg_margin"], 0, 255))
    else:
        th_val = P["th_val"]
        
    f_mask = f_th < th_val if obj_is_dark else f_th > th_val
    
    if P["close_k"] > 0:
        f_mask = nd.binary_closing(f_mask, structure=np.ones((P["close_k"], P["close_k"])))
    if P["open_k"] > 0:
        f_mask = nd.binary_opening(f_mask, structure=np.ones((P["open_k"], P["open_k"])))

    structure = None if P["connectivity"] == "4-연결" else np.ones((3, 3))
    min_area = int(P["min_pct"] / 100 * f.size)
    ek = 0
    if P["erode_mode"] == "자동" and not P["use_edge"]:
        counts = [(k, label_cores(f_mask, k, P["fill"], structure, min_area)[2]) for k in range(3, 82, 4)]
        best = max(c for _, c in counts)
        ek = next(k for k, c in counts if c >= 0.97 * best)
    elif P["erode_mode"] == "직접 지정":
        ek = P["erode_k"]
        
    if P["use_edge"]:
        ek = 0
        full = nd.binary_fill_holes(f_mask) if P["fill"] else f_mask
        fd = f_denoised.astype(float)
        grad = np.hypot(nd.sobel(fd, axis=0), nd.sobel(fd, axis=1))      
        grad8 = (grad / max(grad.max(), 1e-6) * 255).astype(np.uint8)
        edge = grad8 > otsu(grad8) * P["edge_scale"]
        cut = full & ~edge
        labels, n = nd.label(cut, structure=structure)
        a = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        keep0 = np.zeros(n + 1, bool)
        keep0[1:] = a >= min_area
        labels = np.where(keep0[labels], labels, 0)
        labels = nd.maximum_filter(labels, size=5) * full                 
        mask_shown = cut
    else:
        labels, core, _ = label_cores(f_mask, ek, P["fill"], structure, min_area)
        if ek > 0:
            region = f_mask | nd.binary_dilation(core, np.ones((ek, ek)))
            labels = nd.maximum_filter(labels, size=ek + 2) * region
            mask_shown = region
        else:
            mask_shown = core
        if P["erode_mode"] == "자동":
            labels = split_each(labels, min_area)   
    nums = int(labels.max())

    areas = np.bincount(labels.ravel(), minlength=nums + 1)[1:]
    keep = areas >= min_area
    lut = np.zeros(nums + 1, dtype=labels.dtype)
    lut[1:][keep] = np.arange(1, keep.sum() + 1)
    labels = lut[labels]
    areas = areas[keep]
    nums = int(keep.sum())

    if P["area_est"] and nums > 0:
        est = np.maximum(1, np.round(areas / np.median(areas))).astype(int)
    else:
        est = np.ones(nums, dtype=int)
    return dict(f_denoised=f_denoised, f_th=f_th, dark_obj=dark_obj, th_val=th_val, mask_shown=mask_shown,
                ek=ek, labels=labels, areas=areas, nums=nums, est=est, total=int(est.sum()))


@st.cache_data(show_spinner=False)
def auto_answers(f):
    fd = do_denoise(f, "Median", 5)
    h, w = fd.shape
    ans = {}
    st.session_state.update(_giant=False, _obj_size=0.0, _tex_level=3.0, _img_size=float(f.size), _plain=False)

    b = max(3, min(h, w) // 30)
    border = np.concatenate([fd[:b].ravel(), fd[-b:].ravel(), fd[:, :b].ravel(), fd[:, -b:].ravel()])
    bright_bg = np.median(border) > 128
    ans["q_bg"] = QUESTIONS["q_bg"][1][0 if bright_bg else 1]

    mode = int(np.argmax(np.bincount(fd.ravel(), minlength=256)))
    m = nd.uniform_filter(fd.astype(float), 7)
    m2 = nd.uniform_filter(fd.astype(float) ** 2, 7)
    local_std = np.sqrt(np.clip(m2 - m * m, 0, None))
    tex_level = float(np.median(local_std[np.abs(fd.astype(int) - mode) < 20]))
    texture = tex_level > 1.5
    st.session_state["_tex_level"] = tex_level
    ans["q_texture"] = QUESTIONS["q_texture"][1][int(texture)]

    both = False
    if bright_bg and mode < 215:
        brighter = nd.binary_opening(fd.astype(int) > mode + 30, structure=np.ones((5, 5)))
        lab_b, n_b = nd.label(brighter)
        sizes = np.bincount(lab_b.ravel(), minlength=n_b + 1)[1:]
        both = (sizes >= 0.0003 * fd.size).sum() >= 2
    ans["q_both"] = QUESTIONS["q_both"][1][int(both)]

    small_img = np.array(Image.fromarray(fd).resize((max(8, w // 8), max(8, h // 8))))
    k = max(3, min(small_img.shape) // 6)
    if bright_bg:
        bgmap = nd.minimum_filter(nd.maximum_filter(small_img, size=k), size=k)
    else:
        bgmap = nd.maximum_filter(nd.minimum_filter(small_img, size=k), size=k)
    bgmap = nd.uniform_filter(bgmap.astype(float), k)
    uneven = (np.percentile(bgmap, 95) - np.percentile(bgmap, 5)) > 60
    ans["q_light"] = QUESTIONS["q_light"][1][int(uneven)]

    similar = False
    if bright_bg and not texture:
        obj_any = (mode - fd.astype(int)) > 12
        obj_otsu = fd < otsu(fd)
        similar = obj_otsu.sum() / max(1, obj_any.sum()) < 0.85
    ans["q_similar"] = QUESTIONS["q_similar"][1][int(similar)]

    probe = dict(ans, q_small=QUESTIONS["q_small"][1][0], q_touch=QUESTIONS["q_touch"][1][0],
                 q_same=QUESTIONS["q_same"][1][0])
    P = answers_to_cfg(probe)
    P.update(min_pct=0.004, use_edge=False)    
    r0 = process(f, P)
    areas = r0["areas"]
    
    giant = len(areas) > 0 and areas.max() > 0.2 * f.size
    st.session_state["_giant"] = bool(giant)
    if giant:
        areas = process(f, dict(P, erode_mode="자동"))["areas"]
    small = len(areas) >= 10 and np.median(areas) < 0.0004 * f.size
    if small and tex_level > 8:
        la = np.log(areas.astype(float))
        q = ((la - la.min()) / max(la.max() - la.min(), 1e-6) * 255).astype(np.uint8)
        cut = np.exp(la.min() + otsu(q) / 255 * (la.max() - la.min()))
        objs = areas[areas > cut]
        small = len(objs) >= 10 and np.median(objs) < 0.0004 * f.size
    ans["q_small"] = QUESTIONS["q_small"][1][int(small)]
    
    st.session_state["_tex_level"] = tex_level
    no_touch = QUESTIONS["q_touch"][1][0]
    base = dict(answers_to_cfg(dict(ans, q_touch=no_touch, q_same=QUESTIONS["q_same"][1][0])), use_edge=False)
    r_base = process(f, base)
    n0, a = r_base["nums"], r_base["areas"]
    touch = giant
    if not giant and not small and n0 > 0:
        r_ero = process(f, dict(base, erode_mode="자동"))
        touch = r_ero["nums"] >= 1.3 * n0 and r_ero["nums"] - n0 >= 3
        if touch:
            a = r_ero["areas"]          
        elif len(a) >= 3:
            med = np.median(a)
            touch = a.max() > 4 * med and (a > 1.6 * med).sum() >= 3
    ans["q_touch"] = QUESTIONS["q_touch"][1][int(touch)]

    same = False
    if len(a) >= 5:
        med = np.median(a)
        single = a[a < 1.6 * med]          
        if len(single) >= 5:
            q25, q50, q75 = np.percentile(single, [25, 50, 75])
            same = (q75 - q25) / q50 < 0.3
    ans["q_same"] = QUESTIONS["q_same"][1][int(same)]

    a_obj = a
    if len(a) >= 10:
        la = np.log(a.astype(float))
        q = ((la - la.min()) / max(la.max() - la.min(), 1e-6) * 255).astype(np.uint8)
        cut = np.exp(la.min() + otsu(q) / 255 * (la.max() - la.min()))
        if (a > cut).sum() >= 5:
            a_obj = a[a > cut]
    st.session_state["_obj_size"] = float(np.sqrt(np.percentile(a_obj, 90))) if len(a_obj) else 0.0

    # 강화된 _plain (Otsu 강제 덮어쓰기) 검사 로직
    final = answers_to_cfg(ans)
    if final["bg_sub"] or final["th_mode"] != "Otsu 자동":
        r_cur = process(f, final)
        r_plain = process(f, dict(final, bg_sub=False, th_mode="Otsu 자동", use_edge=False, area_est=False))
        # 빛 반사가 있는 환경(r_cur가 더 우수함)에서 _plain으로 잘못 덮어씌워지지 않도록 조건을 더 엄격하게 제한
        if (r_plain["nums"] > r_cur["nums"] * 2 
                and r_cur["nums"] < 30 
                and r_plain["nums"] >= 5 
                and r_plain["areas"].max() < 0.2 * f.size):
            st.session_state["_plain"] = True

    for key in ["_giant", "_obj_size", "_tex_level", "_img_size", "_plain"]:
        ans[key] = st.session_state[key]
    return ans


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
# 왼쪽: 조작 (업로드 → 상황 고르기 → 세부 설정[사진 분석 결과 포함])
# =====================================================================
sb = st.sidebar
sb.title("🔍 Object Counting")
with sb.expander("📖 사용 방법"):
    st.markdown(
        "1. **사진 올리기**\n"
        "2. **사진에 따라 답고르기** \n"
        "3. **오른쪽 결과를 원본과 비교하기.** \n"
        "4. **추가보정 : '결과가 이상하면'에서 선택하기.**\n"
        "5. **세부조정 : 그 외에도 보정하고 싶은 부분이 있으면 '⚙️ 세부 설정'에서 직접 조정하기.**\n"
        "- **binary mask**: 흰색이 물체로 인식된 부분\n"
    )

sb.header("① 사진 올리기")
uploaded = sb.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                            label_visibility="collapsed")

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("analyzed_file") != file_id:
        with st.spinner("사진을 분석해서 설정을 자동으로 고르는 중..."):
            auto = auto_answers(load_gray(uploaded.getvalue(), st.session_state["max_side"]))
        for k, v in auto.items():
            st.session_state[k] = v
        apply_answers()
        st.session_state["analyzed_file"] = file_id

    sb.header("② 결과가 이상하면")
    sb.caption("오른쪽 결과를 보고 해당하는 상황을 누르세요. 누를 때마다 한 단계씩 조정돼요.")
    for label, kind in SITUATIONS.items():
        sb.button(label, on_click=adjust, args=(kind,), use_container_width=True)
    if msg := st.session_state.get("last_change"):
        sb.info(f"**조정됨** {msg}")
    sb.button("↺ 처음 값으로 되돌리기", on_click=apply_answers, use_container_width=True)

with sb.expander("⚙️ 세부 설정 (자동으로 정해져요)"):
    cap = st.caption
    st.markdown("**0. 사진 분석 결과**")
    cap("사진을 올리면 자동으로 분석해서 골라 둬요. 실제 사진과 다른 답이 있으면 바꾸세요.")
    for key, (question, opts) in QUESTIONS.items():
        st.radio(question, opts, key=key, on_change=apply_answers)
    st.divider()

    st.markdown("**1. 이미지**")
    st.slider("처리용 최대 변 길이 (px)", 256, 4000, step=64, key="max_side")

    st.markdown("**2. 노이즈 제거**")
    st.selectbox("필터", ["Median", "Gaussian", "없음"], key="denoise")
    if st.session_state["denoise"] == "Median":
        st.slider("Median kernel 크기 (홀수)", 1, 21, step=2, key="med_k")
    elif st.session_state["denoise"] == "Gaussian":
        st.slider("Gaussian sigma", 0.0, 10.0, step=0.5, key="sigma")
    st.checkbox("조명 보정 (배경 빼기)", key="bg_sub")
    if st.session_state["bg_sub"]:
        st.slider("배경 추정 크기 (물체보다 크게)", 11, 151, step=2, key="bg_k")

    st.markdown("**3. Thresholding**")
    st.radio("물체가 배경보다…", ["어둡다 (f < th)", "밝다 (f > th)", "둘 다 (|f − 배경| > th)"], key="direction")
    st.radio("임계값 방식", ["Otsu 자동", "배경 기준 자동", "직접 지정"], key="th_mode")
    if st.session_state["th_mode"] == "Otsu 자동":
        st.slider("Otsu 값 보정 (±)", -60, 60, step=1, key="th_offset")
    elif st.session_state["th_mode"] == "배경 기준 자동":
        st.slider("배경에서 떨어진 정도", 3, 80, step=1, key="bg_margin")
    else:
        st.slider("임계값", 0, 255, key="th_val")

    st.markdown("**4. 모폴로지**")
    st.slider("Closing 크기 (조각 잇기)", 0, 51, step=2, key="close_k")
    st.slider("Opening 크기 (얇은 연결 끊기)", 0, 51, step=2, key="open_k")
    st.checkbox("Hole filling (물체 안 구멍 채우기)", key="fill")
    st.radio("붙은 물체 떼기 (erosion)", ["끄기", "자동", "직접 지정"], horizontal=True, key="erode_mode")
    if st.session_state["erode_mode"] == "직접 지정":
        st.slider("Erosion 크기", 3, 81, step=2, key="erode_k")

    st.checkbox("윤곽선(Sobel)으로 물체 나누기", key="use_edge")
    if st.session_state["use_edge"]:
        st.slider("윤곽선 민감도 (낮을수록 더 잘게 나눔)", 0.3, 1.5, step=0.05, key="edge_scale")

    st.markdown("**5. Connected component**")
    st.slider("최소 면적 (영상 대비 %)", 0.0, 1.0, step=0.001, format="%.3f", key="min_pct")
    st.radio("연결성", ["8-연결", "4-연결"], horizontal=True, key="connectivity")
    st.checkbox("면적으로 개수 보정 (크기가 같은 물체일 때)", key="area_est")


# =====================================================================
# 오른쪽: 각 과정의 결과만
# =====================================================================
if uploaded is None:
    st.info("👈 왼쪽에서 사진을 올리세요.")
    st.stop()

P = {k: st.session_state[k] for k in DEFAULTS}
f = load_gray(uploaded.getvalue(), P["max_side"])
R = process(f, P)
f_denoised, f_th, th_val, ek = R["f_denoised"], R["f_th"], R["th_val"], R["ek"]
labels, areas, nums, est, total = R["labels"], R["areas"], R["nums"], R["est"], R["total"]
idx = np.arange(1, nums + 1)

# --- 화면: 요약 ---
m1, m2, m3 = st.columns(3)
m1.metric("검출된 객체 수", total)
m2.metric("Connected component 수", nums)
m3.metric("임계값", th_val, help=f"방식: {P['th_mode']}")

# --- 1단계: grayscale 불러오기 ---
st.subheader("1. Grayscale 영상 불러오기")
c = st.columns(3)
with c[0]:
    put(fig_img(f, f"input (grayscale) {f.shape[1]}x{f.shape[0]}"))

# --- 2단계: 노이즈 제거 (+ 조명 보정) ---
st.subheader("2. 전처리 (노이즈 제거" + (" + 조명 보정)" if P["bg_sub"] else ")"))
c = st.columns(3)
with c[0]:
    put(fig_img(f, "before"))
with c[1]:
    put(fig_img(f_denoised, f"denoised ({ {'Median': 'median', 'Gaussian': 'gaussian'}.get(P['denoise'], 'none') })"))
if P["bg_sub"]:
    with c[2]:
        put(fig_img(f_th, f"f - background (median {P['bg_k']})"))

# --- 3단계: thresholding → binary mask ---
st.subheader("3. Thresholding → Binary mask")
c = st.columns(3)
with c[0]:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(f_th.ravel(), bins=256, range=(0, 256), color="gray")
    ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
    ax.set_yscale("log")
    ax.set_title("histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.legend()
    fig.tight_layout()
    put(fig)
with c[1]:
    put(fig_img(R["mask_shown"], "binary mask" + (f" (erosion {ek})" if ek else "")
                + (" (Sobel edge cut)" if P["use_edge"] else "")))

# --- 4단계 ---
st.subheader(f"4. Connected components — {total}개" + (f" (component {nums}개 + 면적 보정)" if total != nums else ""))
if nums == 0:
    st.warning("검출된 객체가 없어요. 왼쪽 ⚙️ 세부 설정의 '0. 사진 분석 결과'(특히 배경 색)를 확인해 보세요.")
    st.stop()
if areas.max() > 0.3 * f.size:
    st.warning("하나의 덩어리가 화면의 30% 이상이에요. 여러 물체가 하나로 뭉쳤을 수 있어요 → "
               "왼쪽 ②에서 **'여러 물체가 하나로 붙어 있어요'**를 눌러 보세요.")
c1, c2 = st.columns([3, 1])
with c1:
    colors = np.random.RandomState(0).rand(nums + 1, 3) * 0.8 + 0.2
    colors[0] = 0
    fig, ax = plt.subplots(figsize=(10, 10 * f.shape[0] / f.shape[1]))
    ax.imshow(f, cmap="gray")
    # 컬러맵 오류 방지를 위해 mcolors.ListedColormap 사용
    ax.imshow(np.ma.masked_where(labels == 0, labels), cmap=mcolors.ListedColormap(colors),
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
    if P["area_est"]:
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
