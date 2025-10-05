import argparse
import sys
import cv2
import yaml
import logging
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

# ------------------------------------------------------------
# ЛОГИ
# ------------------------------------------------------------
def setup_logger(out_dir: Path):
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(str(log_path), encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)]
    )

# ------------------------------------------------------------
# УТИЛИТЫ
# ------------------------------------------------------------
def imread_color(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {p}")
    return img

def ensure_empty_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def resize_max_side(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def clahe_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def safe_write_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise IOError(f"Не удалось записать: {path}")

def safe_open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    f = open(tmp, "w", newline="", encoding="utf-8")
    return f, tmp

# ------------------------------------------------------------
# ВЫРАВНИВАНИЕ + ЗАЩИТА ОТ «ЛУЧЕЙ»
# ------------------------------------------------------------
def _project_corners(H: np.ndarray, w: int, h: int) -> np.ndarray:
    pts = np.array([[0,0,1],[w-1,0,1],[w-1,h-1,1],[0,h-1,1]], dtype=np.float32).T  # 3x4
    pr = H @ pts
    pr = (pr[:2, :] / np.maximum(pr[2:3, :], 1e-8)).T  # 4x2
    return pr

def _poly_area(pts: np.ndarray) -> float:
    x = pts[:,0]; y = pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _is_h_valid(H: np.ndarray, src_shape: Tuple[int,int], dst_shape: Tuple[int,int]) -> bool:
    if H is None or not np.isfinite(H).all():
        return False
    det = float(np.linalg.det(H[:2,:2]))
    if not (0.05 <= abs(det) <= 20.0):
        return False
    # ограничим перспективу (чтобы не было «взрыва»)
    if abs(H[2,0]) > 2e-3 or abs(H[2,1]) > 2e-3:
        return False
    w_src, h_src = src_shape[1], src_shape[0]
    w_dst, h_dst = dst_shape[1], dst_shape[0]
    pr = _project_corners(H, w_src, h_src)
    if _poly_area(pr) < 0.05 * (w_src * h_src):
        return False
    inside = ((pr[:,0] >= -0.25*w_dst) & (pr[:,0] <= 1.25*w_dst) &
              (pr[:,1] >= -0.25*h_dst) & (pr[:,1] <= 1.25*h_dst))
    if inside.sum() < 3:
        return False
    return True

def _to_affine(H: np.ndarray) -> np.ndarray:
    A = H.copy().astype(np.float32)
    A[2,0] = 0.0
    A[2,1] = 0.0
    A[2,2] = 1.0
    return A

def align_image_to_golden(src_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает (aligned_gray, H). Валидирует H, при необходимости деградирует к аффинной или I.
    """
    method = cfg["alignment"]["method"]
    if method not in ("auto","orb","ecc","none"):
        method = "auto"
    max_size = int(cfg["alignment"].get("max_size", 1200))
    g_small, g_scale = resize_max_side(golden_gray, max_size)

    H = None
    tried: List[str] = []

    def _try(method_name: str) -> Optional[np.ndarray]:
        tried.append(method_name)
        if method_name == "orb":
            return compute_homography_orb(src_gray, golden_gray, cfg, g_small, g_scale)
        if method_name == "ecc":
            return compute_ecc_warp(src_gray, golden_gray, cfg, g_small, g_scale)
        if method_name == "none":
            return np.eye(3, dtype=np.float32)
        return None

    if method in ("auto","orb"):
        H = _try("orb")
        if H is None and method == "auto":
            logging.warning("ORB не справился, пробуем ECC...")
            H = _try("ecc")
    elif method == "ecc":
        H = _try("ecc")
    else:
        H = _try("none")

    if H is None:
        logging.warning("Выравнивание не удалось (нет H), используем I.")
        H = np.eye(3, dtype=np.float32)

    if not _is_h_valid(H, src_gray.shape, golden_gray.shape):
        logging.warning("Гомография невалидна после %s, пробуем альтернативу/аффинную.", "+".join(tried))
        alt = "ecc" if "orb" in tried else "orb"
        H_alt = _try(alt)
        if H_alt is not None and _is_h_valid(H_alt, src_gray.shape, golden_gray.shape):
            H = H_alt
        else:
            H_aff = _to_affine(H if H_alt is None else H_alt if H_alt is not None else np.eye(3, np.float32))
            if _is_h_valid(H_aff, src_gray.shape, golden_gray.shape):
                H = H_aff
            else:
                logging.warning("Аффинная тоже плоха — используем I.")
                H = np.eye(3, dtype=np.float32)

    h, w = golden_gray.shape[:2]
    aligned_gray = cv2.warpPerspective(
        src_gray, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,  # без «лучей»
        borderValue=0
    )
    return aligned_gray, H

def compute_homography_orb(src_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict,
                           g_small: np.ndarray, g_scale: float) -> Optional[np.ndarray]:
    try:
        max_size = int(cfg["alignment"].get("max_size", 1200))
        s_small, s_scale = resize_max_side(src_gray, max_size)
        orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
        kp1, des1 = orb.detectAndCompute(s_small, None)
        kp2, des2 = orb.detectAndCompute(g_small, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) / s_scale
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) / g_scale
        ransac_thresh = float(cfg["alignment"].get("ransac_thresh", 5.0))
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        return H
    except Exception as e:
        logging.exception("ORB выравнивание упало: %s", e)
        return None

def compute_ecc_warp(src_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict,
                     g_small: np.ndarray, g_scale: float) -> Optional[np.ndarray]:
    try:
        max_size = int(cfg["alignment"].get("max_size", 1200))
        s_small, s_scale = resize_max_side(src_gray, max_size)
        s = s_small.astype(np.float32) / 255.0
        g = g_small.astype(np.float32) / 255.0
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    int(cfg["alignment"].get("ecc_iterations", 60)),
                    float(cfg["alignment"].get("ecc_eps", 1e-5)))
        _, warp_matrix = cv2.findTransformECC(g, s, warp_matrix, warp_mode, criteria, None, 5)
        A = np.vstack([warp_matrix, [0,0,1]])
        S_src = np.array([[1/s_scale, 0, 0],[0, 1/s_scale, 0],[0, 0, 1]], dtype=np.float32)
        S_g   = np.array([[g_scale, 0, 0],[0, g_scale, 0],[0, 0, 1]], dtype=np.float32)
        H = S_g @ A @ S_src
        return H
    except Exception as e:
        logging.exception("ECC выравнивание упало: %s", e)
        return None

# ------------------------------------------------------------
# ФОТОМЕТРИЯ / ОСВЕЩЕНИЕ
# ------------------------------------------------------------
def highpass(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0,0), sigma)
    hp = cv2.subtract(gray, blur)
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)
    return hp.astype(np.uint8)

def match_mean_std(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s_mean, s_std = float(src.mean()), float(src.std() + 1e-6)
    r_mean, r_std = float(ref.mean()), float(ref.std() + 1e-6)
    k = r_std / s_std
    dst = (src.astype(np.float32) - s_mean) * k + r_mean
    return np.clip(dst, 0, 255).astype(np.uint8)

def preprocess_gray_single(gray: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["preprocess"].get("clahe", True):
        gray = clahe_gray(gray)
    k = int(cfg["preprocess"].get("blur_ksize", 3))
    if k and k >= 3:
        gray = cv2.GaussianBlur(gray, (k,k), 0)
    return gray

def prepare_pair_for_diff(aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    illum = cfg.get("illumination", {})
    sigma = float(illum.get("highpass_sigma", 0))
    g1 = highpass(aligned_gray, sigma) if sigma > 0 else aligned_gray
    g0 = highpass(golden_gray,  sigma) if sigma > 0 else golden_gray
    if illum.get("match_mean_std", False):
        g1 = match_mean_std(g1, g0)
    g1 = preprocess_gray_single(g1, cfg)
    g0 = preprocess_gray_single(g0, cfg)
    return g1, g0

# ------------------------------------------------------------
# ДИФФЫ
# ------------------------------------------------------------
def diff_edges_absdiff(g_aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> np.ndarray:
    e = cfg["diff"]["edges"]
    def edges(x):
        canny = cv2.Canny(x, int(e.get("canny1",80)), int(e.get("canny2",200)))
        if int(e.get("dilate", 0)) > 0:
            k = np.ones((3,3), np.uint8)
            canny = cv2.dilate(canny, k, iterations=int(e["dilate"]))
        return canny
    e1 = edges(g_aligned_gray)
    e2 = edges(golden_gray)
    diff = cv2.absdiff(e1, e2)
    _, binm = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)
    return binm

def diff_gray_absdiff(g_aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> np.ndarray:
    dif = cv2.absdiff(g_aligned_gray, golden_gray)
    t = int(cfg["diff"]["gray_absdiff"].get("thresh", 35))
    if t <= 0:
        _, binm = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binm = cv2.threshold(dif, t, 255, cv2.THRESH_BINARY)
    return binm

def diff_grad_absdiff(g_aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> np.ndarray:
    gcfg = cfg["diff"].get("grad", {})
    use_scharr = bool(gcfg.get("use_scharr", True))
    ksize = int(gcfg.get("ksize", 3))

    def grad_mag(x):
        if use_scharr:
            gx = cv2.Scharr(x, cv2.CV_16S, 1, 0)
            gy = cv2.Scharr(x, cv2.CV_16S, 0, 1)
        else:
            gx = cv2.Sobel(x, cv2.CV_16S, 1, 0, ksize=ksize)
            gy = cv2.Sobel(x, cv2.CV_16S, 0, 1, ksize=ksize)
        mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return mag

    m1 = grad_mag(g_aligned_gray)
    m0 = grad_mag(golden_gray)
    dif = cv2.absdiff(m1, m0)

    t = int(gcfg.get("thresh", 0))
    if t <= 0:
        _, binm = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binm = cv2.threshold(dif, t, 255, cv2.THRESH_BINARY)
    return binm

def diff_ssim(g_aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> np.ndarray:
    dif = cv2.absdiff(g_aligned_gray, golden_gray)
    dif = cv2.GaussianBlur(dif, (7,7), 0)
    _, binm = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binm

def make_defect_map(aligned_gray: np.ndarray, golden_gray: np.ndarray, cfg: dict) -> np.ndarray:
    mode = cfg["diff"]["mode"]
    if mode == "grad_absdiff":
        binm = diff_grad_absdiff(aligned_gray, golden_gray, cfg)
    elif mode == "edges_absdiff":
        binm = diff_edges_absdiff(aligned_gray, golden_gray, cfg)
    elif mode == "gray_absdiff":
        binm = diff_gray_absdiff(aligned_gray, golden_gray, cfg)
    else:
        binm = diff_ssim(aligned_gray, golden_gray, cfg)

    k_open = int(cfg["post"].get("open_ksize", 3))
    k_close = int(cfg["post"].get("close_ksize", 3))
    if k_open >= 3:
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((k_open,k_open), np.uint8))
    if k_close >= 3:
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((k_close,k_close), np.uint8))
    min_area = int(cfg["post"].get("min_blob_area", 0))
    if min_area > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((binm>0).astype(np.uint8), connectivity=8)
        keep = np.zeros_like(binm)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels==i] = 255
        binm = keep
    return binm

# ------------------------------------------------------------
# ROI
# ------------------------------------------------------------
def load_roi_mask(path: Path, target_shape: Tuple[int,int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"ROI-маска не прочитана: {path}")
    if mask.ndim == 3:
        mask = mask[:,:,0]
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def color_map_for_labels(lbl: np.ndarray) -> np.ndarray:
    uniq = np.unique(lbl)
    rng = np.random.default_rng(42)
    colors = {u: (int(rng.integers(64,255)), int(rng.integers(64,255)), int(rng.integers(64,255))) for u in uniq if u!=0}
    out = np.zeros((lbl.shape[0], lbl.shape[1], 3), np.uint8)
    for u,c in colors.items():
        out[lbl==u] = c
    return out

def overlay_defects_on_image(base_bgr: np.ndarray, defect_bin: np.ndarray, alpha: float=0.45) -> np.ndarray:
    color = np.zeros_like(base_bgr)
    color[defect_bin>0] = (0,0,255)
    return cv2.addWeighted(base_bgr, 1.0, color, alpha, 0)

def summarize_rois(defect_bin: np.ndarray, roi_lbl: np.ndarray, cfg: dict) -> List[Dict]:
    results = []
    ids = [int(i) for i in np.unique(roi_lbl) if i != 0]
    for rid in ids:
        roi = (roi_lbl == rid)
        area = int(roi.sum())
        if area == 0:
            continue
        defect_area = int((defect_bin>0)[roi].sum())
        pct = (defect_area / area) * 100.0
        rid_str = str(rid)
        meta = cfg["rois"].get(rid_str, {"name": f"ROI_{rid}", "kind":"text",
                                         "defect_pct_warn":1.0, "defect_pct_fail":3.0})
        status = "OK"
        if pct >= float(meta.get("defect_pct_fail", 3.0)):
            status = "FAIL"
        elif pct >= float(meta.get("defect_pct_warn", 1.0)):
            status = "WARN"
        results.append({
            "roi_id": rid,
            "roi_name": meta.get("name", f"ROI_{rid}"),
            "roi_kind": meta.get("kind", "text"),
            "roi_area_px": area,
            "defect_area_px": int(defect_area),
            "defect_pct": round(pct, 4),
            "status": status
        })
    return sorted(results, key=lambda x: (x["roi_kind"], x["roi_id"]))

def draw_roi_labels(bgr: np.ndarray, roi_lbl: np.ndarray, roi_summ: List[Dict], cfg: dict) -> np.ndarray:
    """
    Имя ROI — белым, процент — всегда зелёным, статус — по цвету статуса.
    """
    out = bgr.copy()
    if not cfg["viz"].get("show_labels", True):
        return out
    for r in roi_summ:
        rid = r["roi_id"]
        ys, xs = np.where(roi_lbl == rid)
        if len(xs)==0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        roi_name = r["roi_name"]
        pct_text = f'{r["defect_pct"]:.2f}%'
        status_text = r["status"]
        status_color = (0,255,0) if status_text=="OK" else ((0,255,255) if status_text=="WARN" else (0,0,255))
        pos = (max(0,cx-60), max(20, cy-10))
        cv2.putText(out, f'{roi_name}:', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        (tw,_), _ = cv2.getTextSize(f'{roi_name}:', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(out, pct_text, (pos[0]+tw+5, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        (tw2,_), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(out, status_text, (pos[0]+tw+tw2+10, pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
    return out

def apply_roi_clip(defect_bin: np.ndarray, roi_lbl: np.ndarray, cfg: dict) -> np.ndarray:
    """Оставляем дефекты только внутри любых ROI (roi_lbl>0), если включено viz.only_inside_rois."""
    if not cfg.get("viz", {}).get("only_inside_rois", True):
        return defect_bin
    mask = (roi_lbl > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(defect_bin, defect_bin, mask=mask)

# ------------------------------------------------------------
# ПАЙПЛАЙН
# ------------------------------------------------------------
def process_one_image(img_path: Path, out_dir: Path, cfg: dict,
                      golden_gray: np.ndarray, golden_bgr_unused: np.ndarray, roi_lbl: np.ndarray) -> Dict:
    """
    Визуализация отличий поверх ВЫРОВНЕННОГО проверяемого изображения (цветного),
    а не на эталоне.
    """
    try:
        color = imread_color(img_path)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        aligned_gray, H = align_image_to_golden(gray, golden_gray, cfg)

        # выровняем также цветной кадр для корректной отрисовки
        h, w = golden_gray.shape[:2]
        aligned_bgr = cv2.warpPerspective(
            color, H, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0)
        )

        g1, g0 = prepare_pair_for_diff(aligned_gray, golden_gray, cfg)
        defect_bin = make_defect_map(g1, g0, cfg)
        defect_bin = apply_roi_clip(defect_bin, roi_lbl, cfg)

        roi_sum = summarize_rois(defect_bin, roi_lbl, cfg)

        overlay = overlay_defects_on_image(aligned_bgr, defect_bin, float(cfg["viz"].get("alpha_overlay", 0.45)))
        overlay = draw_roi_labels(overlay, roi_lbl, roi_sum, cfg)

        base_name = img_path.stem
        safe_write_image(out_dir / "overlays" / f"{base_name}.png", overlay)
        safe_write_image(out_dir / "masks" / f"{base_name}_defects.png", defect_bin)

        if cfg.get("debug",{}).get("save_intermediate", False):
            safe_write_image(out_dir / "debug" / f"{base_name}_norm_g1.png", g1)
            safe_write_image(out_dir / "debug" / f"{base_name}_norm_g0.png", g0)
            safe_write_image(out_dir / "debug" / f"{base_name}_roi.png", color_map_for_labels(roi_lbl))
            dif = cv2.absdiff(g1, g0)
            safe_write_image(out_dir / "debug" / f"{base_name}_diff_gray.png", dif)

        worst = "OK"
        for r in roi_sum:
            if r["status"] == "FAIL":
                worst = "FAIL"; break
            elif r["status"] == "WARN" and worst != "FAIL":
                worst = "WARN"

        return {"image": str(img_path), "status": worst, "rois": roi_sum}

    except Exception as e:
        logging.exception("Ошибка обработки %s: %s", img_path, e)
        return {"image": str(img_path), "status": "ERROR", "error": str(e), "rois": []}

def write_summary_csv(out_dir: Path, rows: List[Dict]):
    csv_path = out_dir / "summary.csv"
    f, tmp = safe_open_csv(csv_path)
    import csv
    with f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["image","status","roi_id","roi_name","roi_kind","roi_area_px","defect_area_px","defect_pct","roi_status"])
        for row in rows:
            if not row.get("rois"):
                writer.writerow([row.get("image"), row.get("status"), "", "", "", "", "", "", ""])
                continue
            for r in row["rois"]:
                writer.writerow([
                    row.get("image"), row.get("status"),
                    r["roi_id"], r["roi_name"], r["roi_kind"],
                    r["roi_area_px"], r["defect_area_px"], r["defect_pct"], r["status"]
                ])
    Path(tmp).replace(csv_path)

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def init_assets(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    golden_bgr = imread_color(Path(cfg["golden_path"]))
    golden_gray = cv2.cvtColor(golden_bgr, cv2.COLOR_BGR2GRAY)
    roi_lbl = load_roi_mask(Path(cfg["roi_mask_path"]), golden_gray.shape[:2])
    return golden_bgr, golden_gray, roi_lbl

def run_dir(input_dir: Path, out_dir: Path, cfg: dict):
    ensure_empty_dir(out_dir / "overlays")
    ensure_empty_dir(out_dir / "masks")
    ensure_empty_dir(out_dir / "debug")
    setup_logger(out_dir)

    golden_bgr, golden_gray, roi_lbl = init_assets(cfg)
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    imgs = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    if not imgs:
        logging.warning("Нет входных изображений в %s", input_dir)
    results = []
    for p in tqdm(sorted(imgs), desc="processing"):
        results.append(process_one_image(p, out_dir, cfg, golden_gray, golden_bgr, roi_lbl))
    write_summary_csv(out_dir, results)
    logging.info("Готово. Обработано изображений: %d", len(results))

def run_one(input_img: Path, out_dir: Path, cfg: dict):
    ensure_empty_dir(out_dir / "overlays")
    ensure_empty_dir(out_dir / "masks")
    ensure_empty_dir(out_dir / "debug")
    setup_logger(out_dir)

    golden_bgr, golden_gray, roi_lbl = init_assets(cfg)
    res = process_one_image(input_img, out_dir, cfg, golden_gray, golden_bgr, roi_lbl)
    write_summary_csv(out_dir, [res])
    logging.info("Готово. Статус: %s", res.get("status"))

def run_camera(cam_spec: str, out_dir: Path, cfg: dict):
    ensure_empty_dir(out_dir / "overlays")
    ensure_empty_dir(out_dir / "masks")
    ensure_empty_dir(out_dir / "debug")
    setup_logger(out_dir)

    golden_bgr, golden_gray, roi_lbl = init_assets(cfg)

    cap = cv2.VideoCapture(cam_spec)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру/поток: {cam_spec}")

    idx = 0
    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.warning("Кадр не прочитан, завершаем.")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            aligned_gray, H = align_image_to_golden(gray, golden_gray, cfg)

            h, w = golden_gray.shape[:2]
            aligned_bgr = cv2.warpPerspective(
                frame, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0,0,0)
            )

            g1, g0 = prepare_pair_for_diff(aligned_gray, golden_gray, cfg)
            defect_bin = make_defect_map(g1, g0, cfg)
            defect_bin = apply_roi_clip(defect_bin, roi_lbl, cfg)
            roi_sum = summarize_rois(defect_bin, roi_lbl, cfg)

            overlay = overlay_defects_on_image(aligned_bgr, defect_bin, float(cfg["viz"].get("alpha_overlay", 0.45)))
            overlay = draw_roi_labels(overlay, roi_lbl, roi_sum, cfg)

            cv2.imshow("ValveQC Overlay", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                base_name = f"camera_{idx:06d}"
                safe_write_image(out_dir / "overlays" / f"{base_name}.png", overlay)
                safe_write_image(out_dir / "masks" / f"{base_name}_defects.png", defect_bin)
                rows.append({"image": base_name, "status": "N/A", "rois": roi_sum})
                logging.info("Снимок сохранён: %s", base_name)
            if key == 27:
                break
            idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if rows:
            write_summary_csv(out_dir, rows)

def parse_args():
    ap = argparse.ArgumentParser("ValveQC")
    ap.add_argument("--config", required=True, help="Путь к qc_config.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-dir", help="Папка с изображениями")
    g.add_argument("--input", help="Один входной файл")
    g.add_argument("--camera", help="Индекс камеры (например, 0) или RTSP/HTTP url")
    ap.add_argument("--out-dir", required=True, help="Папка результатов")
    return ap.parse_args()

def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    out_dir = Path(args.out_dir)
    try:
        if args.input_dir:
            run_dir(Path(args.input_dir), out_dir, cfg)
        elif args.input:
            run_one(Path(args.input), out_dir, cfg)
        else:
            cam_spec = args.camera
            if isinstance(cam_spec, str) and cam_spec.isdigit():
                cam_spec = int(cam_spec)
            run_camera(cam_spec, out_dir, cfg)
    except Exception as e:
        print("FATAL:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
