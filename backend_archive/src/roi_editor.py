import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict

# -----------------------
# УТИЛИТЫ/IO
# -----------------------
def imread_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return img

def imread_mask_or_empty(path: Optional[Path], shape_hw: Tuple[int,int]) -> np.ndarray:
    if path is None:
        return np.zeros(shape_hw, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Не удалось прочитать маску: {path}")
    if m.ndim == 3:
        m = m[:,:,0]
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)

def load_labels_yaml(path: Optional[Path]) -> Dict[int, Dict]:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # ожидаем структуру:
    # rois:
    #   "1": { name: "TEXT_TOP", kind: "text" }
    rois = raw.get("rois", {})
    out = {}
    for k,v in rois.items():
        try:
            out[int(k)] = v or {}
        except Exception:
            pass
    return out

def save_labels_yaml(path: Path, mapping: Dict[int, Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rois": {str(k): v for k,v in sorted(mapping.items())}}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=True)

def safe_write_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"Не удалось записать: {path}")

def color_for_id(i: int) -> Tuple[int,int,int]:
    if i == 0:
        return (0,0,0)
    # стабильная "палитра"
    rng = np.random.default_rng(i * 7919)
    return (int(rng.integers(64,255)), int(rng.integers(64,255)), int(rng.integers(64,255)))

def make_overlay(base_bgr: np.ndarray, mask: np.ndarray, alpha: float=0.45) -> np.ndarray:
    col = np.zeros_like(base_bgr)
    ids = np.unique(mask)
    for rid in ids:
        if rid == 0: 
            continue
        col[mask==rid] = color_for_id(int(rid))
    return cv2.addWeighted(base_bgr, 1.0, col, alpha, 0)

# -----------------------
# СОСТОЯНИЕ РАЗМЕТКИ
# -----------------------
class RoiEditorState:
    def __init__(self, base_bgr: np.ndarray, mask: np.ndarray, labels: Dict[int,Dict]):
        self.base_bgr = base_bgr
        self.h, self.w = base_bgr.shape[:2]
        self.mask = mask  # uint8, индексная
        self.labels = labels  # {id: {name:str, kind:str, ...}}

        self.current_id: int = 1
        self.current_name: str = self.labels.get(1, {}).get("name", "ROI_1")
        self.drawing: bool = False
        self.poly_pts: List[Tuple[int,int]] = []  # текущий многоугольник
        self.polygons: List[Tuple[int, List[Tuple[int,int]]]] = []  # история залитых (для "undo")
        self.window = "ROI Editor"
        self.alpha_overlay = 0.45
        self.show_overlay = True
        self.help_on = True

    def set_current_id(self, rid: int):
        if rid < 0 or rid > 255:
            return
        self.current_id = int(rid)
        self.current_name = self.labels.get(rid, {}).get("name", f"ROI_{rid}")

    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay

    def toggle_help(self):
        self.help_on = not self.help_on

    def add_point(self, x: int, y: int):
        x = int(np.clip(x, 0, self.w-1))
        y = int(np.clip(y, 0, self.h-1))
        self.poly_pts.append((x,y))

    def undo_point(self):
        if self.poly_pts:
            self.poly_pts.pop()

    def start_poly(self):
        if not self.drawing:
            self.drawing = True
            self.poly_pts = []

    def cancel_poly(self):
        self.drawing = False
        self.poly_pts = []

    def close_and_fill(self):
        if not self.drawing or len(self.poly_pts) < 3:
            return False
        pts = np.array(self.poly_pts, dtype=np.int32)
        cv2.fillPoly(self.mask, [pts], int(self.current_id))
        self.polygons.append((int(self.current_id), self.poly_pts.copy()))
        self.drawing = False
        self.poly_pts = []
        return True

    def undo_polygon(self):
        if not self.polygons:
            return
        # перерисуем маску с нуля (быстрее и надёжнее)
        last = self.polygons.pop()
        self.mask[:] = 0
        for rid, pts in self.polygons:
            cv2.fillPoly(self.mask, [np.array(pts, np.int32)], int(rid))

    def draw_ui(self) -> np.ndarray:
        if self.show_overlay:
            canvas = make_overlay(self.base_bgr, self.mask, self.alpha_overlay)
        else:
            canvas = self.base_bgr.copy()

        # текущий полигон (контур)
        if self.drawing and len(self.poly_pts) >= 1:
            for i in range(1, len(self.poly_pts)):
                cv2.line(canvas, self.poly_pts[i-1], self.poly_pts[i], (0,255,255), 2, cv2.LINE_AA)
            # точка старта
            cv2.circle(canvas, self.poly_pts[0], 4, (0,255,0), -1, cv2.LINE_AA)
            # текущее ребро к курсору рисуем в колбэке мыши — не храним

        # HUD
        hud_lines = [
            f"ID: {self.current_id}  Name: {self.current_name}",
            "[Mouse LMB] add point, [RMB] close/fill, [Backspace] undo point, [U] undo polygon",
            "[N] new polygon, [0-9] quick id, [I] enter id, [A] rename id, [O] overlay on/off, [H] help on/off",
            "[S] save, [Esc] exit.  Eraser = fill with ID=0."
        ]
        y = 24
        for ln in hud_lines if self.help_on else hud_lines[:1]:
            cv2.putText(canvas, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 3, cv2.LINE_AA)
            cv2.putText(canvas, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            y += 24
        return canvas

# -----------------------
# КОЛБЭК МЫШИ
# -----------------------
def on_mouse(event, x, y, flags, state: RoiEditorState):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not state.drawing:
            state.start_poly()
        state.add_point(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        # рисуем "резинку" последнего сегмента в on-screen превью
        pass

    elif event == cv2.EVENT_RBUTTONDOWN:
        # закрыть и залить текущий
        ok = state.close_and_fill()
        if not ok:
            # если точек мало — просто отменим "недополигон"
            state.cancel_poly()

# -----------------------
# ОСНОВНОЙ ЦИКЛ
# -----------------------
def main():
    ap = argparse.ArgumentParser("ROI Mask Editor")
    ap.add_argument("--golden", required=True, help="Путь к эталонному изображению (BGR)")
    ap.add_argument("--mask", help="Необязательная существующая индексная маска (grayscale PNG)")
    ap.add_argument("--labels-yaml", help="YAML с именами ROI (rois: {'1': {name: TEXT_TOP}}). Будет создан при сохранении.", default=None)
    ap.add_argument("--save-mask", required=False, help="Куда сохранить маску PNG. По умолчанию рядом с golden как *_roi_mask.png")
    ap.add_argument("--save-preview", required=False, help="Куда сохранить цветной оверлей-превью PNG. По умолчанию *_roi_preview.png")
    args = ap.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"Файл не найден: {golden_path}")
        sys.exit(1)

    base = imread_color(golden_path)
    h, w = base.shape[:2]
    mask_path = Path(args.mask) if args.mask else None
    mask = imread_mask_or_empty(mask_path, (h, w))

    labels_yaml_path = Path(args.labels_yaml) if args.labels_yaml else None
    labels = load_labels_yaml(labels_yaml_path)

    state = RoiEditorState(base, mask, labels)
    cv2.namedWindow(state.window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(state.window, min(1600, max(600, w)), min(1000, max(400, h)))
    cv2.setMouseCallback(state.window, lambda e, x, y, f, p=None: on_mouse(e, x, y, f, state))

    # пути сохранения по умолчанию
    default_mask_out = Path(args.save_mask) if args.save_mask else golden_path.with_name(golden_path.stem + "_roi_mask.png")
    default_prev_out = Path(args.save_preview) if args.save_preview else golden_path.with_name(golden_path.stem + "_roi_preview.png")
    default_yaml_out = labels_yaml_path if labels_yaml_path else golden_path.with_name(golden_path.stem + "_roi_labels.yaml")

    while True:
        frame = state.draw_ui()
        # если идёт отрисовка полигона — проведём "динамическое" ребро к курсору
        if state.drawing and len(state.poly_pts) >= 1:
            # берём текущие координаты мыши через getWindowImageRect недоступно; простое решение — не рисовать dynamic edge
            # (OpenCV HighGUI не даёт позицию курсора напрямую без событий)
            pass

        cv2.imshow(state.window, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('h') or key == ord('H'):
            state.toggle_help()
        elif key == ord('o') or key == ord('O'):
            state.toggle_overlay()
        elif key == ord('n') or key == ord('N'):
            state.start_poly()
        elif key == 8:  # Backspace
            state.undo_point()
        elif key == ord('u') or key == ord('U'):
            state.undo_polygon()
        elif key == ord('i') or key == ord('I'):
            # ввод ID в консоли
            try:
                rid = int(input("Введите целочисленный ROI ID (0..255): ").strip())
                state.set_current_id(rid)
            except Exception:
                print("Неверный ID.")
        elif key == ord('a') or key == ord('A'):
            # переименование текущего ID
            new_name = input(f"Введите имя для ROI {state.current_id}: ").strip()
            if state.current_id not in state.labels:
                state.labels[state.current_id] = {}
            state.labels[state.current_id]["name"] = new_name if new_name else f"ROI_{state.current_id}"
            state.current_name = state.labels[state.current_id]["name"]
        elif key == ord('s') or key == ord('S'):
            # сохранить
            try:
                safe_write_image(default_mask_out, state.mask)
                prev = make_overlay(state.base_bgr, state.mask, state.alpha_overlay)
                safe_write_image(default_prev_out, prev)
                # доп. сохранение YAML с именами
                # соберём только реально присутствующие IDs
                present_ids = [int(i) for i in np.unique(state.mask) if int(i) != 0]
                present_map = {i: state.labels.get(i, {"name": f"ROI_{i}"}) for i in present_ids}
                save_labels_yaml(default_yaml_out, present_map)
                print(f"Сохранено:\n - Маска: {default_mask_out}\n - Превью: {default_prev_out}\n - YAML: {default_yaml_out}")
            except Exception as e:
                print("Ошибка сохранения:", e)
        else:
            # быстрый выбор id по цифрам 0-9
            if key in [ord(str(d)) for d in range(10)]:
                digit = int(chr(key))
                state.set_current_id(digit)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
