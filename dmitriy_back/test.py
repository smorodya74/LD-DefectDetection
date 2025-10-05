from ultralytics import YOLO
import torch, os, csv, json, time
from collections import Counter, defaultdict

MODEL_PATH = "best_2.pt"     # путь к твоим весам
IMGSZ = 1000
CONF = 0.10

# --- авто-выбор девайса для Ultralytics ---
if torch.cuda.is_available():
    device_ultra = 0        # важно: Ultralytics ждёт 0/1/... или 'cpu'
    use_half = True
else:
    device_ultra = "cpu"
    use_half = False

print("device:", "cuda:0" if device_ultra == 0 else "cpu")

model = YOLO(MODEL_PATH)
if use_half:
    try:
        model.fuse()
    except Exception:
        pass  # не страшно, если не поддерживается

# общие папки вывода
os.makedirs("reports", exist_ok=True)
os.makedirs("exports/json", exist_ok=True)

def save_json_result(r, out_dir="exports/json"):
    """Сохраняет результаты одного изображения в JSON."""
    names = r.names
    dets = []
    if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
        for xyxy, conf, cls in zip(r.boxes.xyxy.tolist(),
                                   r.boxes.conf.tolist(),
                                   r.boxes.cls.tolist()):
            cls = int(cls)
            dets.append({
                "class_id": cls,
                "class_name": names.get(cls, str(cls)),
                "conf": float(conf),
                "bbox_xyxy": [float(v) for v in xyxy]
            })
    payload = {
        "image_path": r.path,
        "orig_shape": list(r.orig_shape),
        "detections": dets,
        "speed_ms": r.speed   # {'preprocess':..,'inference':..,'postprocess':..}
    }
    base = os.path.splitext(os.path.basename(r.path))[0]
    with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# --- одиночное изображение (если есть рядом) ---
if os.path.isfile("test.jpg"):
    res = model.predict(
        source="test.jpg",
        imgsz=IMGSZ,
        conf=CONF,
        device=device_ultra,
        save=True,           # сохранит визуализацию в runs/...
        save_txt=True,       # сохранит labels в YOLO-формате
        save_crop=True,      # вырежет детекции в runs/.../crops/<class>/
        half=use_half
    )
    r = res[0]
    print("boxes xyxy:\n", r.boxes.xyxy)
    print("scores:\n", r.boxes.conf)
    print("cls:\n", r.boxes.cls)
    save_json_result(r)

# --- пакетная обработка папки ---
src_dir = "./defective"  # твоя папка
if os.path.isdir(src_dir):
    t_start = time.time()
    results = model.predict(
        source=src_dir,
        imgsz=IMGSZ,
        conf=CONF,
        device=device_ultra,
        save=True,
        save_txt=True,
        save_crop=True,
        stream=False,
        verbose=True,
        half=use_half
    )

    # CSV-итог по всем изображениям
    csv_path = os.path.join("reports", "detections.csv")
    names = model.names
    per_class = Counter()
    per_image_counts = defaultdict(int)
    no_det_images = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])
        for r in results:
            img_name = os.path.basename(r.path)
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None or len(boxes) == 0:
                # логируем отсутствие детекций отдельной строкой
                w.writerow([img_name, -1, "no_detections", 0.0, "", "", "", ""])
                no_det_images.append(img_name)
                continue

            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                cls = int(cls)
                w.writerow([img_name, cls, names.get(cls, str(cls)), float(conf), *[float(v) for v in xyxy]])
                per_class[cls] += 1
                per_image_counts[img_name] += 1

            # JSON на каждое изображение
            save_json_result(r)

    t_total = time.time() - t_start

    # сводка
    total_imgs = len(set(list(per_image_counts.keys()) + no_det_images))
    total_dets = sum(per_class.values())
    avg_pp = []  # средние тайминги по картинкам
    try:
        # results в новых версиях — list[Results]; возьмем среднее по speed
        for r in results:
            s = r.speed
            avg_pp.append((s["preprocess"], s["inference"], s["postprocess"]))
        if avg_pp:
            p = sum(x[0] for x in avg_pp)/len(avg_pp)
            i = sum(x[1] for x in avg_pp)/len(avg_pp)
            q = sum(x[2] for x in avg_pp)/len(avg_pp)
        else:
            p = i = q = 0.0
    except Exception:
        p = i = q = 0.0

    print("\n=== Summary ===")
    print(f"Images processed: {total_imgs}")
    print(f"Detections total: {total_dets}")
    if per_class:
        print("Per-class counts:")
        for cid, cnt in sorted(per_class.items()):
            print(f"  {cid} ({names.get(cid, cid)}): {cnt}")
    if no_det_images:
        print(f"No detections on {len(no_det_images)} images (logged in CSV).")
    print(f"Average speed (ms): preprocess={p:.1f}, inference={i:.1f}, postprocess={q:.1f}")
    if total_imgs > 0:
        print(f"Approx FPS: {1000.0/(p+i+q):.2f}")
    print("CSV saved:", csv_path)

print("Done. Visualizations, txt, and crops are in 'runs/predict-*' (или 'runs/detect-*').")
