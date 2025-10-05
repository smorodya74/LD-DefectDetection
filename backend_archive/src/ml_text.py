import os
import ssl
import urllib.request
from typing import List, Tuple

import cv2
import numpy as np

# ===== Настройки автодокачки EAST =====
EAST_TARBALL_URL = "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1"
EAST_MIN_BYTES = 50_000_000  # защитный минимум, чтобы отсеять битые файлы

# ===== Вспомогательные функции =====
def _download_and_extract_pb(dst_pb_path: str):
    """
    Скачивает tar.gz с моделью EAST и извлекает .pb в dst_pb_path.
    """
    os.makedirs(os.path.dirname(dst_pb_path), exist_ok=True)
    tmp_tar = dst_pb_path + ".tar.gz"
    ctx = ssl.create_default_context()
    req = urllib.request.Request(EAST_TARBALL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=180) as r, open(tmp_tar, "wb") as f:
        f.write(r.read())

    import tarfile
    with tarfile.open(tmp_tar, "r:gz") as tar:
        member_pb = None
        for m in tar.getmembers():
            if m.name.lower().endswith(".pb"):
                member_pb = m
                break
        if member_pb is None:
            raise RuntimeError("В архиве EAST не найден .pb")
        # извлекаем только .pb
        member_pb.name = os.path.basename(member_pb.name)
        tar.extract(member_pb, os.path.dirname(dst_pb_path))

    # переименуем, если имя внутри архива другое
    extracted = os.path.join(os.path.dirname(dst_pb_path), member_pb.name)
    if os.path.abspath(extracted) != os.path.abspath(dst_pb_path):
        os.replace(extracted, dst_pb_path)

    os.remove(tmp_tar)

def _decode_predictions(scores: np.ndarray, geometry: np.ndarray, score_thresh: float) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
    # scores: 1x1xH/4xW/4, geometry: 1x5xH/4xW/4
    assert scores.shape[0] == 1 and geometry.shape[0] == 1
    H, W = scores.shape[2], scores.shape[3]
    rects = []
    confidences = []
    for y in range(H):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(W):
            score = float(scores_data[x])
            if score < score_thresh:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = float(angles[x])
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = float(x0[x] + x2[x])
            w = float(x1[x] + x3[x])
            endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
            endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(score)
    return rects, confidences

def _nms(boxes: List[Tuple[int,int,int,int]], scores: List[float], nms_thresh: float) -> List[int]:
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.float32)
    x1 = boxes_np[:, 0]; y1 = boxes_np[:, 1]; x2 = boxes_np[:, 2]; y2 = boxes_np[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(np.array(scores))[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
    return keep

def boxes_to_mask(shape_hw: Tuple[int,int], boxes: List[Tuple[int,int,int,int,float]]) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), np.uint8)
    for (x1, y1, x2, y2, _) in boxes:
        cv2.rectangle(m, (x1, y1), (x2, y2), 255, thickness=-1)
    return m

# ===== Класс детектора EAST =====
class EastTextDetector:
    def __init__(self, model_path: str, input_size: int = 640, score_thresh: float = 0.5, nms_thresh: float = 0.3, min_box: int = 12):
        """
        model_path: путь к .pb (GraphDef)
        input_size: кратен 32 (320/512/640)
        """
        if input_size % 32 != 0:
            raise ValueError("input_size должен быть кратен 32 (например, 320/512/640).")

        self.input_size = int(input_size)
        self.score_thresh = float(score_thresh)
        self.nms_thresh = float(nms_thresh)
        self.min_box = int(min_box)

        # Докачка, если файла нет или он слишком мал
        need_dl = (not os.path.exists(model_path)) or (os.path.getsize(model_path) < EAST_MIN_BYTES)
        if need_dl:
            _download_and_extract_pb(model_path)

        # Проверяем итоговый размер
        if os.path.getsize(model_path) < EAST_MIN_BYTES:
            raise ValueError(f"EAST .pb повреждён или неполный: {model_path}")

        # Загружаем как TensorFlow GraphDef
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_path)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить EAST граф из {model_path}: {e}")

        # Имена выходных слоёв EAST
        self.out_layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    def detect(self, gray_or_bgr: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Возвращает список боксов (x1,y1,x2,y2,conf) в координатах исходного изображения.
        """
        if gray_or_bgr.ndim == 2:
            img = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
        else:
            img = gray_or_bgr

        H0, W0 = img.shape[:2]
        rW = W0 / float(self.input_size)
        rH = H0 / float(self.input_size)

        resized = cv2.resize(img, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(
            resized, 1.0, (self.input_size, self.input_size),
            (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        try:
            scores, geometry = self.net.forward(self.out_layers)
        except Exception:
            # некоторые сборки требуют forward() по одному имени
            scores  = self.net.forward(self.out_layers[0])
            geometry = self.net.forward(self.out_layers[1])

        rects, confs = _decode_predictions(scores, geometry, self.score_thresh)
        keep = _nms(rects, confs, self.nms_thresh)

        out = []
        for i in keep:
            (sx, sy, ex, ey) = rects[i]
            # рескейл боксов к исходнику
            x1 = max(0, int(sx * rW)); y1 = max(0, int(sy * rH))
            x2 = min(W0 - 1, int(ex * rW)); y2 = min(H0 - 1, int(ey * rH))
            if (x2 - x1 + 1) < self.min_box or (y2 - y1 + 1) < self.min_box:
                continue
            out.append((x1, y1, x2, y2, float(confs[i])))
        return out
