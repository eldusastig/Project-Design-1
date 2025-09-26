#!/usr/bin/env python3
"""
yolo_debris_service.py - safe serial with reconnect
Improvements: lightweight tiled re-check for cluttered/overlapping objects,
reduced default frame size for Raspberry Pi 4, and NMS merging of tile results.
"""

import cv2
import serial
import json
import time
import logging
import signal
import sys
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from collections import deque
from serial import SerialException
import threading

# ----------------------- CONFIGURATION -----------------------
MODEL_PATH = "/home/Team23/Desktop/ProjectDesignMain/Project-Design-1/Weights/Tac02.pt"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DETECTION_THRESHOLD = 2
CONF_THRESHOLD = 0.25
LOG_FILE = "/home/Team23/Desktop/ProjectDesignMain/Project-Design-1/yolo_debris_service.log"
COOLDOWN = 3 # seconds
MAX_TRACK_MEMORY = 50
IOU_THRESHOLD = 0.3

# Camera / resizing (Pi4: keep small to save CPU)
CAP_DEVICE = 0
MAX_SIDE = 320  # smaller default on Pi4 for better throughput

# Tiling fallback (only used when frame detections < DETECTION_THRESHOLD)
TILE_SIZE = 320        # tile size in pixels (works on resized frame)
TILE_OVERLAP = 0.25    # fraction overlap between tiles
TILE_CONF = 0.20       # per-tile confidence threshold (lower so small/cluttered objects can be found)
TILE_NMS_IOU = 0.35    # IoU for merging tile boxes
MAX_TILED_RUNS_PER_MIN = 6  # safety cap to avoid overusing tile mode (per minute)
# Serial reconnect params
SERIAL_RECONNECT_MAX_RETRIES = None   # None => retry forever
SERIAL_RECONNECT_BASE_DELAY = 0.5     # seconds
SERIAL_RECONNECT_MAX_DELAY = 5.0      # seconds
# -------------------------------------------------------------

# Initialize logging for systemd (journald) and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # journald/systemd
        logging.FileHandler(LOG_FILE)
    ]
)

# For debugging turn on:
# logging.getLogger().setLevel(logging.DEBUG)

stop_requested = False
paused = False

def signal_handler(sig, frame):
    global stop_requested
    logging.info("Shutdown signal received, stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------- Threaded camera capture & helpers --------------------
class CamGrab(threading.Thread):
    """Non-blocking camera capture running in a background thread."""
    def __init__(self, src=0, width=None, height=None):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, cv2.CAP_ANY)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False

    def run(self):
        tries = 0
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                if ok and frame is not None:
                    self.frame = frame
            if not ok:
                tries += 1
                time.sleep(0.05 if tries < 10 else 0.5)
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ok, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

def resize_keep_aspect(frame, max_side):
    h, w = frame.shape[:2]
    maxc = max(h, w)
    if maxc <= max_side:
        return frame, 1.0
    scale = max_side / float(maxc)
    neww = int(w * scale)
    newh = int(h * scale)
    return cv2.resize(frame, (neww, newh)), scale

# ----------------------- SERIAL helper with reconnect --------------------
class SerialManager:
    def __init__(self, port, baud, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._open_attempts = 0
        self.open_serial_blocking()

    def open_serial_blocking(self):
        delay = SERIAL_RECONNECT_BASE_DELAY
        attempts = 0
        while True:
            try:
                logging.info("Opening serial %s @ %d ...", self.port, self.baud)
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout, write_timeout=1)
                try:
                    time.sleep(0.05)
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
                logging.info("Serial opened: %s", self.port)
                return
            except Exception as e:
                attempts += 1
                logging.warning("Failed to open serial (%s): %s", self.port, e)
                if SERIAL_RECONNECT_MAX_RETRIES is not None and attempts >= SERIAL_RECONNECT_MAX_RETRIES:
                    raise
                logging.info("Retrying serial open in %.1fs...", delay)
                time.sleep(delay)
                delay = min(delay * 1.7, SERIAL_RECONNECT_MAX_DELAY)

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def reconnect(self):
        logging.info("Reconnecting serial %s ...", self.port)
        try:
            self.close()
            self.open_serial_blocking()
            return True
        except Exception as e:
            logging.error("Serial reconnect failed: %s", e)
            return False

    def safe_write(self, data_bytes):
        if not self.ser:
            logging.warning("Serial not open; attempting to open before write.")
            try:
                self.open_serial_blocking()
            except Exception as e:
                logging.error("Could not open serial before write: %s", e)
                return False

        attempt = 0
        delay = SERIAL_RECONNECT_BASE_DELAY
        while True:
            try:
                self.ser.write(data_bytes)
                try:
                    self.ser.flush()
                except Exception:
                    pass
                return True
            except SerialException as e:
                logging.error("Serial write error: %s", e)
                time.sleep(delay)
                success = self.reconnect()
                if not success:
                    attempt += 1
                    if SERIAL_RECONNECT_MAX_RETRIES is not None and attempt >= SERIAL_RECONNECT_MAX_RETRIES:
                        logging.error("Exceeded serial reconnect attempts.")
                        return False
                    delay = min(delay * 1.7, SERIAL_RECONNECT_MAX_DELAY)
                    continue
                else:
                    try:
                        self.ser.write(data_bytes)
                        try:
                            self.ser.flush()
                        except Exception:
                            pass
                        return True
                    except SerialException as e2:
                        logging.error("Serial write still failing after reconnect: %s", e2)
                        attempt += 1
                        if SERIAL_RECONNECT_MAX_RETRIES is not None and attempt >= SERIAL_RECONNECT_MAX_RETRIES:
                            return False
                        time.sleep(delay)
                        delay = min(delay * 1.7, SERIAL_RECONNECT_MAX_DELAY)
                        continue
            except Exception as e:
                logging.exception("Unexpected error during serial write: %s", e)
                return False

    def safe_readline(self):
        if not self.ser:
            return None
        try:
            if self.ser.in_waiting == 0:
                return None
            raw = self.ser.readline()
            if not raw:
                return None
            try:
                return raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                return None
        except SerialException as e:
            logging.error("Serial read error: %s", e)
            self.reconnect()
            return None
        except Exception as e:
            logging.exception("Unexpected error during serial read: %s", e)
            return None

# ----------------------- UTILITIES -----------------------
def preprocess_frame(frame):
    # CLAHE + sharpening (keeps your original improvements)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    # small denoise can help clustering separation in noisy Pi captures
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
    return frame

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    inter_x1, inter_y1 = max(x1, x1b), max(y1, y1b)
    inter_x2, inter_y2 = min(x2, x2b), min(y2, y2b)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = ((x2 - x1) * (y2 - y1)) + ((x2b - x1b) * (y2b - y1b)) - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def nms_merge(boxes, scores, iou_thresh=0.45, score_thresh=0.0):
    """Return indices of boxes kept after NMS (cv2.dnn.NMSBoxes)."""
    if len(boxes) == 0:
        return []
    boxes_xywh = []
    for b in boxes:
        x1, y1, x2, y2 = b
        boxes_xywh.append([float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))])
    # cv2.dnn.NMSBoxes returns indices
    try:
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_thresh, iou_thresh)
    except Exception:
        # fallback to simple greedy NMS if cv2 fails
        idxs = []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        for i in order:
            keep_i = True
            for j in keep:
                if compute_iou(boxes[i], boxes[j]) > iou_thresh:
                    keep_i = False
                    break
            if keep_i:
                keep.append(i)
        return keep
    keep = []
    if isinstance(idxs, (list, tuple)):
        # e.g. list of lists on some builds
        for it in idxs:
            if isinstance(it, (list, tuple, np.ndarray)):
                if len(it) > 0:
                    keep.append(int(it[0]))
            else:
                keep.append(int(it))
    elif isinstance(idxs, np.ndarray):
        keep = idxs.flatten().astype(int).tolist()
    else:
        try:
            keep = list(idxs)
        except Exception:
            keep = []
    return keep

# ----------------------- Inference helpers -----------------------
def run_model_on_image(model, img, conf):
    """Run model on a single image (numpy BGR). Return boxes, scores, classes."""
    try:
        results = model(img, conf=conf, verbose=False)
    except Exception as e:
        logging.exception("Model inference error: %s", e)
        return [], [], []
    r = results[0]
    boxes = []
    scores = []
    classes = []
    try:
        bobj = r.boxes
        if bobj is None or len(bobj) == 0:
            return [], [], []
        # unify types
        xyxy = bobj.xyxy
        confs = bobj.conf
        clss = bobj.cls
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
            confs = confs.cpu().numpy()
            clss = clss.cpu().numpy()
        else:
            xyxy = np.array(xyxy)
            confs = np.array(confs)
            clss = np.array(clss)
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i][:4].astype(int).tolist()
            confv = float(confs[i])
            clsid = int(clss[i])
            boxes.append((x1, y1, x2, y2))
            scores.append(confv)
            classes.append(clsid)
    except Exception as e:
        logging.exception("Failed to extract boxes from results: %s", e)
    return boxes, scores, classes

def make_tiles(frame_w, frame_h, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    step = int(tile_size * (1.0 - overlap))
    if step <= 0:
        step = tile_size // 2
    xs = list(range(0, max(1, frame_w - tile_size + 1), step))
    ys = list(range(0, max(1, frame_h - tile_size + 1), step))
    if len(xs) == 0 or xs[-1] + tile_size < frame_w:
        xs.append(max(0, frame_w - tile_size))
    if len(ys) == 0 or ys[-1] + tile_size < frame_h:
        ys.append(max(0, frame_h - tile_size))
    tiles = []
    for y in ys:
        for x in xs:
            w = min(tile_size, frame_w - x)
            h = min(tile_size, frame_h - y)
            tiles.append((x, y, w, h))
    return tiles

def tile_inference_and_merge(model, small_img):
    """Run inference on tiles of small_img, map results to small_img coords and merge via NMS."""
    h, w = small_img.shape[:2]
    tiles = make_tiles(w, h, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    all_boxes = []
    all_scores = []
    all_classes = []
    for (tx, ty, tw, th) in tiles:
        crop = small_img[ty:ty+th, tx:tx+tw]
        if crop is None or crop.size == 0:
            continue
        # run model on crop with lower conf so we catch small/cluttered objects
        boxes, scores, classes = run_model_on_image(model, crop, conf=TILE_CONF)
        for b, s, c in zip(boxes, scores, classes):
            x1, y1, x2, y2 = b
            # map to small_img coordinates
            x1f, y1f, x2f, y2f = x1 + tx, y1 + ty, x2 + tx, y2 + ty
            # clamp
            x1f = max(0, min(w-1, x1f)); y1f = max(0, min(h-1, y1f))
            x2f = max(0, min(w-1, x2f)); y2f = max(0, min(h-1, y2f))
            if x2f <= x1f or y2f <= y1f:
                continue
            all_boxes.append((int(x1f), int(y1f), int(x2f), int(y2f)))
            all_scores.append(float(s))
            all_classes.append(int(c))
    if len(all_boxes) == 0:
        return [], [], []
    # NMS merge with tile-level IoU threshold
    keep = nms_merge(all_boxes, all_scores, iou_thresh=TILE_NMS_IOU, score_thresh=TILE_CONF)
    kept_boxes = [all_boxes[i] for i in keep]
    kept_scores = [all_scores[i] for i in keep]
    kept_classes = [all_classes[i] for i in keep]
    return kept_boxes, kept_scores, kept_classes

# ----------------------- MAIN LOGIC -----------------------
def main(show=False):
    global paused

    logging.info("Starting YOLO debris detection service...")
    model = YOLO(MODEL_PATH)
    logging.info("Loaded model %s", MODEL_PATH)

    # Serial manager (handles reconnects)
    serman = None
    try:
        serman = SerialManager(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        logging.error("Failed to initialize SerialManager: %s", e)
        serman = None

    # Camera: use threaded grabber
    grab = CamGrab(CAP_DEVICE)
    grab.start()

    t_start = time.time()
    ok, frame = False, None
    while True:
        ok, frame = grab.read()
        if ok and frame is not None:
            break
        if time.time() - t_start > 5.0:
            logging.warning("Warning: camera didn't produce a frame within 5s. Check CAP_DEVICE.")
            break
        time.sleep(0.01)

    last_collect_time = 0
    tracked_boxes = deque(maxlen=MAX_TRACK_MEMORY)
    external_detections = 0
    tiled_runs = deque()  # timestamps of latest tiled runs (rate-limited)

    while not stop_requested:
        # read serial input
        if serman:
            line = serman.safe_readline()
            if line:
                parsed_json = None
                try:
                    parsed_json = json.loads(line)
                except Exception:
                    parsed_json = None

                # If not directly JSON, attempt to extract JSON substring (e.g. "Published detection: {...}")
                if parsed_json is None:
                    try:
                        idx = line.find('{')
                        if idx != -1:
                            sub = line[idx:]
                            parsed_json = json.loads(sub)
                    except Exception:
                        parsed_json = None

                if parsed_json:
                    try:
                        # existing event handling (appearance events)
                        evt = parsed_json.get("event")
                        if evt == "appearance":
                            sensor = parsed_json.get("sensor", "unknown")
                            dist = parsed_json.get("dist_cm", None)
                            ts = parsed_json.get("ts", None)
                            external_detections += 1
                            logging.info("External appearance from %s dist=%s ts=%s -> external_detections=%d",
                                         sensor, str(dist), str(ts), external_detections)
                        # new: accept YOLO detection log shape and treat it as external detections
                        elif ("frame_detected" in parsed_json) or ("unique_detected" in parsed_json):
                            # prefer unique_detected if present
                            ud = int(parsed_json.get("unique_detected", parsed_json.get("frame_detected", 0) or 0))
                            # add to our external counter
                            external_detections += ud
                            logging.info("Received detection log from RasPi: frame_detected=%s unique_detected=%s classes=%s -> added %d external_detections (now %d)",
                                         str(parsed_json.get("frame_detected")),
                                         str(parsed_json.get("unique_detected")),
                                         str(parsed_json.get("classes")), ud, external_detections)
                        else:
                            logging.info("RX JSON (unhandled): %s", line)
                    except Exception:
                        logging.exception("Error handling JSON serial line: %s", line)
                else:
                    up = line.strip().upper()
                    if up == "DONE":
                        if paused:
                            logging.info("Received DONE -> resuming")
                            paused = False
                            tracked_boxes.clear()
                            external_detections = 0
                        else:
                            logging.info("Received DONE but not paused")
                    else:
                        logging.info("RX (before frame): %s", line)

        if paused:
            time.sleep(0.05)
            continue

        ok, frame = grab.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # preprocess
        frame = preprocess_frame(frame)

        # resize to speed up inference
        small, scale = resize_keep_aspect(frame, MAX_SIDE)

        # inference on resized frame
        boxes, scores, classes = run_model_on_image(model, small, conf=CONF_THRESHOLD)
        frame_boxes = boxes
        frame_scores = scores
        frame_classes = classes

        # --- Log model detections on resized frame so journalctl shows them ---
        try:
            if len(frame_boxes) > 0:
                dets = [f"{model.names.get(c, str(c))}:{s:.2f}@{b}" for b, s, c in zip(frame_boxes, frame_scores, frame_classes)]
                logging.info("Model detections (resized): %s", "; ".join(dets))
            else:
                logging.info("Model detections (resized): none")
        except Exception:
            logging.exception("Failed to build detection debug string")

        frame_count = len(frame_boxes)
        detected_classes_names = [model.names.get(c, str(c)) for c in frame_classes]

        # If frame detections are fewer than the threshold, try the tiled fallback (rate-limited)
        nowt = time.time()
        tiled_allowed = True
        # purge old timestamps older than 60s
        while tiled_runs and tiled_runs[0] < nowt - 60.0:
            tiled_runs.popleft()
        if len(tiled_runs) >= MAX_TILED_RUNS_PER_MIN:
            tiled_allowed = False

        if frame_count < DETECTION_THRESHOLD and tiled_allowed:
            logging.debug("Frame detections (%d) < threshold (%d); running tiled fallback...",
                          frame_count, DETECTION_THRESHOLD)
            tiled_boxes, tiled_scores, tiled_classes = tile_inference_and_merge(model, small)
            tiled_runs.append(nowt)
            if len(tiled_boxes) > 0:
                # Merge frame boxes + tiled boxes and NMS
                merged_boxes = frame_boxes[:] + tiled_boxes
                merged_scores = frame_scores[:] + tiled_scores
                merged_classes = frame_classes[:] + tiled_classes
                keep = nms_merge(merged_boxes, merged_scores, iou_thresh=0.45, score_thresh=min(CONF_THRESHOLD, TILE_CONF))
                merged_boxes = [merged_boxes[i] for i in keep]
                merged_scores = [merged_scores[i] for i in keep]
                merged_classes = [merged_classes[i] for i in keep]
                frame_boxes = merged_boxes
                frame_scores = merged_scores
                frame_classes = merged_classes
                detected_classes_names = [model.names.get(c, str(c)) for c in frame_classes]
                frame_count = len(frame_boxes)
                logging.info("After tiled merge: frame_count=%d", frame_count)

                # --- Log merged/tiled detections as well ---
                try:
                    if len(frame_boxes) > 0:
                        dets2 = [f"{model.names.get(c, str(c))}:{s:.2f}@{b}" for b, s, c in zip(frame_boxes, frame_scores, frame_classes)]
                        logging.info("Model detections (after tiled merge): %s", "; ".join(dets2))
                    else:
                        logging.info("Model detections (after tiled merge): none")
                except Exception:
                    logging.exception("Failed to build detection debug string after tiled merge")
            else:
                logging.debug("Tiled pass found nothing new.")

        # Unique count using IoU against tracked history (existing logic)
        debris_count = 0
        for cb in frame_boxes:
            if not any(compute_iou(cb, tb) > IOU_THRESHOLD for tb in tracked_boxes):
                tracked_boxes.append(cb)
                debris_count += 1

        # build log
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "frame_detected": len(frame_boxes),
            "unique_detected": debris_count,
            "threshold": DETECTION_THRESHOLD,
            "classes": detected_classes_names
        }
        payload = json.dumps(log_data, ensure_ascii=True)
        logging.info(payload)

        # send logs to ESP32 using safe_write (log the payload and the result)
        if serman:
            try:
                logging.info("Writing detection payload to serial: %s", payload)
                ok_write = serman.safe_write((payload + "\n").encode('utf-8'))
                logging.info("Serial write result: %s", ok_write)
                if not ok_write:
                    logging.warning("Failed to write detection payload to serial (will retry later)")
            except Exception as e:
                logging.exception("Exception while writing detection payload to serial: %s", e)

        # Decision: use frame_count (immediate) + external detections
        total_count = len(frame_boxes) + external_detections
        logging.debug("DECISION: frame_count=%d external=%d total=%d", len(frame_boxes), external_detections, total_count)

        if total_count >= DETECTION_THRESHOLD and not paused:
            now_ts = time.time()
            if now_ts - last_collect_time >= COOLDOWN:
                logging.info("Threshold reached (total=%d), attempting COLLECT...", total_count)
                write_ok = False
                if serman:
                    try:
                        logging.info("Sending COLLECT to serial")
                        write_ok = serman.safe_write(b"COLLECT\n")
                        logging.info("safe_write(COLLECT) returned: %s", write_ok)
                    except Exception as e:
                        logging.exception("safe_write raised exception: %s", e)
                        write_ok = False
                else:
                    logging.warning("No serial manager (serman is None). Cannot write to ESP32.")
                if write_ok:
                    paused = True
                    last_collect_time = now_ts
                    external_detections = 0
                    tracked_boxes.clear()
                    logging.info("Sent COLLECT -> paused=True, cleared tracked history.")
                else:
                    logging.warning("COLLECT not sent (serial write failed). Not pausing; will retry after cooldown.")
            else:
                logging.debug("Cooldown active; waiting.")

        time.sleep(0.01)

    # cleanup
    try:
        grab.stop()
    except Exception:
        pass
    if serman:
        serman.close()
    if show:
        cv2.destroyAllWindows()
    logging.info("Service stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show camera output for debugging")
    args = parser.parse_args()
    main(show=args.show)
