#!/usr/bin/env python3
"""
yolo_debris_service.py - safe serial with reconnect
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

# ----------------------- CONFIGURATION -----------------------
MODEL_PATH = "/home/Team23/ProjectDesignMain/Project-Design-1/Weights/main.pt"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DETECTION_THRESHOLD = 4
CONF_THRESHOLD = 0.5
LOG_FILE = "/var/log/yolo_debris_service.log"
COOLDOWN = 10  # seconds
MAX_TRACK_MEMORY = 50
IOU_THRESHOLD = 0.3

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

stop_requested = False
paused = False

def signal_handler(sig, frame):
    global stop_requested
    logging.info("Shutdown signal received, stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------- Serial helper with reconnect --------------------
class SerialManager:
    def __init__(self, port, baud, timeout=1.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None
        self._open_attempts = 0
        self.open_serial_blocking()

    def open_serial_blocking(self):
        """Try to open serial; keep retrying with backoff until success (or until max tries)."""
        delay = SERIAL_RECONNECT_BASE_DELAY
        attempts = 0
        while True:
            try:
                logging.info("Opening serial %s @ %d ...", self.port, self.baud)
                self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout, write_timeout=1)
                # flush any old data
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
        """Write bytes to serial with exception handling and reconnect."""
        if not self.ser:
            # no serial open, attempt reopen
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
                # flush to ensure immediate send; ignore errors from flush if any
                try:
                    self.ser.flush()
                except Exception:
                    pass
                return True
            except SerialException as e:
                logging.error("Serial write error: %s", e)
                # try reconnecting once with backoff
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
                    # after reconnect try write one more time
                    try:
                        self.ser.write(data_bytes)
                        try:
                            self.ser.flush()
                        except Exception:
                            pass
                        return True
                    except SerialException as e2:
                        logging.error("Serial write still failing after reconnect: %s", e2)
                        # continue loop to reconnect again
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
        """Non-blocking read of a single line (returns decoded str or None)."""
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
            # try reconnect
            self.reconnect()
            return None
        except Exception as e:
            logging.exception("Unexpected error during serial read: %s", e)
            return None

# ----------------------- UTILITIES (unchanged) -----------------------
def preprocess_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
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
    return intersection / union

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
        # proceed, but serman may be None (safe_write will handle it)
        serman = None

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera not found (device /dev/video0). Exiting.")
        return

    last_collect_time = 0
    tracked_boxes = deque(maxlen=MAX_TRACK_MEMORY)

    while not stop_requested:
        # read any serial input first (handle DONE or other lines)
        if serman:
            line = serman.safe_readline()
            if line:
                up = line.strip().upper()
                if up == "DONE":
                    if paused:
                        logging.info("Received DONE -> resuming")
                        paused = False
                        tracked_boxes.clear()
                    else:
                        logging.info("Received DONE but not paused")
                else:
                    # log other incoming lines
                    logging.info("RX (before frame): %s", line)

        if paused:
            # while paused keep checking serial for DONE
            time.sleep(0.05)
            continue

        ret, frame = cap.read()
        if not ret:
            logging.warning("camera read failed; retrying")
            time.sleep(0.05)
            continue

        frame = preprocess_frame(frame)

        try:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        except Exception as e:
            logging.error("Model inference failed: %s", e)
            time.sleep(0.05)
            continue

        detections = results[0].boxes
        current_boxes = []
        detected_classes = []
        debris_count = 0

        for box in detections:
            # many ultralytics Box objects provide `.cls`, `.conf`, `.xyxy`
            try:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
            except Exception:
                cls_name = "unknown"
            try:
                conf = float(box.conf[0])
            except Exception:
                conf = 0.0
            if conf >= CONF_THRESHOLD:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                except Exception:
                    coords = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, coords[:4])
                current_boxes.append((x1, y1, x2, y2))
                detected_classes.append(cls_name)

        # Unique count using IoU against tracked history
        for cb in current_boxes:
            if not any(compute_iou(cb, tb) > IOU_THRESHOLD for tb in tracked_boxes):
                tracked_boxes.append(cb)
                debris_count += 1

        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "frame_detected": len(current_boxes),
            "unique_detected": debris_count,
            "threshold": DETECTION_THRESHOLD,
            "classes": detected_classes
        }
        payload = json.dumps(log_data, ensure_ascii=True)

        logging.info(payload)

        # send logs to ESP32 using safe_write
        if serman:
            ok = serman.safe_write((payload + "\n").encode('utf-8'))
            if not ok:
                logging.warning("Failed to write detection payload to serial (will retry later)")

        # check threshold and send COLLECT
        if debris_count >= DETECTION_THRESHOLD and not paused:
            now = time.time()
            if now - last_collect_time >= COOLDOWN:
                logging.info("Threshold reached, sending COLLECT")
                if serman:
                    serman.safe_write(b"COLLECT\n")
                paused = True
                last_collect_time = now

        # small sleep
        time.sleep(0.01)

    # cleanup
    try:
        cap.release()
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
