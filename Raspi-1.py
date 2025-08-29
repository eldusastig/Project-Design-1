#!/usr/bin/env python3
"""
yolo_debris_service.py

Headless YOLOv8 debris counter with:
- IoU-based unique counting
- Pause/resume after "COLLECT"
- JSON logs + serial
- Logging via Python logging (journald/systemd friendly)
- Preprocessing for better detection
- Optional GUI mode for debugging
- Runs as a systemd service on Raspberry Pi
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

# ----------------------- CONFIGURATION -----------------------
MODEL_PATH = "/home/pi/yolo_models/debris_model.pt"
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DETECTION_THRESHOLD = 4
CONF_THRESHOLD = 0.5
LOG_FILE = "/var/log/yolo_debris_service.log"
COOLDOWN = 10  # seconds
MAX_TRACK_MEMORY = 50
IOU_THRESHOLD = 0.3
# -------------------------------------------------------------

# Initialize logging for systemd (journald) and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Journald/systemd
        logging.FileHandler(LOG_FILE)       # Local log file
    ]
)

# Global flags
stop_requested = False
paused = False


def signal_handler(sig, frame):
    global stop_requested
    logging.info("Shutdown signal received, stopping gracefully...")
    stop_requested = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ----------------------- UTILITIES -----------------------
def preprocess_frame(frame):
    """Apply LAB normalization + sharpening for better low-light performance."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame


def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes."""
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

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    logging.info(f"Loaded YOLO model from {MODEL_PATH}")

    # Open serial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    logging.info(f"Opened serial on {SERIAL_PORT} at {BAUD_RATE} baud")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Camera not found!")
        return

    last_collect_time = 0
    tracked_boxes = deque(maxlen=MAX_TRACK_MEMORY)

    while not stop_requested:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = preprocess_frame(frame)

        # YOLO inference
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        detections = results[0].boxes

        current_boxes = []
        detected_classes = []
        debris_count = 0

        for box in detections:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_boxes.append((x1, y1, x2, y2))
                detected_classes.append(cls_name)

        # Unique count using IoU
        for cb in current_boxes:
            if not any(compute_iou(cb, tb) > IOU_THRESHOLD for tb in tracked_boxes):
                tracked_boxes.append(cb)
                debris_count += 1

        # Create log with class names
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_detected": len(current_boxes),
            "unique_detected": debris_count,
            "threshold": DETECTION_THRESHOLD,
            "classes": detected_classes
        }

        logging.info(json.dumps(log_data))

        # Send logs to ESP32 via serial
        ser.write((json.dumps(log_data) + "\n").encode())

        # Check threshold
        if debris_count >= DETECTION_THRESHOLD and not paused:
            now = time.time()
            if now - last_collect_time >= COOLDOWN:
                logging.info("Threshold reached, sending COLLECT to ESP32...")
                ser.write(b"COLLECT\n")
                paused = True
                last_collect_time = now

        # Handle pause/resume logic
        if paused:
            line = ser.readline().decode().strip()
            if line == "DONE":
                logging.info("Received DONE from ESP32, resuming detection...")
                paused = False
                tracked_boxes.clear()

        # Show frame if GUI mode
        if show:
            for (x1, y1, x2, y2) in current_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("YOLO Debris Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    ser.close()
    if show:
        cv2.destroyAllWindows()
    logging.info("Service stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Show camera output for debugging")
    args = parser.parse_args()
    main(show=args.show)
