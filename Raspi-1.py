#!/usr/bin/env python3
"""
yolo_tracker_service.py

Headless-capable YOLOv8 tracking + ESP32 serial control script.
Designed to run as a systemd service on Linux (default camera /dev/video0).
"""
import argparse
import logging
import time
import signal
import sys
import os
from typing import List, Tuple

import cv2
import numpy as np
import serial
from ultralytics import YOLO

# ---------------------------
# CentroidTracker (unchanged logic)
# ---------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}          # object_id -> (centroid, bbox)
        self.disappeared = {}      # object_id -> disappeared_count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects: List[Tuple[int, int, int, int]]):
        if not rects:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array([[(x1 + x2) // 2, (y1 + y2) // 2] for x1, y1, x2, y2 in rects])
        if not self.objects:
            for i, centroid in enumerate(input_centroids):
                self.register(tuple(centroid), rects[i])
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = np.array([c for c, _ in self.objects.values()])
        # distance matrix
        D = np.linalg.norm(object_centroids[:, None] - input_centroids[None, :], axis=2)

        rows, cols = np.where(D <= self.max_distance)
        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            oid = object_ids[r]
            self.objects[oid] = (tuple(input_centroids[c]), rects[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        # increment disappeared for unmatched
        for r, oid in enumerate(object_ids):
            if r not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        # register new unmatched detections
        for c in range(len(rects)):
            if c not in used_cols:
                self.register(tuple(input_centroids[c]), rects[c])

        return self.objects

# ---------------------------
# Helpers
# ---------------------------
def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_class_names(results, model):
    names_map = getattr(model, "names", None) or {}
    class_list = []
    boxes = getattr(results, "boxes", None)
    if not boxes or len(boxes) == 0:
        return []
    cls_attr = getattr(boxes, "cls", None)
    if cls_attr is None:
        return []
    # convert to python list safely
    try:
        cls_indices = cls_attr.cpu().numpy().tolist()
    except Exception:
        try:
            cls_indices = cls_attr.numpy().tolist()
        except Exception:
            try:
                cls_indices = list(cls_attr)
            except Exception:
                cls_indices = []
    for ci in cls_indices:
        try:
            idx = int(ci)
        except Exception:
            continue
        class_list.append(str(names_map[idx]) if idx in names_map else str(idx))
    return class_list

# ---------------------------
# Main
# ---------------------------
STOP = False
def _signal_handler(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

def build_argparser():
    p = argparse.ArgumentParser(description="YOLOv8 Tracking & ESP32 Control (service-friendly)")
    p.add_argument('--serial', default='/dev/ttyUSB0', help='Serial port to ESP32 (default /dev/ttyUSB0)')
    p.add_argument('--baud', type=int, default=115200, help='Serial baud rate (default 115200)')
    p.add_argument('--weights', default=os.path.join('Project-Design-1', 'Weights', 'best2.pt'),
                   help='YOLO weights path (default Project-Design-1/Weights/best2.pt)')
    p.add_argument('--conf', type=float, default=0.43, help='Detection confidence (default 0.43)')
    p.add_argument('--camera', default='/dev/video0', help='Camera device (default /dev/video0)')
    p.add_argument('--device-backend', default='v4l2', choices=['v4l2', 'default'],
                   help='Camera backend (v4l2 recommended on Linux)')
    p.add_argument('--show', action='store_true', help='Show GUI window (overrides headless).')
    p.add_argument('--logfile', default='/var/log/yolo_tracker.log', help='Optional log file (default /var/log/yolo_tracker.log)')
    p.add_argument('--collect-threshold', type=int, default=3, help='Number of tracked objects required to auto-send COLLECT (default 3)')
    return p

def open_serial(port, baud, log):
    if not port:
        return None
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        log.info("Serial opened: %s @ %d", port, baud)
        return ser
    except Exception as e:
        log.warning("Could not open serial %s: %s", port, e)
        return None

def main():
    args = build_argparser().parse_args()
    # ensure logfile directory exists
    try:
        logfile_dir = os.path.dirname(args.logfile)
        if logfile_dir and not os.path.exists(logfile_dir):
            os.makedirs(logfile_dir, exist_ok=True)
    except Exception:
        pass

    # logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.logfile:
        try:
            handlers.append(logging.FileHandler(args.logfile))
        except Exception:
            # fallback to stdout only if file cannot be opened
            pass
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', handlers=handlers)
    log = logging.getLogger('yolo-tracker')

    # Serial
    ser = open_serial(args.serial, args.baud, log)

    # Camera
    try:
        backend = cv2.CAP_V4L2 if args.device_backend == 'v4l2' else 0
        # cv2.VideoCapture accepts int index or string path - use device string
        cap = cv2.VideoCapture(args.camera, backend)
        if not cap.isOpened():
            log.error("Camera cannot be opened: %s (backend=%s)", args.camera, args.device_backend)
            return 1
    except Exception as e:
        log.error("Exception opening camera: %s", e)
        return 1

    # Model
    try:
        model = YOLO(args.weights)
    except Exception as e:
        log.error("Failed to load model weights: %s", e)
        cap.release()
        return 1

    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    detection_enabled = True
    last_sent_summary = None
    last_log_time = time.time()

    log.info("Starting main loop (headless=%s) weights=%s camera=%s", not args.show, args.weights, args.camera)

    try:
        while not STOP:
            # read serial replies immediately
            if ser:
                try:
                    if ser.in_waiting:
                        raw = ser.read(ser.in_waiting)
                        try:
                            text = raw.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            text = str(raw)
                        if text:
                            log.info("[ESP32] %s", text)
                            if text.strip().lower() == 'done':
                                detection_enabled = True
                                tracker = CentroidTracker(max_disappeared=40, max_distance=60)
                                log.info("Re-enabled detection after DONE.")
                except Exception as e:
                    log.debug("Serial read error: %s", e)

            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame from camera")
                time.sleep(0.05)
                continue
            if not detection_enabled:
                if args.show:
                    vis=frame.copy()
                    cv2.putText(vis, "Collecting... (waiting for Done)",(10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2)
                    cv2.imshow("YoloV8 Tracking & Control", vis)
                    if cv2.waitkey(1)& 0xff ==ord('q'):
                        break
                else:
                    time.sleep(0.05)
                continue     

            # Preprocess (same approach as original code)
            try:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                processed = cv2.filter2D(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR),
                                         -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
            except Exception as e:
                log.debug("Preprocess failed, using raw frame: %s", e)
                processed = frame

            # Run model
            try:
                results = model(processed, conf=args.conf)[0]
            except Exception as e:
                log.error("Model error: %s", e)
                time.sleep(0.05)
                continue

            # Build det_boxes list from results
            det_boxes = []
            try:
                xyxy = getattr(results.boxes, "xyxy", None)
                if xyxy is not None:
                    try:
                        xy_list = xyxy.cpu().numpy().tolist()
                    except Exception:
                        xy_list = [b.tolist() for b in xyxy]
                    for b in xy_list:
                        det_boxes.append(tuple(map(int, b[:4])))
            except Exception:
                det_boxes = []

            # Update tracker
            objects = tracker.update(det_boxes)
            count = len(objects)

            # Extract class names and build summary
            class_names = extract_class_names(results, model)
            unique_classes = unique_preserve_order(class_names)
            summary_line = "DETECTIONS:" + (",".join(unique_classes) if unique_classes else "none")

            # Log detailed detections
            try:
                confs = getattr(results.boxes, "conf", None)
                cls = getattr(results.boxes, "cls", None)
                conf_list, cls_list = [], []
                if confs is not None:
                    try:
                        conf_list = confs.cpu().numpy().tolist()
                    except Exception:
                        try:
                            conf_list = [float(x) for x in confs.tolist()]
                        except Exception:
                            conf_list = []
                if cls is not None:
                    try:
                        cls_list = cls.cpu().numpy().astype(int).tolist()
                    except Exception:
                        try:
                            cls_list = [int(x) for x in cls.tolist()]
                        except Exception:
                            cls_list = []

                for i, box in enumerate(det_boxes):
                    nm = "unknown"
                    if i < len(cls_list) and cls_list[i] in model.names:
                        nm = model.names[cls_list[i]]
                    elif i < len(class_names):
                        nm = class_names[i]
                    cf = conf_list[i] if i < len(conf_list) else 0.0
                    log.info("[DETECT] %s (%.2f) bbox=%s", nm, cf, box)
            except Exception:
                log.info("[DETECT] summary=%s count=%d", summary_line, count)

            # send summary when changed
            if ser and summary_line != last_sent_summary:
                try:
                    ser.write((summary_line + "\n").encode('utf-8'))
                    ser.flush()
                    last_sent_summary = summary_line
                    log.info("[SENT->ESP32] %s", summary_line)
                except Exception as e:
                    log.warning("Serial write failed: %s", e)

            # automatic COLLECT threshold (if desired)
            if detection_enabled and count >= args.collect_threshold and ser:
                try:
                    ser.write(b'COLLECT\n')
                    ser.flush()
                    detection_enabled = False
                    log.info("Sent COLLECT command (threshold=%d).", args.collect_threshold)
                except Exception as e:
                    log.warning("Failed to send COLLECT: %s", e)

            # GUI display (only if requested)
            if args.show:
                try:
                    vis = processed.copy()
                    for (x1, y1, x2, y2) in det_boxes:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    for oid, (centroid, bbox) in objects.items():
                        cv2.putText(vis, f"ID:{oid}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
                    cv2.putText(vis, summary_line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.imshow("YOLOv8 Tracking & Control", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("Quit requested via GUI.")
                        break
                except Exception as e:
                    log.debug("GUI display error: %s", e)
            else:
                # headless: throttle CPU a bit and log periodic status
                time.sleep(0.005)
                if time.time() - last_log_time > 2.0:
                    log.info("Headless status: %s | tracked=%d", summary_line, count)
                    last_log_time = time.time()

    except Exception as e:
        log.exception("Unhandled exception in main loop: %s", e)
    finally:
        log.info("Shutting down main loop.")
        try:
            cap.release()
        except Exception:
            pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if ser:
            try:
                ser.close()
            except Exception:
                pass
    return 0

if __name__ == '__main__':
    sys.exit(main())
