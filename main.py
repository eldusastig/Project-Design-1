import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
import glob

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects):
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

        for r, oid in enumerate(object_ids):
            if r not in used_rows:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        for c in range(len(rects)):
            if c not in used_cols:
                self.register(tuple(input_centroids[c]), rects[c])

        return self.objects

# --- Main Application ---
def main():
    # Initialize serial
    try:
<<<<<<< HEAD
        ser = serial.Serial('/dev/ttyUSB0',115200, timeout=0.1)
=======
        ser = serial.Serial('COM5', 9600, timeout=0.1)
>>>>>>> c83fce34125832fc3459017cadaa374d65e4cb48
        time.sleep(2)
        ser.reset_input_buffer()
        print("Serial port opened.")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        ser = None

    # Initialize camera
<<<<<<< HEAD
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
=======
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
>>>>>>> c83fce34125832fc3459017cadaa374d65e4cb48
    if not cap.isOpened():
        print("Error: Camera not found or cannot be opened.")
        return
    cv2.namedWindow("YOLOv8 Tracking & Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Tracking & Control", 800, 600)

<<<<<<< HEAD
    model = YOLO(r"Project-Design-1/Weights/best2.pt")
=======
    model = YOLO(r"Weights\best2.pt")
>>>>>>> c83fce34125832fc3459017cadaa374d65e4cb48
    tracker = CentroidTracker(max_disappeared=40, max_distance=60)
    detection_enabled = True
    print("Press 'q' to quit.")

    while True:
        # Read and display any serial data immediately
        if ser and ser.in_waiting:
            data = ser.read(ser.in_waiting)
            try:
                text = data.decode('utf-8', errors='ignore').strip()
            except Exception:
                text = str(data)
            print(f"[ESP32] {text}")
            # Re-enable detection only on DONE
            if text.strip().lower() == 'done':
                detection_enabled = True
                tracker = CentroidTracker(max_disappeared=40, max_distance=60)
                print("Re-enabled detection after DONE.")

        # Read camera frame
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame")
            time.sleep(0.1)
            continue

        if detection_enabled:
            # Preprocess
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            processed = cv2.filter2D(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), -1,
                                     np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
            # Detect
            try:
                results = model(processed, conf=0.43)[0]
            except Exception as e:
                print(f"Model error: {e}")
                continue
            rects = [tuple(map(int, box.tolist())) for box in results.boxes.xyxy]
            objects = tracker.update(rects)
            count = len(objects)

            # Draw
            vis = processed.copy()
            for _, bbox in objects.values():
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
            cv2.putText(vis, f"Count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # Send collect if needed
<<<<<<< HEAD
            if count > 0   and ser:
=======
            if count >= 3   and ser:
>>>>>>> c83fce34125832fc3459017cadaa374d65e4cb48
                ser.write(b'COLLECT\n')
                detection_enabled = False
                print("Sent COLLECT command.")
        else:
            vis = frame.copy()
            cv2.putText(vis, "Collecting...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Display
        cv2.imshow("YOLOv8 Tracking & Control", vis)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
