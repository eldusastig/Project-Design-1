import cv2
import numpy as np
from ultralytics import YOLO

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        # Dict: object ID -> (centroid, bbox)
        self.objects = {}
        # Counts how many consecutive frames an object has disappeared
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        # If no detections, mark all existing as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Compute input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # If no existing objects, register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(tuple(centroid), rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [c for c, _ in self.objects.values()]

            # Compute distance between existing and input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            # Match existing IDs
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = (tuple(input_centroids[col]), rects[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Unmatched existing
            for row in set(range(D.shape[0])) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # New detections
            for col in set(range(D.shape[1])) - used_cols:
                self.register(tuple(input_centroids[col]), rects[col])

        return self.objects

# ——— Setup —————————————————————————————————————————————————————
model = YOLO(r"Weights\best2.pt")
cap   = cv2.VideoCapture(0, cv2.CAP_DSHOW)
tracker = CentroidTracker(max_disappeared=40)

print("Press 'q' to quit.")

# ——— Main loop ——————————————————————————————————————————————————
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected, exiting.")
        break

    # Preprocessing: Normalize L channel and sharpen
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_norm = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab_norm = cv2.merge((l_norm, a, b))
    enhanced = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    processed = cv2.filter2D(enhanced, -1, kernel)

    # YOLO inference
    results = model(processed, conf=0.43)[0]
    rects = [tuple(box.int().tolist()) for box in results.boxes.xyxy]

    # Update tracker
    objects = tracker.update(rects)

    # Draw tracked bounding boxes and IDs
    display = processed.copy()
    for object_id, (centroid, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"ID {object_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(display, f"Count: {len(objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 + Bounding Box Tracker", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting.")
        break

cap.release()
cv2.destroyAllWindows()

