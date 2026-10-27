"""Webcam object detection using YOLO and a simple centroid tracker."""
import cv2
from ultralytics import YOLO

VIDEO_SOURCE = 0  # Change to "sample_video.mp4" to use a video file.
MODEL_NAME = "yolov8n.pt"  # Downloaded automatically by Ultralytics on first run.

class CentroidTracker:
    """Small educational tracker that matches detections to nearest previous centroids."""
    def __init__(self, max_distance=60):
        self.next_id = 1
        self.centroids = {}
        self.max_distance = max_distance

    def update(self, boxes):
        new_centroids = []
        for x1, y1, x2, y2, *_ in boxes:
            new_centroids.append(((int(x1+x2)//2), (int(y1+y2)//2)))
        assigned = {}
        used_ids = set()
        for centroid in new_centroids:
            best_id, best_dist = None, self.max_distance
            for object_id, old in self.centroids.items():
                if object_id in used_ids:
                    continue
                dist = ((centroid[0]-old[0])**2 + (centroid[1]-old[1])**2) ** 0.5
                if dist < best_dist:
                    best_id, best_dist = object_id, dist
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            assigned[best_id] = centroid
            used_ids.add(best_id)
        self.centroids = assigned
        return list(assigned.items())

def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam/video. Check permissions or VIDEO_SOURCE.")
    tracker = CentroidTracker()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            if confidence >= 0.35:
                boxes.append((x1, y1, x2, y2, class_id, confidence))
        tracked = tracker.update(boxes)
        # Match current detections to nearest tracked centroid for display.
        for x1, y1, x2, y2, class_id, confidence in boxes:
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            object_id = min(tracked, key=lambda item: (item[1][0]-cx)**2 + (item[1][1]-cy)**2)[0] if tracked else 0
            label = f"{model.names[class_id]} ID:{object_id} {confidence:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(frame, label, (int(x1), max(20, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        cv2.imshow("YOLO Object Detection and Tracking (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
