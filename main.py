import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Step 1: Load YOLO model (pre-trained COCO dataset)
model = YOLO("yolov8n.pt")  # YOLOv8 nano model (fast & small)

# Step 2: Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Step 3: Open video file or webcam
cap = cv2.VideoCapture(0)  # अगर webcam चाहिए तो cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 4: Object Detection with YOLO
    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # DeepSORT format: [x1, y1, x2, y2, confidence, class]
        detections.append(([x1, y1, x2, y2], conf, label))

    # Step 5: Tracking with DeepSORT
    tracks = tracker.update_tracks(detections, frame=frame)

    # Step 6: Draw results on frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Step 7: Show output
    cv2.imshow("Object Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
