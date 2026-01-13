import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO(model="yolov8n.pt")

# -----------------------------
# Video input
# -----------------------------
video_path = r"D:\motion detection\samples\sample2.mp4"
cap = cv2.VideoCapture(filename=video_path)

while True:
    success, frame = cap.read()
    if not success:
        break

    # -----------------------------
    # YOLO inference
    # -----------------------------
    results = model(source=frame, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # YOLO class 0 = person
            if class_id == 0 and confidence > 0.5:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # -----------------------------
                # Draw bounding box
                # -----------------------------
                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=(0, 255, 0),
                    thickness=2
                )

                # -----------------------------
                # Draw label text
                # -----------------------------
                cv2.putText(
                    img=frame,
                    text=f"Person {confidence:.2f}",
                    org=(x1, y1 - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

    # -----------------------------
    # Show output
    # -----------------------------
    cv2.imshow(
        winname="YOLO Human Detection",
        mat=frame
    )

    # -----------------------------
    # Exit on ESC
    # -----------------------------
    if cv2.waitKey(delay=30) & 0xFF == 27:
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
