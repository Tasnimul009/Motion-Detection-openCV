import cv2
from ultralytics import YOLO
model = YOLO(model = "yolov8n.pt")
video = r"D:\motion detection\samples\sample1.mp4"
cap = cv2.VideoCapture(video)
while True:
    succ, frame = cap.read()
    if not succ : break
    results = model(source = frame, stream = True)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            if class_id == 0 and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img = frame, 
                              pt1 = (x1, y1), 
                              pt2 = (x2, y2),
                              color = (0, 255, 0),
                              thickness = 2)
                cv2.putText(img = frame,
                            text = f"person {confidence:.2f}",
                            org = (x1, y1-10),
                            fontFace = cv2.FONT_HERSHEY_COMPLEX,
                            fontScale = 0.6,
                            color = (255, 255, 0),
                            thickness = 2,
                            lineType = cv2.LINE_4)
    cv2.imshow(winname = "YOLO human detect",
               mat = frame)
    if cv2.waitKey(delay = 30) & 0xFF == 27: break
cap.release()
cv2.destroyAllWIndows()
