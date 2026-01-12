import cv2
import numpy as np

cap = cv2.VideoCapture("sampe2.mp4")

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5,5), 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Frame", frame)
    #cv2.imshow("Diff", diff)
    #cv2.imshow("Thresh", thresh)
    #cv2.imshow("Dilated", dilated)

    prev_gray = gray.copy()

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
