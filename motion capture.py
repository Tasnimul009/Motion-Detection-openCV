import cv2
video = "D:\motion detection\samples\sample2.mp4"
cap = cv2.VideoCapture(video)
suc, frame = cap.read()
if not suc:
    exit()
prev_gray = cv2.cvtColor(src = frame, 
                         code = cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(src = prev_gray, 
                             ksize = (5, 5), 
                             sigmaX = 0)
while True:
    success, curr_frame = cap.read()
    if not success : break
    curr_gray = cv2.cvtColor(src = curr_frame, 
                             code = cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(src = curr_gray, 
                                 ksize = (5,5), 
                                 sigmaX = 0)
    diff = cv2.absdiff(src1 = prev_gray, 
                       src2 = curr_gray)
    threshold_value = 19
    _, threshold = cv2.threshold(src = diff, 
                                 thresh = threshold_value,
                                 maxval = 255,
                                 type = cv2.THRESH_BINARY)
    dilate = cv2.dilate(src = threshold, 
                        kernel = None,
                        iterations = 6)
    contours, _ = cv2.findContours(image = dilate, 
                                   mode = cv2.RETR_EXTERNAL,
                                   method = cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 800
    for c in contours: 
        if cv2.contourArea(c) < min_contour_area : continue 
        x, y, w, h = cv2.boundingRect(array = c)
        cv2.rectangle(img = curr_frame,
                      pt1 = (x, y),
                      pt2 = (x+w, y+h), 
                      color = (255, 192, 203),
                      thickness = 3)
    cv2.imshow(winname = 'Final', mat = curr_frame)
    cv2.imshow(winname = '2', mat = dilate)
    cv2.imshow(winname = '3', mat = threshold)
    cv2.imshow(winname = '4', mat = diff)
    prev_gray = curr_gray.copy()
    if cv2.waitKey(30) & 0xFF == 27: break
cap.release()
cv2.destroyAllWinodows()