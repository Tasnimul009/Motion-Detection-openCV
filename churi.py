import cv2
import numpy as np

# -----------------------------
# 1️⃣ Video Input
# -----------------------------
video_file_path = "video.mp4"  # Replace with your video file
video_capture = cv2.VideoCapture(filename=video_file_path)

if not video_capture.isOpened():
    print("❌ Cannot open video file")
    exit()

# -----------------------------
# 2️⃣ Read First Frame (Reference Frame)
# -----------------------------
success, first_frame = video_capture.read()
if not success:
    print("❌ Cannot read first frame from video")
    exit()

# Convert first frame to grayscale
reference_gray_frame = cv2.cvtColor(src=first_frame, code=cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
reference_gray_frame = cv2.GaussianBlur(src=reference_gray_frame, ksize=(5, 5), sigmaX=0)

# -----------------------------
# 3️⃣ Main Loop — Process Video Frames
# -----------------------------
while True:
    success, current_frame = video_capture.read()
    if not success:
        break  # End of video

    # Convert current frame to grayscale
    current_gray_frame = cv2.cvtColor(src=current_frame, code=cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    current_gray_frame = cv2.GaussianBlur(src=current_gray_frame, ksize=(5, 5), sigmaX=0)

    # -----------------------------
    # 4️⃣ Motion Detection — Frame Difference
    # -----------------------------
    frame_difference = cv2.absdiff(src1=reference_gray_frame, src2=current_gray_frame)

    # -----------------------------
    # 5️⃣ Thresholding — Binary Mask
    # -----------------------------
    threshold_value = 25
    max_binary_value = 255
    _, binary_motion_mask = cv2.threshold(
        src=frame_difference,
        thresh=threshold_value,
        maxval=max_binary_value,
        type=cv2.THRESH_BINARY
    )

    # -----------------------------
    # 6️⃣ Morphology — Clean the Mask
    # -----------------------------
    dilated_motion_mask = cv2.dilate(
        src=binary_motion_mask,
        kernel=None,
        iterations=3,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # -----------------------------
    # 7️⃣ Contours — Detect Moving Objects
    # -----------------------------
    contours, hierarchy = cv2.findContours(
        image=dilated_motion_mask,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    # -----------------------------
    # 8️⃣ Filter Small Contours & Draw Bounding Boxes
    # -----------------------------
    min_contour_area = 800  # Ignore small movements / noise
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        x, y, width, height = cv2.boundingRect(array=contour)
        cv2.rectangle(
            img=current_frame,
            pt1=(x, y),
            pt2=(x + width, y + height),
            color=(0, 255, 0),
            thickness=2
        )

    # -----------------------------
    # 9️⃣ Display Windows
    # -----------------------------
    cv2.imshow(winname="Original Video with Motion Boxes", mat=current_frame)
    cv2.imshow(winname="Frame Difference", mat=frame_difference)
    cv2.imshow(winname="Binary Mask (Threshold)", mat=binary_motion_mask)
    cv2.imshow(winname="Dilated Mask (Cleaned)", mat=dilated_motion_mask)

    # -----------------------------
    # 10️⃣ Update Reference Frame
    # -----------------------------
    reference_gray_frame = current_gray_frame.copy()

    # Exit on ESC key
    if cv2.waitKey(delay=30) & 0xFF == 27:
        break

# -----------------------------
# 11️⃣ Clean Up
# -----------------------------
video_capture.release()
cv2.destroyAllWindows()
