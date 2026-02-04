import os
import cv2

os.environ["QT_QPA_PLATFORM"] = "xcb"

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Current capture resolution: {int(width)}x{int(height)}")

cap.release()
