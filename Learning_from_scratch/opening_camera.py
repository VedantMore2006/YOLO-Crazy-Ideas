import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    if cv2.waitKey(1)== ord("q"):
        break
vc.release()
cv2.destroyAllWindows("preview")
vc.release()