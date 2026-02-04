import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import os 
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load YOLOv8n model
model = YOLO("yolov8n.pt")
model.fuse()  # CHANGED: fuse model layers once for slightly faster inference

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(2)

# Set camera to capture at low resolution natively (no software downscaling)
CAPTURE_WIDTH = 426
CAPTURE_HEIGHT = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CHANGED: reduce internal buffer to cut latency

# Performance tuning
#DETECT_CONF = 0.1
DETECT_CONF = 0.5  # Lower threshold for more detections (trades speed for coverage)
DETECT_IOU = 0.5
#MODEL_IMGSZ = 320  # CHANGED: lower inference resolution to reduce CPU load (multiple of 32)
MODEL_IMGSZ = 288  # Alternative: even smaller for maximum speed boost (~10% faster)
DISPLAY_SCALE = 2  # scale up for better visibility
BOX_THICKNESS = 1  # thinner boxes for low resolution
TEXT_THICKNESS = 1

# Dictionary to store trajectory points for each track ID
track_history = defaultdict(list)

# Max number of points to keep per trajectory
MAX_TRAIL_LENGTH = 30
# MAX_TRAIL_LENGTH = 20  # Alternative: reduce trajectory memory (~5-10% speed improvement)

# Predefined color palette (BGR) for nice, distinct colors
COLOR_PALETTE = [
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (0, 128, 255),   # Orange
    (128, 0, 255),   # Purple
    (255, 128, 0),   # Light Blue
    (128, 255, 0),   # Lime
]

# Dictionary to store a consistent color per track ID
track_colors = {}


def get_track_color(track_id: int):
    if track_id not in track_colors:
        track_colors[track_id] = COLOR_PALETTE[track_id % len(COLOR_PALETTE)]
    return track_colors[track_id]


# OPTIONAL: Threading for faster frame capture (uncomment if CPU is maxed)
import threading
frame_buffer = None
lock = threading.Lock()

def capture_frames():
    global frame_buffer
    while cap.isOpened():
        ret, frame = cap.read()
        with lock:
            frame_buffer = frame if ret else None

capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

while cap.isOpened():
    # ALTERNATIVE: For threaded capture, use this instead:
    with lock:
        frame = frame_buffer
    if frame is None:
        break
    
    # ret, frame = cap.read()
    # if not ret:
    #     break

    # Run YOLOv8 tracking (ByteTrack is used internally)
    results = model.track(
        frame,
        persist=True,
        classes=[0],      # class 0 = person
        conf=DETECT_CONF,
        iou=DETECT_IOU,
        imgsz=MODEL_IMGSZ,  # CHANGED: use reduced inference size for speed
        verbose=False
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id

        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Calculate centroid
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Store centroid history
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > MAX_TRAIL_LENGTH:
                    track_history[track_id].pop(0)

                # Get consistent color for this track ID
                color = get_track_color(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

                # Draw ID label
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    TEXT_THICKNESS
                )

                # Draw trajectory line
                points = track_history[track_id]
                if len(points) >= 2:
                    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))  # CHANGED: vectorized polyline
                    cv2.polylines(frame, [pts], False, color, BOX_THICKNESS)  # CHANGED: faster than per-segment lines

    # Apply display scale for zoomed preview
    display_frame = cv2.resize(
        frame,
        (CAPTURE_WIDTH * DISPLAY_SCALE, CAPTURE_HEIGHT * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imshow("YOLOv8 Person Tracking + Trajectory", display_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
