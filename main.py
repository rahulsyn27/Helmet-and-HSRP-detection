import cv2
from ultralytics import YOLO

# Load pretrained model
model = YOLO("hemletYoloV8.pt")

# Input video
cap = cv2.VideoCapture("traffic.mp4")

# Video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

# Output video
fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec (Mac friendly)

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    fps,
    (w, h)
)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls]

            # Color logic
            if "helmet" in label.lower():
                color = (0,255,0)      # green
            elif "no" in label.lower():
                color = (0,0,255)      # red
            elif "plate" in label.lower():
                color = (255,0,0)      # blue
            else:
                color = (255,255,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    out.write(frame)

cap.release()
out.release()

print("Saved output.mp4")
