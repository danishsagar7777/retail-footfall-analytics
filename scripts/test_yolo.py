import os

from ultralytics import YOLO

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame


def main():
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    print("Loading YOLO...")

    model = YOLO("yolov8n.pt")

    reader = RTSPReader(rtsp_url)
    reader.connect()

    ret, frame = reader.read()

    if not ret:
        raise RuntimeError("Could not read frame.")

    frame = resize_frame(frame)

    print("Running YOLO...")

    results = model(
        frame,
        classes=[0],
        conf=0.40,
        verbose=False
    )

    detections = results[0].boxes

    print(f"Persons detected: {len(detections)}")

    for box in detections:
        xyxy = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        print(
            f"Person | "
            f"confidence={confidence:.2f} | "
            f"box={xyxy}"
        )

    reader.release()


if __name__ == "__main__":
    main()
