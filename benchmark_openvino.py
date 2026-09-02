cat > scripts/benchmark_openvino.py <<'EOF'
import os
import time

from ultralytics import YOLO

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame


MODEL_PATH = "yolov8n_openvino_model"


def main():

    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    print("Loading OpenVINO YOLO model...")
    model = YOLO(MODEL_PATH)

    reader = RTSPReader(rtsp_url)

    print("Connecting to RTSP stream...")
    reader.connect()

    print("RTSP connected.")
    print("Benchmarking OpenVINO YOLO...")
    print("Press Ctrl+C to stop.")

    frames = 0
    start_time = time.monotonic()

    try:

        while True:

            # Read frame
            ret, frame = reader.read()

            if not ret:
                continue

            # Resize 3200x1800 -> 1280x720
            frame = resize_frame(frame)

            # OpenVINO inference
            model(
                frame,
                imgsz=640,
                classes=[0],
                conf=0.40,
                verbose=False,
            )

            frames += 1

            if frames % 100 == 0:

                elapsed = time.monotonic() - start_time

                fps = frames / elapsed

                print(
                    f"Frames: {frames} | "
                    f"OpenVINO YOLO FPS: {fps:.2f}"
                )

    except KeyboardInterrupt:

        elapsed = time.monotonic() - start_time

        print("\nStopping...")

        if elapsed > 0:
            print(
                f"Final OpenVINO YOLO FPS: "
                f"{frames / elapsed:.2f}"
            )

    finally:

        reader.release()


if __name__ == "__main__":
    main()
EOF
