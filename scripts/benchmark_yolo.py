import os
import time

from ultralytics import YOLO

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame


def main():
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    model = YOLO("yolov8n.pt")

    reader = RTSPReader(rtsp_url)
    reader.connect()

    total_frames = 0
    total_time = 0.0

    print("Benchmarking YOLO...")

    while total_frames < 100:
        ret, frame = reader.read()

        if not ret:
            continue

        frame = resize_frame(frame)

        start = time.perf_counter()

        model(
            frame,
            classes=[0],
            conf=0.40,
            verbose=False
        )

        elapsed = time.perf_counter() - start

        total_time += elapsed
        total_frames += 1

    reader.release()

    avg_time = total_time / total_frames
    fps = 1.0 / avg_time

    print()
    print("========== YOLO BENCHMARK ==========")
    print(f"Frames tested : {total_frames}")
    print(f"Avg latency   : {avg_time * 1000:.2f} ms")
    print(f"Model FPS     : {fps:.2f}")
    print("=====================================")


if __name__ == "__main__":
    main()
