import os
import cv2
import time

from src.input.rtsp_reader import RTSPReader


def main():
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError(
            "RTSP_URL environment variable is not set."
        )

    reader = RTSPReader(rtsp_url)
    reader.connect()

    frame_count = 0
    start_time = time.time()

    print("Receiving frames...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = reader.read()

            if not ret:
                print("Failed to read frame.")
                break

            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time

                fps = frame_count / elapsed

                height, width = frame.shape[:2]

                print(
                    f"Frames: {frame_count} | "
                    f"FPS: {fps:.2f} | "
                    f"Resolution: {width}x{height}"
                )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        reader.release()


if __name__ == "__main__":
    main()
