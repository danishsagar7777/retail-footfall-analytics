import time

from src.input.rtsp_reader import RTSPReader


RTSP_URL = (
    "rtsp://admin:Admin1234@192.168.1.165/cam/realmonitor?channel=1&subtype=00&authbasic=YWRtaW46QWRtaW4xMjMh"
)


def main():
    reader = RTSPReader(RTSP_URL)

    try:
        reader.start()

        print("Receiving frames...")
        print("Press Ctrl+C to stop.")

        frame_count = 0
        start_time = time.monotonic()

        while True:
            ret, frame = reader.read()

            if not ret:
                time.sleep(0.005)
                continue

            frame_count += 1

            elapsed = time.monotonic() - start_time

            if elapsed >= 5.0:
                fps = frame_count / elapsed

                print(
                    f"Frames: {frame_count} | "
                    f"FPS: {fps:.2f} | "
                    f"Shape: {frame.shape}"
                )

                frame_count = 0
                start_time = time.monotonic()

    except KeyboardInterrupt:
        print("\nStopping RTSP test...")

    finally:
        reader.release()


if __name__ == "__main__":
    main()
