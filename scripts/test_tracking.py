import os

from src.input.rtsp_reader import RTSPReader
from src.input.frame_processor import resize_frame
from src.tracking.tracker import PersonTracker


def main():
    rtsp_url = os.getenv("RTSP_URL")

    if not rtsp_url:
        raise RuntimeError("RTSP_URL is not set.")

    reader = RTSPReader(rtsp_url)

    tracker = PersonTracker()

    reader.connect()

    frame_count = 0

    print("Starting tracking...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = reader.read()

            if not ret:
                continue

            frame = resize_frame(frame)

            result = tracker.update(frame)

            frame_count += 1

            if frame_count % 30 == 0:

                if result.boxes.id is not None:
                    track_ids = (
                        result.boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                    print(
                        f"Frame {frame_count} | "
                        f"Track IDs: {track_ids}"
                    )

                else:
                    print(
                        f"Frame {frame_count} | "
                        "No people tracked"
                    )

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        reader.release()


if __name__ == "__main__":
    main()
