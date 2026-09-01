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

    try:
        while frame_count < 100:

            ret, frame = reader.read()

            if not ret:
                continue

            frame = resize_frame(frame)

            tracks = tracker.update(frame)

            frame_count += 1

            if frame_count % 30 == 0:

                print(f"\nFrame: {frame_count}")

                for track in tracks:
                    print(
                        f"ID={track.track_id} "
                        f"BBox={track.bbox} "
                        f"Center={track.center} "
                        f"Confidence={track.confidence:.2f}"
                    )

    finally:
        reader.release()


if __name__ == "__main__":
    main()
