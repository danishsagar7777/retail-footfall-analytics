import os
import cv2

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

    line_start = (200, 400)
    line_end = (1100, 400)

    try:

        for _ in range(100):

            ret, frame = reader.read()

            if not ret:
                continue

            frame = resize_frame(frame)

            tracks = tracker.update(frame)

            cv2.line(
                frame,
                line_start,
                line_end,
                (0, 255, 255),
                3,
            )

            for track in tracks:

                x1, y1, x2, y2 = map(
                    int,
                    track.bbox
                )

                cx, cy = map(
                    int,
                    track.center
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"ID: {track.track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imwrite(
                "tracking_debug.jpg",
                frame
            )

            print("Saved tracking_debug.jpg")
            break

    finally:
        reader.release()


if __name__ == "__main__":
    main()
