import os
import cv2
import threading
import time


class RTSPReader:
    """
    Threaded RTSP reader.

    Continuously receives frames in the background and keeps
    only the newest successfully decoded frame.
    """

    def __init__(self, url, reconnect_delay=2.0):
        self.url = url
        self.reconnect_delay = float(reconnect_delay)

        self.cap = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()

        self.latest_frame = None
        self.frame_id = 0
        self.last_delivered_frame_id = 0

        self.consecutive_failures = 0
        self.reconnect_count = 0
        self.dropped_frames = 0

    def connect(self):
        print("Connecting to RTSP stream...")

        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp"
        )

        cap = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
        )

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                "Could not open RTSP stream"
            )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        with self.lock:
            self.cap = cap

        print("RTSP stream connected.")

    def start(self):
        if self.running:
            return

        self.connect()

        self.running = True

        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )

        self.thread.start()

    def _reader_loop(self):
        while self.running:

            with self.lock:
                cap = self.cap

            if cap is None:
                self._reconnect()
                continue

            ret, frame = cap.read()

            if ret and frame is not None:

                with self.lock:
                    self.latest_frame = frame
                    self.frame_id += 1

                self.consecutive_failures = 0
                continue

            self.consecutive_failures += 1

            if self.consecutive_failures < 3:
                time.sleep(0.01)
                continue

            print(
                "RTSP frame read failed. "
                "Reconnecting..."
            )

            self._reconnect()

    def _reconnect(self):
        if not self.running:
            return

        self._release_capture()

        self.reconnect_count += 1

        print(
            f"RTSP reconnect attempt "
            f"#{self.reconnect_count}"
        )

        time.sleep(self.reconnect_delay)

        if not self.running:
            return

        try:
            self.connect()
            self.consecutive_failures = 0

        except RuntimeError:
            print(
                "RTSP reconnect failed. "
                "Will retry..."
            )

    def read(self):
        """
        Return only a new frame.

        If no new frame is available, return False.
        """

        with self.lock:

            if self.latest_frame is None:
                return False, None

            if (
                self.frame_id
                == self.last_delivered_frame_id
            ):
                return False, None

            frame = self.latest_frame.copy()

            previous_id = self.last_delivered_frame_id

            self.last_delivered_frame_id = self.frame_id

            if previous_id > 0:

                skipped = (
                    self.frame_id
                    - previous_id
                    - 1
                )

                if skipped > 0:
                    self.dropped_frames += skipped

        return True, frame

    def _release_capture(self):

        with self.lock:
            cap = self.cap
            self.cap = None

        if cap is not None:
            cap.release()

    def release(self):

        self.running = False

        # Wait for reader thread to leave cap.read()
        # before releasing the capture again.
        if self.thread is not None:

            self.thread.join(
                timeout=3.0
            )

            self.thread = None

        self._release_capture()

        with self.lock:
            self.latest_frame = None
            self.frame_id = 0
            self.last_delivered_frame_id = 0

        print(
            "RTSP reader stopped. "
            f"Reconnects: {self.reconnect_count}, "
            f"Skipped frames: {self.dropped_frames}"
        )
