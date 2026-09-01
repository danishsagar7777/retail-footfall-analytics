import time


class FPSCounter:

    def __init__(self):

        self.start_time = time.monotonic()
        self.frames = 0

    def update(self):

        self.frames += 1

    @property
    def fps(self):

        elapsed = time.monotonic() - self.start_time

        if elapsed <= 0:
            return 0.0

        return self.frames / elapsed
