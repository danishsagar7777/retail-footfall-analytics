import threading


class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()

        self.running = False
        self.camera_connected = False
        self.model_loaded = False

        self.frames_processed = 0
        self.fps = 0.0

        self.entries = 0
        self.exits = 0
        self.occupancy = 0

        self.reconnects = 0
        self.skipped_frames = 0

        self.last_event = None

    def update(
        self,
        running=None,
        camera_connected=None,
        model_loaded=None,
        frames_processed=None,
        fps=None,
        entries=None,
        exits=None,
        occupancy=None,
        reconnects=None,
        skipped_frames=None,
        last_event=None,
    ):
        with self.lock:
            if running is not None:
                self.running = running

            if camera_connected is not None:
                self.camera_connected = camera_connected

            if model_loaded is not None:
                self.model_loaded = model_loaded

            if frames_processed is not None:
                self.frames_processed = frames_processed

            if fps is not None:
                self.fps = fps

            if entries is not None:
                self.entries = entries

            if exits is not None:
                self.exits = exits

            if occupancy is not None:
                self.occupancy = occupancy

            if reconnects is not None:
                self.reconnects = reconnects

            if skipped_frames is not None:
                self.skipped_frames = skipped_frames

            if last_event is not None:
                self.last_event = last_event

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "camera_connected": self.camera_connected,
                "model_loaded": self.model_loaded,
                "frames_processed": self.frames_processed,
                "fps": round(self.fps, 2),
                "entries": self.entries,
                "exits": self.exits,
                "occupancy": self.occupancy,
                "reconnects": self.reconnects,
                "skipped_frames": self.skipped_frames,
                "last_event": self.last_event,
            }


pipeline_state = PipelineState()
