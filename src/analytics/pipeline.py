import time

from src.analytics.line_crossing import LineCrossingDetector
from src.analytics.footfall_counter import FootfallCounter
from src.analytics.event_logger import EventLogger


class FootfallPipeline:
    def __init__(
        self,
        counting_line,
        log_file="logs/footfall_events.csv",
        event_cooldown=2.0,
        line_buffer=12.0,
    ):
        self.crossing_detector = LineCrossingDetector(
            counting_line,
            buffer=line_buffer,
        )

        self.counter = FootfallCounter()
        self.logger = EventLogger(log_file)

        self.event_cooldown = float(event_cooldown)
        self.last_event = {}

    def update(self, tracks):
        events = []
        current_time = time.monotonic()

        active_track_ids = {
            track.track_id
            for track in tracks
        }

        for track in tracks:
            event = self.crossing_detector.update(
                track.track_id,
                track.center,
            )

            if not event:
                continue

            track_id = track.track_id
            previous_event = self.last_event.get(track_id)

            if previous_event is not None:
                last_time, last_event_type = previous_event
                elapsed = current_time - last_time

                if (
                    last_event_type == event
                    and elapsed < self.event_cooldown
                ):
                    continue

            self.last_event[track_id] = (
                current_time,
                event,
            )

            self.counter.update(event)
            self.logger.log(track_id, event)

            events.append(
                {
                    "track_id": track_id,
                    "event": event,
                }
            )

        self.crossing_detector.cleanup(active_track_ids)

        stale_event_ids = [
            track_id
            for track_id in self.last_event
            if track_id not in active_track_ids
        ]

        for track_id in stale_event_ids:
            del self.last_event[track_id]

        return events

    def summary(self):
        return self.counter.summary()

