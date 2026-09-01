from src.analytics.line_crossing import LineCrossingDetector
from src.analytics.footfall_counter import FootfallCounter


class FootfallPipeline:
    def __init__(self, counting_line):

        self.crossing_detector = LineCrossingDetector(
            counting_line
        )

        self.counter = FootfallCounter()

    def update(self, tracks):

        events = []

        for track in tracks:

            event = self.crossing_detector.update(
                track.track_id,
                track.center,
            )

            if event:
                self.counter.update(event)

                events.append(
                    {
                        "track_id": track.track_id,
                        "event": event,
                    }
                )

        return events

    def summary(self):
        return self.counter.summary()
