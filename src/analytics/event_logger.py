import csv
import os
from datetime import datetime


class EventLogger:
    def __init__(self, log_file="logs/footfall_events.csv"):
        self.log_file = log_file

        directory = os.path.dirname(log_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "track_id", "event"]
                )

    def log(self, track_id, event):
        timestamp = datetime.now().isoformat(timespec="seconds")

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [timestamp, track_id, event]
            )
