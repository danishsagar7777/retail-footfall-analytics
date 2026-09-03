import math


class LineCrossingDetector:
    """
    Detect tracked-person crossings of a single counting line.

    A small buffer around the line acts as a dead zone.
    This prevents tiny tracking fluctuations around the line
    from generating false crossing events.

    counting_line format:
        [x1, y1, x2, y2]

    Direction:
        negative side -> positive side = entry
        positive side -> negative side = exit
    """

    def __init__(self, counting_line, buffer=12.0):
        self.x1, self.y1, self.x2, self.y2 = counting_line
        self.buffer = float(buffer)

        self.line_length = math.hypot(
            self.x2 - self.x1,
            self.y2 - self.y1,
        )

        if self.line_length == 0:
            raise ValueError(
                "Counting line must have non-zero length"
            )

        self.confirmed_sides = {}

    def update(self, track_id, point):
        current_side = self._classify_position(point)

        # Person is inside the dead zone around the line.
        # Do not change state.
        if current_side == 0:
            return None

        previous_side = self.confirmed_sides.get(track_id)

        # First observation establishes the person's side.
        if previous_side is None:
            self.confirmed_sides[track_id] = current_side
            return None

        # Person remains on the same side.
        if previous_side == current_side:
            return None

        # The person moved from one confirmed side
        # to the other, so a crossing occurred.
        self.confirmed_sides[track_id] = current_side

        if previous_side < 0 and current_side > 0:
            return "entry"

        if previous_side > 0 and current_side < 0:
            return "exit"

        return None

    def cleanup(self, active_track_ids):
        """
        Remove tracking state for tracks that no longer exist.
        """
        active_track_ids = set(active_track_ids)

        stale_ids = [
            track_id
            for track_id in self.confirmed_sides
            if track_id not in active_track_ids
        ]

        for track_id in stale_ids:
            del self.confirmed_sides[track_id]

    def _classify_position(self, point):
        """
        Return:

            1  -> positive side
           -1  -> negative side
            0  -> inside the line buffer
        """
        px, py = point

        cross_product = (
            (self.x2 - self.x1) * (py - self.y1)
            - (self.y2 - self.y1) * (px - self.x1)
        )

        distance = cross_product / self.line_length

        if distance > self.buffer:
            return 1

        if distance < -self.buffer:
            return -1

        return 0
