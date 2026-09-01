class LineCrossingDetector:
    """
    Detect when a tracked person crosses the counting line.

    counting_line format:
        [x1, y1, x2, y2]
    """

    def __init__(self, counting_line):
        self.x1, self.y1, self.x2, self.y2 = counting_line

        # Previous side of the line for each track ID.
        self.previous_sides = {}

    def update(self, track_id, point):
        """
        Check whether a tracked person crossed the line.

        Returns:
            "entry" - person crossed in one direction
            "exit"  - person crossed in the opposite direction
            None    - no crossing
        """

        current_side = self._point_side(point)

        previous_side = self.previous_sides.get(track_id)

        # Save current position for the next frame.
        self.previous_sides[track_id] = current_side

        # First observation of this track.
        if previous_side is None:
            return None

        # Crossing from one side to the other.
        if previous_side < 0 and current_side > 0:
            return "entry"

        # Crossing from the other side back.
        if previous_side > 0 and current_side < 0:
            return "exit"

        return None

    def _point_side(self, point):
        """
        Determine which side of the counting line
        the point is located on.

        Returns:
            1
            -1
            0
        """

        px, py = point

        value = (
            (self.x2 - self.x1) * (py - self.y1)
            - (self.y2 - self.y1) * (px - self.x1)
        )

        if value > 0:
            return 1

        if value < 0:
            return -1

        return 0
