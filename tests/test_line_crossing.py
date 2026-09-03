from src.analytics.line_crossing import LineCrossingDetector


def test_entry_crossing():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 50)) is None
    assert detector.update(1, (50, 150)) == "entry"


def test_exit_crossing():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 150)) is None
    assert detector.update(1, (50, 50)) == "exit"


def test_same_side_does_not_generate_event():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 50)) is None
    assert detector.update(1, (60, 60)) is None
    assert detector.update(1, (70, 70)) is None


def test_buffer_zone_does_not_generate_event():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 50)) is None

    # Inside the 12-pixel buffer.
    assert detector.update(1, (50, 105)) is None

    # Crossing is confirmed only after reaching
    # the opposite side beyond the buffer.
    assert detector.update(1, (50, 120)) == "entry"


def test_multiple_tracks_are_independent():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 50)) is None
    assert detector.update(2, (50, 150)) is None

    assert detector.update(1, (50, 150)) == "entry"
    assert detector.update(2, (50, 50)) == "exit"


def test_repeated_crossings_are_detected():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    assert detector.update(1, (50, 50)) is None
    assert detector.update(1, (50, 150)) == "entry"
    assert detector.update(1, (50, 50)) == "exit"


def test_cleanup_removes_old_track_state():
    detector = LineCrossingDetector(
        [0, 100, 100, 100],
        buffer=12,
    )

    detector.update(1, (50, 50))

    assert 1 in detector.confirmed_sides

    detector.cleanup([])

    assert 1 not in detector.confirmed_sides
