from src.analytics.line_crossing import LineCrossingDetector


def test_crossing():

    detector = LineCrossingDetector(
        line_start=(0, 100),
        line_end=(100, 100),
    )

    # Start above the line
    assert detector.update(
        1,
        (50, 50)
    ) is None

    # Move below the line
    result = detector.update(
        1,
        (50, 150)
    )

    assert result == "entry"
