import numpy as np
from src.flaw_detection import _extract_contiguous_flaws


def test_flaw_interval_extraction():
    times = np.linspace(0, 10.0, 100)
    # Mask active between 2.0s and 4.0s (indices 20 to 40)
    mask = np.zeros(100, dtype=bool)
    mask[20:41] = True
    diff_array = np.ones(100) * 2.5

    flaws = _extract_contiguous_flaws(
        mask=mask,
        times=times,
        category="pacing_fluency",
        flaw_name="Test Flaw",
        threshold_val=1.5,
        diff_array=diff_array,
        min_duration=0.5,
        is_ratio=True
    )

    assert len(flaws) == 1
    assert flaws[0]["start_time"] == pytest.approx(2.02, abs=0.1)
    assert flaws[0]["end_time"] == pytest.approx(4.04, abs=0.1)
    assert flaws[0]["severity"] == "Severe"