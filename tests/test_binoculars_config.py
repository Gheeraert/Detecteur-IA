import pytest

from statistical.binoculars.config import get_thresholds


def test_known_pair_returns_calibrated_thresholds():
    thresholds = get_thresholds("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct")
    assert 0 < thresholds.low_fpr < 1
    assert 0 < thresholds.accuracy < 1


def test_unregistered_pair_raises_instead_of_silently_reusing_falcon_threshold():
    with pytest.raises(ValueError, match="No calibrated threshold"):
        get_thresholds("some/other-observer", "some/other-performer")
