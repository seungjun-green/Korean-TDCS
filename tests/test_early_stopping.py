import pytest

from korean_math_tdcs.training.early_stopping import EarlyStopping, validation_check_steps


def test_validation_schedule_has_four_checks_per_epoch() -> None:
    steps = validation_check_steps(total_steps=339, epochs=4, checks_per_epoch=4)

    assert len(steps) == 16
    assert steps == sorted(set(steps))
    assert steps[-1] == 339
    for epoch_index in range(4):
        first = epoch_index * 4
        epoch_start = (epoch_index * 339) // 4 + 1
        epoch_end = ((epoch_index + 1) * 339) // 4
        assert all(epoch_start <= step <= epoch_end for step in steps[first : first + 4])


def test_early_stopping_stops_after_three_bad_checks() -> None:
    stopper = EarlyStopping(patience=3)

    assert stopper.update(1.0, 10) == (True, False)
    assert stopper.update(1.1, 20) == (False, False)
    assert stopper.update(1.0, 30) == (False, False)
    assert stopper.update(1.2, 40) == (False, True)
    assert stopper.best_step == 10


def test_improvement_resets_patience() -> None:
    stopper = EarlyStopping(patience=3, min_delta=0.01)

    stopper.update(1.0, 10)
    stopper.update(0.995, 20)
    improved, should_stop = stopper.update(0.98, 30)

    assert improved is True
    assert should_stop is False
    assert stopper.bad_checks == 0


def test_invalid_validation_frequency_is_rejected() -> None:
    with pytest.raises(ValueError):
        validation_check_steps(total_steps=3, epochs=1, checks_per_epoch=4)
