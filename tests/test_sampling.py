
from korean_math_tdcs.data.sampling import (
    random_epoch_indices,
    resolved_training_budget,
    stage_boundaries,
    stage_for_step,
)


def test_resolved_primary_budget():
    assert resolved_training_budget(2708, 4, 32) == (10832, 339)


def test_primary_stage_boundaries():
    assert stage_boundaries(339) == [(1, 68), (69, 136), (137, 203), (204, 271), (272, 339)]
    boundary_steps = (1, 68, 69, 136, 137, 203, 204, 271, 272, 339)
    assert [stage_for_step(step, 339) for step in boundary_steps] == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
        5,
        5,
    ]


def test_olympiad_stage_boundaries_are_five_equal_75_step_stages():
    assert stage_boundaries(375) == [
        (1, 75),
        (76, 150),
        (151, 225),
        (226, 300),
        (301, 375),
    ]


def test_random_epochs_visit_every_row_once_per_epoch():
    indices = random_epoch_indices(7, 4, 42)
    assert len(indices) == 28
    for epoch in range(4):
        assert sorted(indices[epoch * 7 : (epoch + 1) * 7]) == list(range(7))
