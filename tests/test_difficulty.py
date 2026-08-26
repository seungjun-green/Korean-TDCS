import pytest

from korean_math_tdcs.data.difficulty import operator_count, tdcs_level


@pytest.mark.parametrize("count,expected", [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (12, 5)])
def test_primary_difficulty_mapping(count, expected):
    assert tdcs_level(count) == expected


def test_operator_count_uses_gold_trace():
    assert operator_count({"gold_trace": ["+", "*", "-"]}) == 3


def test_empty_trace_is_rejected():
    with pytest.raises(ValueError):
        operator_count({"gold_trace": []})

