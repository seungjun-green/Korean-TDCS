from korean_math_tdcs.data.loading import _normalize_olympiad_sft, _solution_final_answer


def test_olympiad_solution_uses_last_nested_boxed_answer():
    solution = (
        r"먼저 중간 결과는 \boxed{3}이다. "
        r"따라서 최종 답은 \boxed{\frac{1}{2(n+1)}}이다."
    )

    assert _solution_final_answer(solution) == r"\frac{1}{2(n+1)}"


def test_olympiad_row_is_normalized_for_exaone_reasoning_sft():
    row = _normalize_olympiad_sft(
        {
            "problem_ko": "문제",
            "solution_ko": r"풀이 과정. 최종적으로 \boxed{42}",
            "difficulty_level": 3,
        }
    )

    assert row["instruction"] == "문제"
    assert row["reasoning"] == r"풀이 과정. 최종적으로 \boxed{42}"
    assert row["response"] == r"\boxed{42}"
    assert row["gold_answer"] == "42"
    assert row["tdcs_level"] == 3
