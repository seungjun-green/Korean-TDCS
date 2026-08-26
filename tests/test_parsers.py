from korean_math_tdcs.evaluation.benchmarks import (
    normalize_choice,
    normalize_math,
    normalize_numeric,
    parse_answer,
)


def test_numeric_parser_uses_final_reasoning_text():
    assert parse_answer("<think>2 + 3 = 5</think>\n최종 답은 \\boxed{5}", "numeric") == "5"
    assert normalize_numeric("#### 18,000") == "18000"
    assert normalize_numeric("답: 1/2") == "0.5"
    assert normalize_numeric(r"답: \frac{1}{6}") == "0.1666666666666666666666666667"


def test_choice_parser():
    assert normalize_choice(1) == "A"
    assert parse_answer("<think>A도 검토한다.</think>\n정답: C", "choice") == "C"
    assert parse_answer("<think>1번도 검토한다.</think>\n정답: 4", "choice") == "D"


def test_symbolic_math_parser_does_not_collapse_to_last_number():
    assert normalize_math(r"\boxed{3\sqrt{5}}") == r"3\sqrt{5}"
    assert normalize_math(r"\frac{1}{6}") == "0.1666666666666666666666666667"
