from __future__ import annotations

import math
import re


def _split_top_level(expression: str) -> list[str]:
    """Split comma-separated answers without splitting tuples or LaTeX groups."""
    opening = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    parts: list[str] = []
    start = 0
    for index, character in enumerate(expression):
        if character in opening:
            stack.append(opening[character])
        elif stack and character == stack[-1]:
            stack.pop()
        elif character == "," and not stack:
            parts.append(expression[start:index].strip())
            start = index + 1
    parts.append(expression[start:].strip())
    return [part for part in parts if part]


def _expand_plus_minus(expressions: list[str]) -> list[str]:
    result = []
    for expression in expressions:
        if "\\pm" in expression:
            result.extend([expression.replace("\\pm", "+"), expression.replace("\\pm", "-")])
        else:
            result.append(expression)
    return result


def _clean(expression: str) -> str:
    replacements = {
        "\\left": "",
        "\\right": "",
        "∶": ":",
        "，": ",",
        "$": "",
        "\\approx": "=",
        "\\simeq": "=",
        "\\sim": "=",
        "^\\prime": "'",
        "^{\\prime}": "'",
        "^\\circ": "",
        "%": "",
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    expression = re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", expression)
    return expression.strip("\n $,.:;^_=+`!@#$%^&*~，。")


def _numeric_equal(reference: str, prediction: str, precision: float) -> bool:
    reference_number = float(reference)
    prediction_number = float(prediction)
    return any(
        math.isclose(candidate, prediction_number, rel_tol=0.0, abs_tol=precision * 1.01)
        for candidate in (reference_number / 100, reference_number, reference_number * 100)
    )


def _latex(expression: str):
    from sympy.parsing.latex import parse_latex

    return parse_latex(expression)


def _expression_equal(reference: str, prediction: str, precision: float) -> bool:
    from sympy import Symbol, simplify

    def right_hand_side(expression: str) -> str:
        return expression.split("=", 1)[1].strip() if "=" in expression else expression

    reference_value = _latex(right_hand_side(reference))
    prediction_value = _latex(right_hand_side(prediction))
    if reference_value == prediction_value:
        return True
    reference_has_symbols = bool(reference_value.atoms(Symbol))
    prediction_has_symbols = bool(prediction_value.atoms(Symbol))
    if reference_has_symbols != prediction_has_symbols:
        return False
    difference = simplify(reference_value - prediction_value)
    if not reference_has_symbols:
        return abs(complex(difference.evalf())) <= precision * 1.01
    return simplify(difference) == 0


def _equation_equal(reference: str, prediction: str) -> bool:
    from sympy import simplify

    def zero_form(equation: str):
        left, right = equation.split("=", 1)
        return simplify(_latex(left) - _latex(right))

    reference_zero = zero_form(reference)
    prediction_zero = zero_form(prediction)
    if reference_zero == 0 or prediction_zero == 0:
        return reference_zero == prediction_zero
    ratio = simplify(reference_zero / prediction_zero)
    return bool(ratio.is_number and ratio != 0)


def _interval_equal(reference: str, prediction: str, precision: float) -> bool:
    if reference[0] != prediction[0] or reference[-1] != prediction[-1]:
        return False
    reference_parts = _split_top_level(reference[1:-1])
    prediction_parts = _split_top_level(prediction[1:-1])
    return len(reference_parts) == len(prediction_parts) and all(
        _item_equal(expected, actual, precision)
        for expected, actual in zip(reference_parts, prediction_parts, strict=True)
    )


def _item_equal(reference: str, prediction: str, precision: float) -> bool:
    if reference == prediction and reference:
        return True
    if (
        reference.startswith(("(", "["))
        and prediction.startswith(("(", "["))
        and reference.endswith((")", "]"))
        and prediction.endswith((")", "]"))
    ):
        try:
            return _interval_equal(reference, prediction, precision)
        except Exception:
            pass
    try:
        if _numeric_equal(reference, prediction, precision):
            return True
    except (TypeError, ValueError):
        pass
    try:
        if not ("=" in reference and "=" in prediction):
            return _expression_equal(reference, prediction, precision)
    except Exception:
        pass
    try:
        if "=" in reference and "=" in prediction:
            return _equation_equal(reference, prediction)
    except Exception:
        pass
    return False


def olympiad_math_equal(reference: str, prediction: str, precision: float = 1e-8) -> bool:
    """Compare OlympiadBench answers using its order-insensitive symbolic convention."""
    expected = _expand_plus_minus(_split_top_level(_clean(reference)))
    actual = _expand_plus_minus(_split_top_level(_clean(prediction)))
    if len(expected) != len(actual):
        return False
    unmatched = actual.copy()
    for expected_item in expected:
        for index, actual_item in enumerate(unmatched):
            if _item_equal(expected_item, actual_item, precision):
                unmatched.pop(index)
                break
        else:
            return False
    return True
