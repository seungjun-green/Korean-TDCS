from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    uid: str
    prompt: str
    answer: Any
    parser: str
    subset: str


MATH_STEM_SUBJECTS = [
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Ecology",
    "Electrical-Engineering",
    "Information-Technology",
    "Materials-Engineering",
    "Math",
    "Mechanical-Engineering",
]


def _load_hrm8k(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import load_dataset

    subsets = spec.get("subsets", ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"])
    examples = []
    for subset in subsets:
        dataset = load_dataset(
            spec.get("dataset", "HAERAE-HUB/HRM8K"),
            subset,
            split=spec.get("split", "test"),
        )
        for index, row in enumerate(dataset):
            question = row.get("question") or row.get("problem")
            answer = row["answer"] if "answer" in row else row.get("label")
            if subset == "MMMLU":
                prompt = (
                    f"{question}\n\n문제를 푼 뒤 마지막 줄에 정답 선택지 번호(1, 2, 3, 4)만 쓰세요."
                )
                parser = "choice"
            else:
                prompt = (
                    f"{question}\n\n풀이 후 마지막 줄에 최종 답을 \\boxed{{...}} 형식으로 쓰세요."
                )
                parser = "math"
            examples.append(
                EvalExample(f"hrm8k/{subset}/{index}", prompt, answer, parser, subset)
            )
    return examples


def _load_ko_gsm8k(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import load_dataset

    dataset = load_dataset(
        spec.get("dataset", "thunder-research-group/SNU_Ko-GSM8K"),
        split=spec.get("split", "test"),
    )
    return [
        EvalExample(
            f"ko_gsm8k/{index}",
            str(row["question"]),
            row["answer"],
            "numeric",
            "test",
        )
        for index, row in enumerate(dataset)
    ]


def _choice_prompt(row: dict[str, Any]) -> str:
    choices = "\n".join(f"{letter}. {row[letter]}" for letter in "ABCD")
    return (
        f"{row['question']}\n\n{choices}\n\n"
        "정답인 선택지 하나를 고르고, 마지막 줄에 선택지 문자(A, B, C, D)만 쓰세요."
    )


def _load_kmmlu(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import get_dataset_config_names, load_dataset

    dataset_name = spec.get("dataset", "HAERAE-HUB/KMMLU")
    available = set(get_dataset_config_names(dataset_name))
    requested = spec.get("subjects", MATH_STEM_SUBJECTS)
    subjects = [subject for subject in requested if subject in available]
    missing = sorted(set(requested) - available)
    if missing and spec.get("strict_subjects", False):
        raise ValueError(f"KMMLU subjects do not exist: {missing}")
    if not subjects:
        raise ValueError("No configured KMMLU Math/STEM subjects are available")
    examples = []
    for subject in subjects:
        dataset = load_dataset(dataset_name, subject, split=spec.get("split", "test"))
        for index, row in enumerate(dataset):
            examples.append(
                EvalExample(
                    f"kmmlu/{subject}/{index}",
                    _choice_prompt(row),
                    row["answer"],
                    "choice",
                    subject,
                )
            )
    return examples


def _load_validation(spec: dict[str, Any]) -> list[EvalExample]:
    from datasets import load_dataset

    dataset = load_dataset(
        spec.get("dataset", "keunhyeung/dmath-ko-reasoning-dpo"),
        spec.get("config", "reasoning-sft"),
        split=spec.get("split", "validation"),
    )
    return [
        EvalExample(
            f"validation/{index}",
            str(row["instruction"]),
            row["gold_answer"],
            "math",
            "reasoning-sft-validation",
        )
        for index, row in enumerate(dataset)
    ]


LOADERS = {
    "hrm8k": _load_hrm8k,
    "ko_gsm8k": _load_ko_gsm8k,
    "kmmlu_math_stem": _load_kmmlu,
    "validation_math": _load_validation,
}


def load_benchmark(name: str, spec: dict[str, Any]) -> list[EvalExample]:
    try:
        examples = LOADERS[name](spec)
    except KeyError as error:
        raise ValueError(f"Unknown benchmark: {name}") from error
    max_samples = spec.get("max_samples")
    return examples[: int(max_samples)] if max_samples is not None else examples


def reasoning_final(text: str) -> str:
    return text.rsplit("</think>", 1)[-1].strip()


def _last_numeric(text: str) -> str | None:
    latex_fractions = re.findall(
        r"\\(?:d?frac)\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}"
        r"\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
        text,
    )
    if latex_fractions:
        numerator, denominator = latex_fractions[-1]
        return f"{numerator.replace(',', '')}/{denominator.replace(',', '')}"
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    hashes = re.findall(r"####\s*([^\n]+)", text)
    candidates = boxed or hashes
    if candidates:
        text = candidates[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[+-]?\d[\d,]*)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def normalize_numeric(value: Any) -> str | None:
    candidate = _last_numeric(str(value))
    if candidate is None:
        return None
    try:
        if "/" in candidate:
            numerator, denominator = candidate.split("/", 1)
            number = Decimal(numerator) / Decimal(denominator)
        else:
            number = Decimal(candidate)
        return format(number.normalize(), "f")
    except (InvalidOperation, ZeroDivisionError):
        return candidate


def normalize_choice(value: Any) -> str | None:
    stripped = str(value).strip()
    if stripped.isdigit():
        number = int(stripped)
        if 1 <= number <= 4:
            return "ABCD"[number - 1]
        if 0 <= number <= 3:
            return "ABCD"[number]
    matches = re.findall(r"(?<![A-Za-z])[ABCD](?![A-Za-z])", str(value).upper())
    if matches:
        return matches[-1]
    number_matches = re.findall(r"(?<!\d)([1-4])(?!\d)", str(value))
    return "ABCD"[int(number_matches[-1]) - 1] if number_matches else None


def _boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    depth = 1
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index]
    return None


def normalize_math(value: Any) -> str | None:
    text = reasoning_final(str(value)).strip()
    if not text:
        return None
    candidate = _boxed_content(text)
    if candidate is None:
        hashes = re.findall(r"####\s*([^\n]+)", text)
        if hashes:
            candidate = hashes[-1]
    if candidate is None:
        answer_markers = re.findall(r"(?:최종\s*답|정답|답)\s*[:：은는]?\s*([^\n]+)", text)
        candidate = answer_markers[-1] if answer_markers else text.splitlines()[-1]
    candidate = candidate.strip().rstrip(".。")
    numeric = normalize_numeric(candidate)
    pure_numeric = re.fullmatch(
        r"\s*(?:\\(?:d?frac)\s*\{[-+]?\d[\d,]*(?:\.\d+)?\}"
        r"\s*\{[-+]?\d[\d,]*(?:\.\d+)?\}|"
        r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*)?)\s*",
        candidate,
    )
    if pure_numeric and numeric is not None:
        return numeric
    canonical = candidate
    for removable in ("\\(", "\\)", "\\[", "\\]", "$", "\\left", "\\right"):
        canonical = canonical.replace(removable, "")
    canonical = canonical.replace("\\dfrac", "\\frac")
    canonical = re.sub(r"\s+", "", canonical)
    if canonical.startswith("{") and canonical.endswith("}"):
        depth = 0
        closes_at_end = False
        for index, character in enumerate(canonical):
            depth += character == "{"
            depth -= character == "}"
            if depth == 0:
                closes_at_end = index == len(canonical) - 1
                break
        if closes_at_end:
            canonical = canonical[1:-1]
    return canonical or None


def parse_answer(text: str, parser: str) -> str | None:
    final = reasoning_final(text)
    if parser == "numeric":
        return normalize_numeric(final)
    if parser == "choice":
        return normalize_choice(final)
    if parser == "math":
        return normalize_math(final)
    raise ValueError(f"Unknown parser: {parser}")


def gold_answer(value: Any, parser: str) -> str | None:
    if parser == "numeric":
        return normalize_numeric(value)
    if parser == "choice":
        return normalize_choice(value)
    if parser == "math":
        return normalize_math(value)
    raise ValueError(f"Unknown parser: {parser}")
