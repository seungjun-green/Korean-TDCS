from korean_math_tdcs.data.formatting import format_eval_prompt, format_sft_text
from korean_math_tdcs.evaluation.benchmarks import (
    normalize_choice,
    normalize_math,
    normalize_numeric,
    parse_answer,
)


class TemplateTokenizer:
    def apply_chat_template(self, *_args, **kwargs):
        assert kwargs["tokenize"] is False
        return "rendered prompt"

    def __call__(self, text, *, add_special_tokens):
        assert text == "rendered prompt"
        assert add_special_tokens is False
        return {"input_ids": [11, 12, 13]}


class SFTTemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False
        assistant = messages[-1]
        assert assistant["reasoning_content"] == "풀이 과정"
        assert assistant["content"] == "정답"
        assert "<think>" not in assistant["content"]
        return (
            f"[|user|]\n{messages[0]['content']}\n"
            f"[|assistant|]\n<think>\n{assistant['reasoning_content']}\n</think>\n\n"
            f"{assistant['content']}"
        )


def test_eval_prompt_is_always_a_plain_token_id_list():
    assert format_eval_prompt(TemplateTokenizer(), "question") == [11, 12, 13]


def test_sft_prompt_uses_structured_reasoning_content():
    rendered = format_sft_text(
        SFTTemplateTokenizer(),
        {"instruction": "문제", "reasoning": "풀이 과정", "response": "정답"},
    )

    assert "<think>\n풀이 과정\n</think>" in rendered


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
    assert normalize_math("") is None
