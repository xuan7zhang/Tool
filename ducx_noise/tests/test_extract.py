"""Unit tests for the temp-0 answer extractor (Task 1).

Covers: the "XXX is a XXX" article trap (old rule wrong, new extractor right),
closed-set normalisation + comparison, abstain handling, the reject-then-choose
case, the LLM backend via a fake client, and on-disk caching. CPU-only; no GPU or
live model server needed.
"""

import json
import re

import pytest

from ducx_noise import extract as ex


# Replicate the SHIPPED extract_choice so tests are self-contained and prove the
# old rule fails on the same inputs the new extractor handles.
def old_extract_choice(text):
    if text is None:
        return None
    text = str(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in reversed(lines):
        m = re.fullmatch(r"([A-F])", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    matches = re.findall(r"\b([A-F])\b", text, re.IGNORECASE)
    return matches[-1].upper() if matches else None


# Real ChestAgentBench-style traps: model concludes one letter in a bold marker,
# but later prose contains a stray standalone article "a"/letter that the old
# last-match fallback latches onto.
TRAP_CASES = [
    # (model_answer, ground_truth, expected_letter)
    (
        "Based on the findings, the correct statement is:\n\n"
        "**C) Septic pulmonary embolism due to tricuspid valve endocarditis**\n\n"
        "This is characterized by the presence of a cavitary lesion with "
        "peripheral nodules in a patient with a history of IV drug use.",
        "C",
        "C",
    ),
    (
        "The most accurate interpretation is:\n\n"
        "**B) The well-circumscribed opacification is a localized fibrous tumor.**\n\n"
        "Other options describe a mass effect that is not present.",
        "B",
        "B",
    ),
    (
        "Answer: D) Figure 3 demonstrates honeycombing.\n"
        "This reflects a progressive fibrotic process.",
        "D",
        "D",
    ),
]


@pytest.mark.parametrize("answer,gt,want", TRAP_CASES)
def test_article_trap_old_wrong_new_right(answer, gt, want):
    # Old rule is fooled by the trailing article "a" (or other stray letter).
    old = old_extract_choice(answer)
    # New deterministic extractor recovers the model's real conclusion.
    new, method = ex.RuleExtractor().extract(answer)
    assert new == want, f"new extractor should read {want}, got {new} ({method})"
    assert ex.compare_closed_set(new, gt) is True
    # At least demonstrate the old rule diverges from truth on these traps.
    assert old != want or old is None, (
        f"trap case expected to fool old rule but old returned {old}"
    )


def test_old_rule_latches_onto_article():
    # Direct demonstration of the failure mode the fix targets.
    answer = "**C) Diagnosis is X**\nThis is a hallmark finding."
    assert old_extract_choice(answer) == "A"  # the article "a"
    assert ex.RuleExtractor().extract(answer)[0] == "C"


def test_normalize_choice():
    assert ex.normalize_choice("C") == "C"
    assert ex.normalize_choice(" c ") == "C"
    assert ex.normalize_choice("(B)") == "B"
    assert ex.normalize_choice("Answer: D") == "D"
    assert ex.normalize_choice("NONE") is None
    assert ex.normalize_choice("") is None
    assert ex.normalize_choice(None) is None


def test_compare_closed_set():
    assert ex.compare_closed_set("C", "C") is True
    assert ex.compare_closed_set("a", "B") is False
    assert ex.compare_closed_set(None, "B") is False  # abstain scored wrong
    assert ex.compare_closed_set("C", None) is None  # no GT -> unscorable


def test_reject_then_choose_rule_limitation_and_llm_fix():
    # Model echoes C, rejects it, picks D. Rule backend picks the first marker (C)
    # -- documented limitation -- while the LLM backend reads the final choice.
    answer = (
        "The correct option is:\n\n**C) ...**\n\n"
        "However this is not accurate. The correct answer is:\n\n**D) ...**"
    )
    assert ex.RuleExtractor().extract(answer)[0] == "C"  # known rule limitation

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    # A real reader returns the final choice.
                    msg = type("M", (), {"content": "D"})
                    choice = type("C", (), {"message": msg})
                    return type("R", (), {"choices": [choice]})

    llm = ex.LLMExtractor.__new__(ex.LLMExtractor)
    llm.model = "fake"
    llm.max_tokens = 4
    llm.client = FakeClient()
    llm.version = "extract-v1-llm:fake"
    assert llm.extract(answer)[0] == "D"


def test_cache_roundtrip(tmp_path):
    entry = {
        "question_id": "q1",
        "model_answer": "**C) foo**\nthis is a finding",
        "correct_answer": "C",
        "predicted_answer": "A",   # what the old rule recorded
        "is_correct": False,        # old: wrong
    }
    cache_dir = str(tmp_path / "extraction_cache")
    rec = ex.extract_entry(entry, ex.RuleExtractor(), cache_dir)
    assert rec["extracted_answer"] == "C"
    assert rec["is_correct_new"] is True
    assert rec["old_is_correct"] is False  # kept side by side
    # Second call hits the cache (file exists, same version).
    rid = ex.rollout_id(entry)
    assert (tmp_path / "extraction_cache" / f"{rid}.json").exists()
    rec2 = ex.extract_entry(entry, ex.RuleExtractor(), cache_dir)
    assert rec2 == rec


def test_recompute_run(tmp_path):
    log = tmp_path / "run.json"
    rows = [
        {"question_id": "q1", "model_answer": "**C) x**\nis a thing",
         "correct_answer": "C", "predicted_answer": "A", "is_correct": False},
        {"question_id": "q2", "model_answer": "**B) y**",
         "correct_answer": "B", "predicted_answer": "B", "is_correct": True},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    row, records = ex.recompute_run(
        str(log), ex.RuleExtractor(), str(tmp_path / "cache"), label="base", seed=0
    )
    assert row["n_scored"] == 2
    assert row["task_accuracy_old"] == 0.5
    assert row["task_accuracy_new"] == 1.0
    assert row["n_flip_old_wrong_new_right"] == 1
    assert row["n_flip_old_right_new_wrong"] == 0
