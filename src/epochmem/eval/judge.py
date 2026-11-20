"""Local LLM judge.

Call 1 (correctness) follows the LongMemEval per-type grading rubric.
Call 2 (behavior) classifies whether the response abstained and, for knowledge-update
questions, whether it returned the superseded pre-update value.
Both calls use JSON-schema constrained decoding with thinking disabled.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..data.stream import PreparedInstance
from ..llm import OllamaClient

CORRECT_SCHEMA = {"type": "object", "properties": {"correct": {"type": "boolean"}}, "required": ["correct"]}
BEHAVIOR_SCHEMA = {
    "type": "object",
    "properties": {"abstained": {"type": "boolean"}, "uses_superseded": {"type": "boolean"}},
    "required": ["abstained", "uses_superseded"],
}

_BASE = ("I will give you a question, a correct answer, and a response from a model. Answer yes if the "
         "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
         "correct answer or contains all the intermediate steps to get the correct answer, you should also "
         "answer yes. If the response only contains a subset of the information required by the answer, "
         "answer no.")
_TEMPORAL = (" In addition, do not penalize off-by-one errors for the number of days. If the question asks for "
             "the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 "
             "days when the answer is 18), the model's response is still correct.")
_KU = (" If the response contains some previous information along with an updated answer, the response should "
       "be considered as correct as long as the updated answer is the required answer.")
_ABS = ("I will give you an unanswerable question, an explanation, and a response from a model. Answer yes if "
        "the model correctly identifies the question as unanswerable. The model could say that the information "
        "is incomplete, or some other information is given but the asked information is not.")

_PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. "
         "Answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need "
         "to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the "
         "user's personal information correctly.")


_STATIC = ("The user's memory contains an earlier TRUE statement and a later FALSE statement that contradicts it. "
           "I will give you a question, the reference answer (which states the correct value), and a response from a "
           "model. Answer yes if the response's answer matches the correct value in the reference; it does not need "
           "to mention that sources disagree. If the response answers with the other (false) value, answer no.")


def memconflict_trap(inst) -> str | None:
    """The value a MemConflict reader should NOT answer with: dynamic -> the superseded Before value;
    static -> the newer FALSE statement (Point_B). Conditional conflicts have no single trap value."""
    if not str(inst.qtype).startswith("memconflict-"):
        return None
    rec = (inst.meta or {}).get("conflict") or {}
    if inst.qtype == "memconflict-dynamic" and isinstance(rec.get("update"), dict):
        return f"Earlier (superseded) value: {json.dumps(rec['update'].get('Before'), ensure_ascii=False)}"
    if inst.qtype == "memconflict-static" and isinstance(rec.get("point"), dict):
        return f"Later FALSE statement (should not be believed): {rec['point'].get('Value')}"
    return None


def correctness_prompt(prep: PreparedInstance, response: str) -> str:
    inst = prep.inst
    if inst.qtype == "memconflict-static":
        return (f"{_STATIC}\n\nQuestion: {inst.question}\n\nReference: {inst.answer}\n\nModel Response: {response}\n\n"
                'Is the model response correct? Reply as JSON {"correct": true|false}.')
    if inst.qtype == "memconflict-dynamic":
        return (f"{_BASE}{_KU}\n\nQuestion: {inst.question}\n\nCorrect Answer: {inst.answer}\n\nModel Response: {response}\n\n"
                'Is the model response correct? Reply as JSON {"correct": true|false}.')
    if inst.is_abstention:
        return (f"{_ABS}\n\nQuestion: {inst.question}\n\nExplanation: {inst.answer}\n\nModel Response: {response}\n\n"
                'Does the model correctly identify the question as unanswerable? Reply as JSON {"correct": true|false}.')
    if inst.qtype == "single-session-preference":
        # LongMemEval's preference rubric: gold answers are rubric descriptions, not answer strings
        return (f"{_PREF}\n\nQuestion: {inst.question}\n\nRubric: {inst.answer}\n\nModel Response: {response}\n\n"
                'Is the model response correct? Reply as JSON {"correct": true|false}.')
    rubric = _BASE + (_TEMPORAL if inst.qtype == "temporal-reasoning" else "") + (
        _KU if inst.qtype == "knowledge-update" else "")
    return (f"{rubric}\n\nQuestion: {inst.question}\n\nCorrect Answer: {inst.answer}\n\nModel Response: {response}\n\n"
            'Is the model response correct? Reply as JSON {"correct": true|false}.')


def behavior_prompt(prep: PreparedInstance, response: str) -> str:
    inst = prep.inst
    old = prep.superseded_evidence() if inst.qtype == "knowledge-update" and not inst.is_abstention else []
    trap = memconflict_trap(inst)
    if trap:
        old = [trap]
    old_block = "\n".join(old) if old else "N/A"
    return (
        "Classify a model's response to a question about the user's past conversations.\n\n"
        f"Question: {inst.question}\n"
        f"Current correct answer: {inst.answer}\n"
        f"Outdated (superseded) information from earlier conversations:\n{old_block}\n\n"
        f"Model Response: {response}\n\n"
        "abstained: true if the response declines to answer, says it does not know, or says the information "
        "is not available; false if it commits to an answer.\n"
        "uses_superseded: true only if outdated information is given above AND the response's final answer "
        "matches the outdated value instead of the current correct answer; otherwise false.\n"
        'Reply as JSON {"abstained": true|false, "uses_superseded": true|false}.'
    )


@dataclass
class Verdict:
    correct: bool
    abstained: bool
    uses_superseded: bool
    judge_parse_errors: int


class Judge:
    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def judge(self, prep: PreparedInstance, response: str) -> Verdict:
        errs = 0
        r1, _ = self.client.chat_json(self.model, [{"role": "user", "content": correctness_prompt(prep, response)}],
                                      schema=CORRECT_SCHEMA, max_tokens=32)
        correct = bool(isinstance(r1.parsed, dict) and r1.parsed.get("correct") is True)
        errs += int(r1.parse_error is not None)
        r2, _ = self.client.chat_json(self.model, [{"role": "user", "content": behavior_prompt(prep, response)}],
                                      schema=BEHAVIOR_SCHEMA, max_tokens=48)
        p2 = r2.parsed if isinstance(r2.parsed, dict) else {}
        errs += int(r2.parse_error is not None)
        return Verdict(correct=correct, abstained=bool(p2.get("abstained") is True),
                       uses_superseded=bool(p2.get("uses_superseded") is True), judge_parse_errors=errs)
