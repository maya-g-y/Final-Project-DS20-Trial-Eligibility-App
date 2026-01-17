
from __future__ import annotations
from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field

from .llm import json_from_llm

Score5 = Literal[1,2,3,4,5]

class JudgeOutput(BaseModel):
    consistency_with_criteria: Score5 = Field(description="Is the final decision consistent with the criteria + patient data?")
    groundedness: Score5 = Field(description="Are reasons supported by provided evidence (patient fields/snippets) without hallucination?")
    uncertainty_appropriateness: Score5 = Field(description="If uncertain: is that justified? If not uncertain: was that safe?")
    hallucination_risk: Score5 = Field(description="1=low risk, 5=high risk of hallucination/made-up facts")
    notes: List[str] = Field(default_factory=list)

JUDGE_SYSTEM_PROMPT = """
You are an evaluator (LLM-as-a-judge) for a clinical trial prescreening system.
You must score the model output using ONLY the provided inputs.
Be strict about hallucinations: referencing missing fields (HbA1c, pregnancy, CGM history, smartphone, etc.) as if known should increase hallucination_risk.
Return JSON matching the schema exactly.
"""

def judge_match(
    patient: Dict[str, Any],
    criteria: Dict[str, Any],
    match_result: Dict[str, Any],
    evidence_snippets: List[str],
    model: str = "gemini-2.0-flash-exp"
) -> JudgeOutput:

    user_prompt = f"""
STUDY CRITERIA JSON:
{criteria}

PATIENT FIELDS:
{patient}

EVIDENCE SNIPPETS:
{evidence_snippets}

MODEL OUTPUT (match_result):
{match_result}

Task:
Provide 1-5 scores and short notes.
"""
    return json_from_llm(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=JudgeOutput,
        model=model,
        temperature=0.0
    )
