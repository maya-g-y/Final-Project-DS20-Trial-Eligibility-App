
from __future__ import annotations
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

from .llm import json_from_llm

ALLOWED_FIELDS = [
    "age",
    "gender",
    "egfr",
    "insulin_user",
    "current_medications",
    "comorbidities",
]

Op = Literal[
    ">=", "<=", ">", "<", "==",
    "contains_any", "contains_all",
    "not_contains_any"
]

class Criterion(BaseModel):
    field: Literal["age","gender","egfr","insulin_user","current_medications","comorbidities"]
    op: Op
    value: Union[str, float, int, bool, List[str], List[float], List[int]]
    note: Optional[str] = Field(default=None, description="Short explanation of mapping/assumption")

class StudyCriteria(BaseModel):
    study_id: str
    title: str
    description: Optional[str] = None
    inclusion: List[Criterion] = Field(default_factory=list)
    exclusion: List[Criterion] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=lambda: ["comorbidities","current_medications"])

SYSTEM_PROMPT = f"""
You convert clinical trial inclusion/exclusion criteria into STRICT JSON.
You MUST only use fields from this allowed list:
{ALLOWED_FIELDS}

Rules:
- If a criterion is not available in the dataset (e.g., pregnancy, HbA1c, smartphone, CGM history), do NOT invent a field.
  Instead: omit it from criteria and mention it as missing in description or in a note.
- Conservative mapping examples:
  - "Type 2 Diabetes" -> comorbidities contains_any ["type 2 diabetes", "t2d", "diabetes mellitus type 2"]
  - "Type 1 Diabetes" -> exclusion: comorbidities contains_any ["type 1 diabetes", "t1d"]
  - "Non-insulin treated" -> inclusion: insulin_user == false
  - "Age between 40 and 70" -> inclusion: age >= 40 AND age <= 70
  - "High risk cardiovascular" -> proxy: comorbidities contains_any
    ["coronary artery disease","cad","myocardial infarction","mi","stroke","heart failure","hf","hypertension"]
    (Add a note that it's a proxy.)
Output must match the provided schema exactly.
"""

def parse_study_text_to_criteria(study_text: str) -> StudyCriteria:
    user_prompt = f"Convert this trial into StudyCriteria JSON.\n\nTRIAL TEXT:\n{study_text}"
    return json_from_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=StudyCriteria,
        temperature=0.0
    )
