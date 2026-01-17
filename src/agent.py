
from __future__ import annotations
from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field

from .tools import rule_screen
from .rag_store import PatientRAGStore
from .llm import json_from_llm

Decision = Literal["eligible", "not_eligible", "uncertain"]
UncertaintyReason = Literal["missing patient data", "missing essential trial info", "conflict with verifier"]

class MatchResult(BaseModel):
    patient_id: str
    study_id: str

    rule_decision: Decision
    final_decision: Decision

    rule_reasons: List[str] = Field(default_factory=list)

    verifier_agree: bool
    verifier_suggested_decision: Decision
    verifier_notes: List[str] = Field(default_factory=list)

    evidence: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)

    conflict: bool = False
    uncertainty_reason: Optional[UncertaintyReason] = None

class VerificationResult(BaseModel):
    agree: bool
    suggested_decision: Decision
    notes: List[str] = Field(default_factory=list)
    groundedness: Literal["high","medium","low"] = "medium"
    used_only_available_fields: bool = True

VERIFIER_SYSTEM_PROMPT = """
You are a strict verifier for clinical trial prescreening decisions.
You must:
- Check whether the RULE-BASED decision is consistent with the provided criteria and patient data.
- Only use the provided patient fields and retrieved evidence.
- Do NOT guess missing medical facts. If key info is missing to decide safely, you should suggest 'uncertain'.
Return JSON matching the schema exactly.
"""

def build_query_from_criteria(criteria: Dict[str, Any]) -> str:
    keywords: List[str] = []
    for block in ["inclusion", "exclusion"]:
        for c in criteria.get(block, []):
            if c["op"] in ["contains_any", "contains_all", "not_contains_any"] and isinstance(c["value"], list):
                keywords.extend([str(x) for x in c["value"]])
            else:
                keywords.append(f"{c['field']} {c['op']} {c['value']}")
    return " ; ".join(keywords[:25])

def verify_rule_decision(
    patient: Dict[str, Any],
    criteria: Dict[str, Any],
    rule_decision: Decision,
    rule_reasons: List[str],
    evidence_texts: List[str],
    model: str
) -> VerificationResult:

    user_prompt = f"""
STUDY CRITERIA JSON:
{criteria}

PATIENT FIELDS:
{patient}

RETRIEVED EVIDENCE SNIPPETS:
{evidence_texts}

RULE-BASED OUTPUT:
decision = {rule_decision}
reasons = {rule_reasons}

Task:
1) Decide if you AGREE with the rule-based decision.
2) If you DISAGREE, provide suggested_decision.
3) If information is missing to decide safely, suggested_decision must be "uncertain".
4) Add short notes explaining the check.
"""
    return json_from_llm(
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=VerificationResult,
        model=model,
        temperature=0.0
    )

def infer_uncertainty_reason(
    final_decision: Decision,
    conflict: bool,
    missing_info: List[str]
) -> Optional[UncertaintyReason]:
    if final_decision != "uncertain":
        return None
    if conflict:
        return "conflict with verifier"
    if missing_info and len(missing_info) > 0:
        return "missing patient data"
    return "missing essential trial info"

def run_patient_study_match(
    patient: Dict[str, Any],
    criteria: Dict[str, Any],
    store: PatientRAGStore,
    top_k: int = 3,
    model: str = "gemini-2.0-flash-exp"
) -> MatchResult:
    pid = str(patient["patient_id"])
    study_id = str(criteria.get("study_id", "UNKNOWN"))

    # 1) Deterministic rule screen
    rule_decision, rule_reasons, missing = rule_screen(patient, criteria)

    # 2) Retrieve some evidence for verifier + UI explanations
    query = build_query_from_criteria(criteria)
    retrieved = store.retrieve(query=query, top_k=top_k, where={"patient_id": pid})
    evidence_texts = [r["document"] for r in retrieved]

    # 3) LLM verifies the rule decision
    ver = verify_rule_decision(
        patient=patient,
        criteria=criteria,
        rule_decision=rule_decision,
        rule_reasons=rule_reasons,
        evidence_texts=evidence_texts,
        model=model
    )

    # 4) Safe policy:
    # - if verifier agrees -> keep rule decision
    # - if verifier disagrees -> mark conflict and fall back to UNCERTAIN
    if ver.agree:
        final_decision: Decision = rule_decision
        conflict = False
    else:
        final_decision = "uncertain"
        conflict = True

    uncertainty_reason = infer_uncertainty_reason(final_decision, conflict, missing)

    return MatchResult(
        patient_id=pid,
        study_id=study_id,
        rule_decision=rule_decision,
        final_decision=final_decision,
        rule_reasons=rule_reasons,
        verifier_agree=ver.agree,
        verifier_suggested_decision=ver.suggested_decision,
        verifier_notes=ver.notes,
        evidence=evidence_texts,
        missing_info=missing,
        conflict=conflict,
        uncertainty_reason=uncertainty_reason
    )
