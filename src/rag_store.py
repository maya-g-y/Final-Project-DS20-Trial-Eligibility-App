
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_patient_rag_text(patient: Dict[str, Any]) -> str:
    # Prefer prebuilt patient_card if exists (best for evidence + consistency)
    if patient.get("patient_card"):
        return str(patient["patient_card"])
    # fallback (if patient_card isn't provided)
    return (
        f"Patient ID: {patient.get('patient_id')}\n"
        f"Age: {patient.get('age')}\n"
        f"Gender: {patient.get('gender')}\n"
        f"eGFR: {patient.get('egfr')}\n"
        f"Insulin user: {patient.get('insulin_user')}\n"
        f"Current medications: {patient.get('current_medications')}\n"
        f"Comorbidities: {patient.get('comorbidities')}\n"
    )

class PatientRAGStore:
    def __init__(self, persist_dir: str = "trial-screening-agent/data/chroma", collection_name: str = "patients"):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._embedder = SentenceTransformer(DEFAULT_EMBED_MODEL)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(name=collection_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts, normalize_embeddings=True).tolist()

    def upsert_patients(self, patients: List[Dict[str, Any]]) -> None:
        ids = []
        docs = []
        metas = []
        for p in patients:
            pid = str(p["patient_id"])
            ids.append(pid)
            docs.append(build_patient_rag_text(p))
            metas.append({"patient_id": pid})

        embs = self._embed(docs)
        self._col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    def retrieve(self, query: str, top_k: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qemb = self._embed([query])[0]
        res = self._col.query(
            query_embeddings=[qemb],
            n_results=top_k,
            where=where
        )
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return out
