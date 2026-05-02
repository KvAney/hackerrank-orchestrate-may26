"""Deterministic support triage agent for HackerRank Orchestrate.

The implementation follows triage_system_guide.md while staying offline and
dependency-free. It builds one unified retrieval index over ./data, resolves the
best company context, classifies the ticket with rules, and writes predictions
to support_tickets/output.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
import numpy as np


SUPPORTED_COMPANIES = {"hackerrank": "HackerRank", "claude": "Claude", "visa": "Visa"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
LOW_CONFIDENCE_THRESHOLD = 0.55
COMPANY_OVERRIDE_MARGIN = 0.05
COMPANY_OVERRIDE_MIN_SCORE = 0.70
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "the",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    company: str
    product_area: str
    source_path: str
    title: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class CompanyResolution:
    requested_company: Optional[str]
    resolved_company: Optional[str]
    filtered_score: float
    global_score: float
    overridden: bool


@dataclass(frozen=True)
class TriageDecision:
    status: str
    product_area: str
    request_type: str
    low_confidence: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LLMAssessment:
    product_area: str
    request_type: str
    response: str
    confidence: float
    rationale: str
    needs_human: bool = False # ADD THIS

def normalize_text(value: object) -> str:
    """Normalize text but keep important phrases such as "my card" intact."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\b(thanks|thank you|regards|best regards|sincerely)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9']+", normalize_text(text)) if token not in STOPWORDS]


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def clean_markdown(markdown: str) -> str:
    markdown = re.sub(r"^---\s.*?---\s*", " ", markdown, flags=re.S)
    markdown = re.sub(r"!\[[^\]]*]\([^)]*\)", " ", markdown)
    markdown = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", markdown)
    markdown = re.sub(r"<[^>]+>", " ", markdown)
    markdown = re.sub(r"`{1,3}.*?`{1,3}", " ", markdown, flags=re.S)
    markdown = re.sub(r"^[#>*\-\s]+", "", markdown, flags=re.M)
    markdown = re.sub(r"\s+", " ", markdown)
    return html.unescape(markdown).strip()


def load_env_file(path: Path) -> None:
    """Load simple KEY=value pairs without overriding existing environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dense_cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def title_from_markdown(markdown: str, path: Path) -> str:
    for line in markdown.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if match:
            return clean_markdown(match.group(1))
    stem = re.sub(r"^\d+-", "", path.stem)
    return stem.replace("-", " ").replace("_", " ").strip().title()


def chunk_words(text: str, size: int = 380, overlap: int = 60) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if len(piece) >= 30 or start == 0:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def infer_company(path: Path, data_dir: Path) -> str:
    rel = path.relative_to(data_dir)
    root = rel.parts[0].lower() if rel.parts else ""
    return SUPPORTED_COMPANIES.get(root, root.title())


def infer_product_area(path: Path, data_dir: Path) -> str:
    rel = path.relative_to(data_dir)
    parts = [part.lower().replace("-", "_") for part in rel.parts]
    company = parts[0] if parts else ""

    if len(parts) <= 2 or path.name.lower() in {"index.md", "support.md"}:
        return "general_support"

    if company == "visa":
        joined = "/".join(parts)
        if "travel" in joined or "traveler" in joined:
            return "travel_support"
        if "fraud" in joined:
            return "fraud_protection"
        if "dispute" in joined or "charge" in joined:
            return "dispute_resolution"
        if "small_business" in joined or "merchant" in joined:
            return "small_business"
        return "general_support"

    if company == "hackerrank":
        area = parts[1]
        if area == "hackerrank_community":
            return "community"
        if area == "general_help":
            return "general_support"
        return area

    if company == "claude":
        area = parts[1]
        if area == "privacy_and_legal":
            return "privacy"
        if area == "identity_management_sso_jit_scim":
            return "identity_management"
        if len(parts) > 2 and parts[1] == "claude":
            nested = parts[2]
            if nested == "conversation_management":
                return "conversation_management"
            if nested == "account_management":
                return "account_management"
            if nested == "personalization_and_settings":
                return "privacy" if "privacy" in "/".join(parts) else "settings"
            return nested
        return area

    return "general_support"

from sentence_transformers import SentenceTransformer

class LocalEmbeddingClient:
    """Free, local alternative to OpenAI for generating vectors."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # This downloads the model (80MB) on first run and saves it locally
        try:
            self.model = SentenceTransformer(model_name)
            self.enabled = True
            self.embedding_model = model_name
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self.enabled = False

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Generates vectors locally using your CPU/GPU
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class OpenRouterClient:
    """OpenRouter chat-completions client for LLM-only assessment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = OPENROUTER_API_BASE,
        chat_model: str = "google/gemma-4-31b-it:free",
        timeout_seconds: int = 90,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        self.api_base = api_base.rstrip("/")
        self.chat_model = os.getenv("OPENROUTER_MODEL", chat_model)
        self.timeout_seconds = timeout_seconds

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        request = urllib.request.Request(
            url=f"{self.api_base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://www.hackerrank.com/",
                "X-Title": "HackerRank Orchestrate Support Agent",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter API error {exc.code}: {body}") from exc

    def assess_ticket(
        self,
        query: str,
        company: Optional[str],
        preliminary_status: str,
        tfidf_product_area: str,
        tfidf_request_type: str,
        results: list[RetrievalResult],
    ) -> Optional[LLMAssessment]:
        product_area_options = sorted({result.chunk.product_area for result in results if result.chunk.product_area})
        documents = [
            {
                "source": Path(result.chunk.source_path).as_posix(),
                "company": result.chunk.company,
                "product_area": result.chunk.product_area,
                "score": round(result.score, 4),
                "text": " ".join(result.chunk.text.split()[:220]),
            }
            for result in results[:3]
        ]
        user_payload = {
            "query": query,
            "resolved_company": company,
            "preliminary_status": preliminary_status,
            "tfidf_product_area": tfidf_product_area,
            "tfidf_request_type": tfidf_request_type,
            "allowed_product_areas": product_area_options,
            "allowed_request_types": sorted(ALLOWED_REQUEST_TYPES),
            "retrieved_documents": documents,
        }
        system_prompt = (
         
"You are a Support Triage Expert. Analyze the user query against the retrieved documents.\n\n"
    "DECISION LOGIC:\n"
    "1. If the docs explain HOW to do exactly what the user wants: set request_type='product_issue' and needs_human=false.\n"
    "2. If the user wants to do something the docs DO NOT mention as a feature: set request_type='feature_request' and needs_human=true.\n"
    "3. If the user says a feature IS NOT WORKING (error, down, fails): set request_type='bug' and needs_human=true.\n"
    "4. If the user lacks permission (non-admin asking for admin action): set request_type='product_issue' and needs_human=true (Escalate for Policy).\n\n"
    "Return JSON: {product_area, request_type, response, needs_human, rationale}"
    )
        payload = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
            "temperature": 0,
            "max_tokens": 700,
            "response_format": {"type": "json_object"},
        }
        response = self._post("/chat/completions", payload)
        message = response.get("choices", [{}])[0].get("message", {})
        parsed = extract_json_object(str(message.get("content") or ""))
        if not parsed:
            return None

        product_area = str(parsed.get("product_area") or "").strip()
        request_type = str(parsed.get("request_type") or "").strip()
        llm_response = str(parsed.get("response") or "").strip()
        rationale = str(parsed.get("rationale") or "").strip()
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        if request_type not in ALLOWED_REQUEST_TYPES:
            request_type = ""
        if product_area_options and product_area not in product_area_options:
            product_area = ""
        if not llm_response:
            return None
        return LLMAssessment(
            product_area=product_area,
            request_type=request_type,
            response=llm_response,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=rationale,
        )


def extract_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text") or content.get("refusal")
                if text:
                    parts.append(str(text))
    return "\n".join(parts).strip()

class VectorStoreManager:
    """Simplified Semantic Vector Store using Hugging Face and Numpy."""
    def __init__(self, data_dir: Path, embedder: LocalEmbeddingClient):
        self.data_dir = data_dir
        self.embedder = embedder
        self.chunks: list[DocumentChunk] = []
        self.embedding_vectors: Optional[np.ndarray] = None
        self.cache_dir = data_dir / "index"

    def build(self) -> None:
        """Process documents and generate local embeddings using existing helpers."""
        markdown_files = sorted(self.data_dir.rglob("*.md"))
        
        for path in markdown_files:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            # Using your existing helper functions from main.py
            title = title_from_markdown(raw, path)
            cleaned = clean_markdown(raw)
            company = infer_company(path, self.data_dir)
            product_area = infer_product_area(path, self.data_dir)
            
            text_for_index = f"{title}. {cleaned}"
            
            # Using your existing chunk_words helper
            for chunk in chunk_words(text_for_index):
                self.chunks.append(DocumentChunk(
                    text=chunk, 
                    company=company, 
                    product_area=product_area,
                    source_path=str(path), 
                    title=title
                ))

        if self.chunks:
            print(f"Generating embeddings for {len(self.chunks)} chunks...")
            texts = [f"{c.title}\n{c.text}" for c in self.chunks]
            # This line will now work because np is imported at the top
            vectors = self.embedder.embed_batch(texts)
            self.embedding_vectors = np.array(vectors, dtype="float32")
            print("Semantic index built successfully.")
        



    def semantic_search(self, query: str, company: Optional[str] = None, top_k: int = 3) -> list[RetrievalResult]:
        """Search using Cosine Similarity via Numpy dot product."""
        if self.embedding_vectors is None or not self.chunks:
            return []

        # Get query vector
        query_vec = np.array(self.embedder.embed_batch([query])[0], dtype="float32")
        
        # Dot product of normalized vectors = Cosine Similarity
        # (SentenceTransformer.encode normalizes by default)
        scores = np.dot(self.embedding_vectors, query_vec)
        
        results: list[RetrievalResult] = []
        company_normalized = (company or "").lower()

        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            # Filter by company if specified
            if company_normalized and chunk.company.lower() != company_normalized:
                continue
            
            results.append(RetrievalResult(chunk=chunk, score=float(score)))

        # Sort by best semantic match
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    


    def _fit_faiss(self) -> None:
        # 1. Check if we even have data to index
            if not self.embedding_vectors or not FAISS_AVAILABLE:
                print("DEBUG: FAISS not initialized - check if OPENAI_API_KEY is set and functional.")
                return
            
            # 2. Build the index
            matrix = np.array(self.embedding_vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            
            self.faiss_index = index
            self.faiss_available = True
            
            # 3. Save to disk (Ensure directory exists first)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            index_path = str(self.cache_dir / "faiss.index")
            faiss.write_index(index, index_path)
            print(f"DEBUG: FAISS index saved to {index_path}")

    def faiss_search(self, query: str, company: Optional[str] = None, top_k: int = 3) -> list[RetrievalResult]:
        query_embedding = self.query_embedding(query)
        if query_embedding is None or not self.faiss_available or np is None or faiss is None:
            return []
        query_matrix = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query_matrix)
        search_k = len(self.chunks)
        scores, indices = self.faiss_index.search(query_matrix, search_k)
        company_normalized = (company or "").lower()
        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if company_normalized and chunk.company.lower() != company_normalized:
                continue
            results.append(RetrievalResult(chunk=chunk, score=max(0.0, float(score))))
            if len(results) >= top_k:
                break
        return results


class Retriever:
    def __init__(self, store: VectorStoreManager, mode: str = "tfidf") -> None:
        self.store = store
        if mode == "auto":
            mode = "faiss" if store.faiss_available else "hybrid" if store.embedding_available else "tfidf"
        if mode == "faiss" and not store.faiss_available:
            mode = "embedding" if store.embedding_available else "tfidf"
        if mode in {"embedding", "hybrid"} and not store.embedding_available:
            mode = "tfidf"
        self.mode = mode

    # def search(self, query: str, company: Optional[str] = None, top_k: int = 3) -> list[RetrievalResult]:
    #     if self.mode == "faiss":
    #         faiss_results = self.store.faiss_search(query, company=company, top_k=top_k)
    #         if faiss_results:
    #             return faiss_results

    #     query_vector, query_norm, query_terms = self.store.query_vector(query)
    #     query_embedding = self.store.query_embedding(query) if self.mode in {"embedding", "hybrid"} else None
    #     results: list[RetrievalResult] = []
    #     company_normalized = (company or "").lower()

    #     for idx, chunk in enumerate(self.store.chunks):
    #         if company_normalized and chunk.company.lower() != company_normalized:
    #             continue
    #         tfidf_score = self._tfidf_score(idx, query_vector, query_norm, query_terms, chunk)
    #         embedding_score = 0.0
    #         if query_embedding is not None and idx < len(self.store.embedding_vectors):
    #             embedding_score = max(0.0, dense_cosine(query_embedding, self.store.embedding_vectors[idx]))

    #         if self.mode == "embedding":
    #             score = embedding_score
    #         elif self.mode == "hybrid":
    #             score = (embedding_score * 0.65) + (tfidf_score * 0.35)
    #         else:
    #             score = tfidf_score
    #         if score > 0:
    #             results.append(RetrievalResult(chunk=chunk, score=score))

    #     results.sort(key=lambda item: (item.score, item.chunk.company, item.chunk.source_path), reverse=True)
    #     return results[:top_k]

    def search(self, query: str, company: Optional[str] = None, top_k: int = 3) -> list[RetrievalResult]:
        """Directly uses the new semantic search from VectorStoreManager."""
        return self.store.semantic_search(query, company=company, top_k=top_k)
    
    def resolve_company(self, query: str, requested_company: Optional[str]) -> CompanyResolution:
        requested = normalize_company(requested_company)
        global_results = self.search(query, company=None, top_k=1)
        filtered_results = self.search(query, company=requested, top_k=1) if requested else []
        global_score = global_results[0].score if global_results else 0.0
        filtered_score = filtered_results[0].score if filtered_results else 0.0
        global_company = global_results[0].chunk.company if global_results else requested

        if (
            requested
            and global_company
            and global_company != requested
            and global_score >= COMPANY_OVERRIDE_MIN_SCORE
            and global_score > filtered_score + COMPANY_OVERRIDE_MARGIN
        ):
            return CompanyResolution(requested, global_company, filtered_score, global_score, True)
        return CompanyResolution(requested, requested or global_company, filtered_score, global_score, False)

    def _tfidf_score(
        self,
        idx: int,
        query_vector: dict[str, float],
        query_norm: float,
        query_terms: set[str],
        chunk: DocumentChunk,
    ) -> float:
        vector = self.store.vectors[idx]
        raw_score = sum(query_vector.get(token, 0.0) * weight for token, weight in vector.items())
        raw_score = raw_score / (query_norm * self.store.norms[idx])
        return self._calibrated_score(raw_score, query_terms, chunk)

    @staticmethod
    def _calibrated_score(raw_score: float, query_terms: set[str], chunk: DocumentChunk) -> float:
        chunk_terms = set(tokenize(f"{chunk.title} {chunk.product_area} {chunk.text}"))
        coverage = len(query_terms & chunk_terms) / max(1, len(query_terms))
        title_terms = set(tokenize(chunk.title))
        title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
        score = (raw_score * 2.45) + (coverage * 0.60) + (title_overlap * 0.25)
        if "delete" in query_terms and "delete" in title_terms:
            score += 0.15
        if "conversation" in query_terms and "conversation" in title_terms:
            score += 0.10
        return min(1.0, score)


def normalize_company(company: object) -> Optional[str]:
    text = normalize_text(company)
    if not text or text in {"none", "null", "nan", "n/a"}:
        return None
    if "hacker" in text:
        return "HackerRank"
    if "claude" in text or "anthropic" in text:
        return "Claude"
    if "visa" in text:
        return "Visa"
    return None


class Classifier:

    def classify_request_type(self, query: str, top_score: float) -> str:
        """
        Provides a simple baseline. 
        The LLM will provide the high-intelligence classification later.
        """
        text = normalize_text(query)
        
        # Absolute basics for 'invalid'
        if not text.strip() or (len(tokenize(text)) <= 2 and top_score < 0.30):
            return "invalid"
            
        # Default to product_issue; let the LLM refine to 'bug' or 'feature_request'
        return "product_issue"

    def decide(
        self, 
        query: str, 
        results: list[RetrievalResult], 
        product_area: str, 
        request_type: str, 
        llm_assessment: Optional[LLMAssessment] = None
    ) -> TriageDecision:
        """
        Makes the final decision based on LLM 'needs_human' flag 
        or a strict low-confidence fallback.
        """
        top_score = results[0].score if results else 0.0
        reasons: list[str] = []

        # 1. PRIORITY: If LLM flagged it (Down, Sensitive, or No Permission)
        if llm_assessment and getattr(llm_assessment, 'needs_human', False):
            reasons.append(f"AI Flagged: {llm_assessment.rationale}")

        # 2. RANTS: If LLM labeled it invalid, return REPLIED immediately
        if request_type == "invalid":
            return TriageDecision("replied", product_area, "invalid", False, ("out of scope",))

        # 3. SAFETY FALLBACK: If AI is offline, use score threshold
        if not llm_assessment and top_score < LOW_CONFIDENCE_THRESHOLD:
            reasons.append(f"Low retrieval confidence ({top_score:.2f})")

        status = "escalated" if reasons else "replied"
        return TriageDecision(status, product_area, request_type, top_score < LOW_CONFIDENCE_THRESHOLD, tuple(reasons))
    
class ResponseGenerator:
    def generate(
        self,
        query: str,
        company: Optional[str],
        decision: TriageDecision,
        results: list[RetrievalResult],
        resolution: CompanyResolution,
        llm_assessment: Optional[LLMAssessment] = None,
        label_notes: Optional[list[str]] = None,
    ) -> tuple[str, str]:
        # 1. Logic for Response selection (Existing Logic)
        if llm_assessment and llm_assessment.response and decision.request_type != "invalid":
            response = llm_assessment.response
            if decision.status == "escalated" and "escalat" not in normalize_text(response):
                response = self._escalation_response(query, company, results, decision.reasons)
        elif decision.request_type == "invalid":
            response = self._invalid_response(query)
        elif decision.status == "escalated":
            response = self._escalation_response(query, company, results, decision.reasons)
        else:
            response = self._grounded_response(query, results)

        # 2. Build Human-Readable Justification
        # A: Clean up company logic
        company_status = f"Company: {resolution.resolved_company}"
        if resolution.overridden:
            company_status += f" (Overridden from {resolution.requested_company})"

        # B: Format Retrieval Context
        score_val = results[0].score if results else 0.0
        retrieval_info = f"Match Confidence: {score_val:.2f}"
        
        # C: Format Decision Logic
        if decision.status == "escalated":
            logic_summary = f"Action: ESCALATED due to {', '.join(decision.reasons)}"
        else:
            logic_summary = "Action: REPLIED using documentation"

        # D: Extract Filenames only (no more long paths like F:/Contests/...)
        source_names = [Path(r.chunk.source_path).name for r in results[:2]]
        sources_text = f"Sources: {', '.join(source_names) if source_names else 'None found'}"

        # E: Handle AI/LLM Status
        llm_info = "AI: " + ("; ".join(label_notes) if label_notes else "Active")

        # Combine into a professional, pipe-separated string
        justification = (
            f"{logic_summary} | {company_status} | {retrieval_info} | "
            f"Area: {decision.product_area} | Type: {decision.request_type} | "
            f"{sources_text} | {llm_info}"
        )

        return response, justification

    @staticmethod
    def _invalid_response(query: str) -> str:
        text = normalize_text(query)
        if "thank" in text:
            return "Happy to help."
        return "I am sorry, this request is outside the support scope I can answer from the provided documentation."

    @staticmethod
    def _escalation_response(
        query: str,
        company: Optional[str],
        results: list[RetrievalResult],
        reasons: tuple[str, ...],
    ) -> str:
        text = normalize_text(query)
        prefix = "I cannot safely complete that action directly from the support documentation."
        if "score" in text or "recruiter" in text:
            prefix = "I cannot review, change, or influence assessment scores or recruiter decisions."
        elif "refund" in text or "charge" in text or "payment" in text:
            prefix = "I cannot issue refunds, reverse charges, or make account-specific payment decisions."
        elif "security vulnerability" in text or "bug bounty" in text:
            prefix = "Security vulnerability reports need specialized review."
        elif "identity" in text and "stolen" in text:
            prefix = "Identity theft or possible fraud needs urgent specialist handling."
        elif "lost access" in text or "removed my seat" in text:
            prefix = "Workspace access changes must be handled by the appropriate workspace or organization administrator."

        doc_hint = ""
        if results:
            best = results[0].chunk
            doc_hint = f" The closest support area is {best.product_area} for {best.company}."
        reason = f" Reason: {'; '.join(reasons)}." if reasons else ""
        return f"{prefix} I will escalate this to a human support specialist for review.{doc_hint}{reason}"

    @staticmethod
    def _grounded_response(query: str, results: list[RetrievalResult]) -> str:
        if not results:
            return "I could not find enough support documentation to answer this safely, so this should be reviewed by support."

        query_terms = set(tokenize(query))
        sentences: list[tuple[float, str]] = []
        for result in results[:3]:
            for sentence in split_sentences(result.chunk.text):
                if re.search(r"last (updated|modified)|^_last", sentence, flags=re.I):
                    continue
                sentence_tokens = set(tokenize(sentence))
                if len(sentence_tokens) < 4:
                    continue
                overlap = len(query_terms & sentence_tokens)
                score = overlap + (result.score * 2.0)
                if overlap > 0:
                    sentences.append((score, sentence.strip()))

        if not sentences:
            best = results[0].chunk
            snippet = " ".join(best.text.split()[:90])
            return f"Based on the {best.company} support documentation: {snippet}"

        selected: list[str] = []
        seen: set[str] = set()
        for _, sentence in sorted(sentences, key=lambda item: item[0], reverse=True):
            normalized = normalize_text(sentence)
            if normalized in seen:
                continue
            selected.append(sentence)
            seen.add(normalized)
            if len(selected) >= 4:
                break
        return " ".join(selected)


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"\s+", " ", piece).strip(" -") for piece in pieces if piece.strip()]


def choose_product_area(query: str, results: list[RetrievalResult]) -> str:
    if not results:
        return ""
    text = f" {normalize_text(query)} "
    top = results[0].chunk

    if top.company == "HackerRank":
        if re.search(r"\b(mock interview|resume builder|certificate|community|apply tab)\b", text):
            return "community"
        if re.search(r"\b(subscription|billing|payment|refund|money)\b", text):
            return "community" if "mock interview" in text else "settings"
        if re.search(r"\b(remove|deactivate|delete)\b", text) and re.search(
            r"\b(user|interviewer|employee|member|hiring account)\b", text
        ):
            return "settings"
        if re.search(r"\b(interview|lobby|zoom)\b", text) and "assessment" not in text:
            return "interviews"
        if re.search(r"\b(test|assessment|candidate|submission|score|recruiter|variant|invite|reinvite|extra time)\b", text):
            return "screen"

    if top.company == "Visa":
        if "traveller" in text or "traveler" in text:
            return "travel_support"
        if "dispute" in text or "charge" in text:
            return "dispute_resolution"
        if "lost or stolen" in text or "stolen visa card" in text or "minimum" in text:
            return "general_support"

    if top.company == "Claude":
        if "conversation" in text or "temporary chat" in text:
            return "conversation_management"
        if "crawl" in text or "data" in text or "privacy" in text:
            return "privacy"
        if "bedrock" in text:
            return "amazon_bedrock"
        if "lti" in text or "canvas" in text or "students" in text:
            return "claude_for_education"

    return top.product_area


def reconcile_labels(
    baseline_product_area: str,
    baseline_request_type: str,
    llm_assessment: Optional[LLMAssessment],
    retrieval_score: float,
) -> tuple[str, str, list[str]]:
    """Prioritize LLM intelligence over local baselines."""
    notes: list[str] = []
    
    # 1. Handle LLM failure immediately
    if not llm_assessment:
        notes.append("AI: Offline - using local baseline")
        return baseline_product_area, baseline_request_type, notes

    # 2. Start with LLM values (The "Brain")
    product_area = llm_assessment.product_area or baseline_product_area
    request_type = llm_assessment.request_type or baseline_request_type
    
    # 3. Add transparency to the Justification
    if llm_assessment.product_area == baseline_product_area:
        notes.append(f"Area: {product_area} (Confirmed by AI)")
    else:
        notes.append(f"Area: {product_area} (AI override)")

    if llm_assessment.request_type == baseline_request_type:
        notes.append(f"Type: {request_type} (Confirmed by AI)")
    else:
        notes.append(f"Type: {request_type} (AI override)")

    if llm_assessment.rationale:
        notes.append(f"Rationale: {llm_assessment.rationale}")

    return product_area, request_type, notes


class TriageEngine:
    def __init__(
        self,
        data_dir: Path,
        debug: bool = False,
        retrieval_mode: str = "auto",
        use_llm: bool = True,
        llm_provider: str = "auto",
    ) -> None:
        load_env_file(Path(__file__).resolve().parents[1] / ".env")

        self.openrouter_client = OpenRouterClient()
        self.local_embedder = LocalEmbeddingClient()

        self.store = VectorStoreManager(data_dir, embedder=self.local_embedder)
        self.store.build()
        self.retriever = Retriever(self.store)
        self.classifier = Classifier()
        self.generator = ResponseGenerator()
        self.debug = debug
        self.llm_client = self._select_llm_client(llm_provider)
        self.use_llm = use_llm and self.llm_client is not None

    def _select_llm_client(self, llm_provider: str) -> Optional[Any]:
        if llm_provider == "openai":
            return self.openai_client if self.openai_client.enabled else None
        if llm_provider == "openrouter":
            return self.openrouter_client if self.openrouter_client.enabled else None
        if self.openrouter_client.enabled:
            return self.openrouter_client
        if self.openai_client.enabled:
            return self.openai_client
        return None

    def process_ticket(self, issue: str, subject: str, company: object) -> dict[str, str]:
        query = f"{subject or ''}\n{issue or ''}".strip()
        resolution = self.retriever.resolve_company(query, company)
        
        # 1. Retrieval
        results = self.retriever.search(query, company=resolution.resolved_company, top_k=3)
        top_score = results[0].score if results else 0.0
        
        # 2. Baseline Labels
        base_area = choose_product_area(query, results)
        base_type = self.classifier.classify_request_type(query, top_score)
        
        # 3. LLM Assessment (The Brain)
        # It now returns {needs_human: True/False} based on your new prompt
        llm_assessment = self._llm_assessment(
            query, resolution.resolved_company, "pending", base_area, base_type, results
        )
        
        # 4. Reconcile
        product_area, request_type, label_notes = reconcile_labels(
            base_area, base_type, llm_assessment, top_score
        )
        
        # 5. FINAL DECISION (Pass the llm_assessment here!)
        decision = self.classifier.decide(
            query, results, product_area, request_type, llm_assessment=llm_assessment
        )
        
        # 6. Response
        response, justification = self.generator.generate(
            query, resolution.resolved_company, decision, results, resolution, 
            llm_assessment=llm_assessment, label_notes=label_notes
        )
        
        return {
            "issue": issue, "subject": subject, "company": company,
            "status": decision.status, "product_area": product_area,
            "request_type": request_type, "response": response, "justification": justification
        }

    def _llm_assessment(
        self,
        query: str,
        company: Optional[str],
        preliminary_status: str,
        product_area: str,
        request_type: str,
        results: list[RetrievalResult],
    ) -> Optional[LLMAssessment]:
        if not self.use_llm:
            return None
        try:
            return self.llm_client.assess_ticket(
                query=query,
                company=company,
                preliminary_status=preliminary_status,
                tfidf_product_area=product_area,
                tfidf_request_type=request_type,
                results=results,
            )
        except Exception as exc:
            if self.debug:
                print(json.dumps({"llm_error": str(exc)}, ensure_ascii=True))
            return None

    @staticmethod
    def _log_debug(
        query: str,
        resolution: CompanyResolution,
        results: list[RetrievalResult],
        decision: TriageDecision,
    ) -> None:
        event = {
            "query": query[:120],
            "requested_company": resolution.requested_company,
            "resolved_company": resolution.resolved_company,
            "filtered_score": round(resolution.filtered_score, 4),
            "global_score": round(resolution.global_score, 4),
            "overridden": resolution.overridden,
            "status": decision.status,
            "product_area": decision.product_area,
            "request_type": decision.request_type,
            "scores": [round(result.score, 4) for result in results],
            "sources": [result.chunk.source_path for result in results],
        }
        print(json.dumps(event, ensure_ascii=True))


def run(
    input_csv: Path,
    output_csv: Path,
    data_dir: Path,
    debug: bool = False,
    retrieval_mode: str = "auto",
    use_llm: bool = True,
    llm_provider: str = "auto",
) -> None:
    engine = TriageEngine(
        data_dir=data_dir,
        debug=debug,
        retrieval_mode=retrieval_mode,
        use_llm=use_llm,
        llm_provider=llm_provider,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    fieldnames = ["issue","subject","company","status", "product_area", "request_type", "response", "justification"]
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            prediction = engine.process_ticket(
                issue=row.get("Issue") or row.get("issue") or "",
                subject=row.get("Subject") or row.get("subject") or "",
                company=row.get("Company") or row.get("company") or "",
            )
            writer.writerow(prediction)

    print(f"Processed {len(rows)} tickets -> {output_csv}")


def check_openrouter_connectivity() -> int:
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    client = OpenRouterClient()
    print(f"OPENROUTER_API_KEY detected: {client.enabled}")
    if not client.enabled:
        print("OpenRouter check skipped: set OPENROUTER_API_KEY in the process environment or repo .env file.")
        return 1
    try:
        response = client._post(
            "/chat/completions",
            {
                "model": client.chat_model,
                "messages": [{"role": "user", "content": "Reply with only: ok"}],
                "temperature": 0,
                "max_tokens": 20,
            },
        )
        message = response.get("choices", [{}])[0].get("message", {})
        text = str(message.get("content") or "").strip()
        print(f"OpenRouter response call succeeded: model={client.chat_model} output_preview={text[:80]!r}")
    except Exception as exc:
        print(f"OpenRouter response call failed: {exc}")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run deterministic support triage over support_tickets.csv")
    parser.add_argument("--input", type=Path, default=repo_root / "support_tickets" / "support_tickets.csv")
    parser.add_argument("--output", type=Path, default=repo_root / "support_tickets" / "output.csv")
    parser.add_argument("--data", type=Path, default=repo_root / "data")
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "tfidf", "embedding", "hybrid", "faiss"),
        default=os.getenv("RETRIEVAL_MODE", "auto"),
        help="auto uses FAISS when available, then hybrid embeddings, otherwise TF-IDF",
    )
    parser.add_argument("--disable-llm", action="store_true", help="Disable LLM labels/responses even if OPENAI_API_KEY exists")
    parser.add_argument(
        "--llm-provider",
        choices=("auto", "openai", "openrouter"),
        default=os.getenv("LLM_PROVIDER", "auto"),
        help="auto prefers OpenRouter when OPENROUTER_API_KEY exists, then OpenAI",
    )
    parser.add_argument("--check-openai", action="store_true", help="Run a small OpenAI embedding and response smoke test")
    parser.add_argument("--check-openrouter", action="store_true", help="Run a small OpenRouter chat completion smoke test")
    parser.add_argument("--debug", action="store_true", help="Print retrieval scores and company override decisions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_openrouter:
        raise SystemExit(check_openrouter_connectivity())
    run(
        input_csv=args.input,
        output_csv=args.output,
        data_dir=args.data,
        debug=args.debug,
        retrieval_mode=args.retrieval_mode,
        use_llm=not args.disable_llm,
        llm_provider=args.llm_provider,
    )


if __name__ == "__main__":
    main()
