"""
data_processing/paper_ingestion.py

Downloads, parses, chunks, and indexes prompt injection research papers
into a dedicated ChromaDB collection ("research_papers").

Install:
    pip install arxiv pymupdf sentence-transformers chromadb requests

One-time setup before running evaluation:
    python -m data_processing.paper_ingestion

Test a query:
    python -m data_processing.paper_ingestion --test "ignore previous instructions"

Force full re-index:
    python -m data_processing.paper_ingestion --force
"""

import os
import re
import time
import logging
import hashlib
import requests
import arxiv
import fitz          # PyMuPDF — import name is 'fitz'
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PAPER REGISTRY  — add more papers here anytime, pipeline is idempotent
# ---------------------------------------------------------------------------
PAPERS = [
    {
        "arxiv_id":   "2302.12173",
        "title":      "Not What You've Signed Up For: Indirect Prompt Injection",
        "authors":    "Greshake et al.",
        "year":       2023,
        "categories": ["indirect_injection", "real_world_attacks", "taxonomy"],
        "focus":      "attack",
    },
    {
        "arxiv_id":   "2306.05499",
        "title":      "Prompt Injection Attack against LLM-Integrated Applications (HouYi)",
        "authors":    "Liu et al.",
        "year":       2023,
        "categories": ["black_box_attack", "context_partition", "direct_injection"],
        "focus":      "attack",
    },
    {
        "arxiv_id":   "2310.12815",
        "title":      "Formalizing and Benchmarking Prompt Injection Attacks and Defenses",
        "authors":    "Liu et al.",
        "year":       2023,
        "categories": ["benchmark", "taxonomy", "defense",
                       "direct_injection", "indirect_injection"],
        "focus":      "both",
    },
    {
        "arxiv_id":   "2403.04957",
        "title":      "Automatic and Universal Prompt Injection Attacks",
        "authors":    "Liu et al.",
        "year":       2024,
        "categories": ["universal_attack", "gradient_based", "automated"],
        "focus":      "attack",
    },
    {
        "arxiv_id":   "2411.00459",
        "title":      "Defense Against Prompt Injection by Leveraging Attack Techniques",
        "authors":    "Chen et al.",
        "year":       2024,
        "categories": ["defense", "direct_injection", "indirect_injection"],
        "focus":      "defense",
    },
]

CHUNK_SIZE    = 600   # characters per chunk
CHUNK_OVERLAP = 120   # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# 2. MAIN CLASS
# ---------------------------------------------------------------------------
class PaperIngestion:
    """
    Full pipeline: download PDF → extract text → chunk → embed → upsert ChromaDB.

    Completely idempotent — re-running skips already-indexed chunks.
    Uses a SEPARATE collection ("research_papers") so it never interferes
    with your existing "malicious_prompts" collection in RagDefense.
    """

    COLLECTION_NAME = "research_papers"
    PDF_DIR         = "./data_processing/papers"
    CHROMA_PATH     = "./chroma_db"      # same root dir as RagDefense

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        os.makedirs(self.PDF_DIR, exist_ok=True)

        self.model  = SentenceTransformer(model_name)
        self.chroma = chromadb.PersistentClient(path=self.CHROMA_PATH)

        # Separate collection from malicious_prompts
        self.collection = self.chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Prompt injection research paper chunks"}
        )
        logger.info(
            f"Collection '{self.COLLECTION_NAME}' ready "
            f"({self.collection.count()} chunks already indexed)."
        )

    # ── PUBLIC API ─────────────────────────────────────────────────────────

    def run(self, force_reindex: bool = False):
        """
        Run the ingestion pipeline for all papers in the PAPERS registry.
        Set force_reindex=True to wipe and rebuild from scratch.
        """
        if force_reindex:
            logger.warning("force_reindex=True → clearing paper collection.")
            self.chroma.delete_collection(self.COLLECTION_NAME)
            self.collection = self.chroma.get_or_create_collection(
                name=self.COLLECTION_NAME
            )

        for paper in PAPERS:
            self._process_paper(paper)

        logger.info(
            f"Ingestion complete. "
            f"Total chunks indexed: {self.collection.count()}"
        )

    def search(self, query: str, k: int = 5) -> str:
        """
        Retrieve the top-k most relevant paper chunks for a query.
        Returns a formatted, citation-bearing string ready to inject
        directly into an LLM prompt as grounding context.
        """
        if self.collection.count() == 0:
            return "[No papers indexed. Run PaperIngestion().run() first.]"

        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=emb,
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks    = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = []
        for chunk, meta, dist in zip(chunks, metadatas, distances):
            formatted.append(
                f"[Source: {meta.get('title', '?')} — "
                f"{meta.get('authors', '?')}, {meta.get('year', '?')} | "
                f"Focus: {meta.get('focus', '?')} | "
                f"Categories: {meta.get('categories', '?')} | "
                f"Relevance: {1 - dist:.2f}]\n{chunk}"
            )

        return "\n\n---\n\n".join(formatted)

    # ── INTERNAL PIPELINE ──────────────────────────────────────────────────

    def _process_paper(self, paper: dict):
        arxiv_id = paper["arxiv_id"]
        pdf_path = os.path.join(
            self.PDF_DIR,
            f"{arxiv_id.replace('/', '_')}.pdf"
        )

        # Step 1: Download (skip if already cached)
        if not os.path.exists(pdf_path):
            ok = self._download(arxiv_id, pdf_path)
            if not ok:
                logger.warning(f"Skipping {arxiv_id} — download failed.")
                return
        else:
            logger.info(f"Cached PDF found: {pdf_path}")

        # Step 2: Extract text
        text = self._extract_text(pdf_path)
        if not text.strip():
            logger.warning(f"No text from {pdf_path}. Skipping.")
            return

        # Step 3: Chunk
        chunks = self._chunk(text)
        logger.info(f"  [{arxiv_id}] {len(chunks)} chunks extracted.")

        # Step 4: Filter already-indexed chunks (idempotency check)
        new_docs, new_ids, new_meta = [], [], []
        for i, chunk in enumerate(chunks):
            cid = self._chunk_id(arxiv_id, i)
            if self.collection.get(ids=[cid])["ids"]:
                continue   # already in DB
            new_docs.append(chunk)
            new_ids.append(cid)
            new_meta.append({
                "arxiv_id":   arxiv_id,
                "title":      paper["title"],
                "authors":    paper["authors"],
                "year":       str(paper["year"]),
                "categories": ", ".join(paper["categories"]),
                "focus":      paper["focus"],
                "chunk_idx":  str(i),
            })

        if not new_docs:
            logger.info(f"  [{arxiv_id}] All chunks already indexed. Skipping.")
            return

        # Step 5: Embed and upsert
        logger.info(f"  [{arxiv_id}] Embedding {len(new_docs)} new chunks...")
        embeddings = self.model.encode(
            new_docs,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        self.collection.upsert(
            documents=new_docs,
            embeddings=embeddings,
            ids=new_ids,
            metadatas=new_meta,
        )
        logger.info(f"  [{arxiv_id}] Upserted {len(new_docs)} chunks.")

    def _download(self, arxiv_id: str, save_path: str) -> bool:
        """
        Download via arxiv Python library, falling back to direct HTTP.
        Sleeps 3s between downloads to respect arXiv's rate limit policy.
        """
        # Method A: arxiv library
        try:
            client  = arxiv.Client()
            results = list(client.results(arxiv.Search(id_list=[arxiv_id])))
            if results:
                # export subdomain avoids IP blocks on bulk downloads
                pdf_url = results[0].pdf_url.replace(
                    "arxiv.org", "export.arxiv.org"
                )
                logger.info(f"  Downloading {arxiv_id} → {pdf_url}")
                r = requests.get(pdf_url, timeout=30)
                if r.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(r.content)
                    time.sleep(3)
                    return True
        except Exception as e:
            logger.warning(f"  arxiv lib failed for {arxiv_id}: {e}")

        # Method B: direct URL fallback
        try:
            url = f"https://export.arxiv.org/pdf/{arxiv_id}"
            logger.info(f"  Fallback: {url}")
            r = requests.get(url, timeout=30, headers={
                "User-Agent": "research-bot/1.0 (prompt-injection-defense)"
            })
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                time.sleep(3)
                return True
        except Exception as e:
            logger.error(f"  Fallback also failed for {arxiv_id}: {e}")

        return False

    def _extract_text(self, pdf_path: str) -> str:
        """Extract and clean text from a PDF using PyMuPDF."""
        try:
            doc  = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            return text
        except Exception as e:
            logger.error(f"Extraction error {pdf_path}: {e}")
            return ""

    def _chunk(self, text: str) -> list:
        """
        Sliding-window chunker that snaps to sentence boundaries
        rather than cutting mid-sentence.
        """
        chunks = []
        start  = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            if end < len(text):
                boundary = text.rfind(". ", start, end)
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - CHUNK_OVERLAP

        return chunks

    @staticmethod
    def _chunk_id(arxiv_id: str, idx: int) -> str:
        """Deterministic, stable chunk ID across runs."""
        return hashlib.md5(f"{arxiv_id}::chunk::{idx}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# 3. SINGLETON + TOOL FUNCTION  (imported by agentic_guardrail.py)
# ---------------------------------------------------------------------------
_instance = None


def get_ingestion() -> PaperIngestion:
    global _instance
    if _instance is None:
        _instance = PaperIngestion()
    return _instance


def search_research_papers(query: str) -> str:
    """
    Search indexed prompt injection research papers for content relevant
    to the query. Returns cited paper excerpts with relevance scores.

    Use this SECOND (after check_malicious_database) to get academic
    grounding — identifies which known attack technique the input matches,
    with a citation to the source paper.
    """
    return get_ingestion().search(query, k=5)


# ---------------------------------------------------------------------------
# 4. CLI ENTRYPOINT
#    python -m data_processing.paper_ingestion [--force] [--test QUERY]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest prompt injection research papers into ChromaDB."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Wipe and re-index all papers from scratch."
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Run a test search query after ingestion."
    )
    args = parser.parse_args()

    ingestion = PaperIngestion()
    ingestion.run(force_reindex=args.force)

    if args.test:
        print("\n" + "=" * 60)
        print(f"TEST QUERY: {args.test}")
        print("=" * 60)
        print(ingestion.search(args.test))