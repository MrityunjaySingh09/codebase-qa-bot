"""
eval/ragas_eval.py — Evaluate RAG pipeline quality using RAGAS metrics.

Metrics measured:
  - Faithfulness:       Are claims in the answer supported by the retrieved context?
  - Answer Relevancy:   Does the answer actually address the question asked?
  - Context Precision:  Are the retrieved chunks relevant to the question?
  - Context Recall:     Did retrieval surface the chunks needed to answer?

How it works:
  1. Load a test question set (JSON file or built-in defaults)
  2. Run each question through the full RAG pipeline (retrieve + generate)
  3. Feed (question, answer, contexts, ground_truth) into RAGAS
  4. Save scores to eval/results/ as JSON + human-readable Markdown table

RAGAS uses an LLM internally to judge faithfulness/relevancy.
We point it at the same local Ollama instance — fully free, no OpenAI needed.

Usage:
  python -m eval.ragas_eval --repo-url https://github.com/owner/repo
  python -m eval.ragas_eval --repo-url https://github.com/owner/repo \\
                             --questions eval/sample_questions.json \\
                             --output eval/results/my_run.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Default test questions (used when no custom file provided)
# ---------------------------------------------------------------------------

DEFAULT_QUESTIONS = [
    {
        "question": "What is the main purpose of this repository?",
        "ground_truth": "The repository serves a specific software purpose that can be determined from its README and main entry points.",
    },
    {
        "question": "How is the project structured and what are the main modules?",
        "ground_truth": "The project has a defined directory structure with modules for different concerns.",
    },
    {
        "question": "What dependencies or libraries does this project use?",
        "ground_truth": "The project uses several external libraries listed in its dependency files.",
    },
    {
        "question": "Where is the main entry point or application startup code?",
        "ground_truth": "There is a main file or entry point where the application initialises.",
    },
    {
        "question": "What testing approach and test files exist in this project?",
        "ground_truth": "The project contains test files that verify the functionality of its components.",
    },
]


# ---------------------------------------------------------------------------
# RAGAS evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    repo_url: str,
    questions: Optional[list[dict]] = None,
    output_path: Optional[Path] = None,
    top_k: int = 5,
) -> dict:
    """
    Run RAGAS evaluation over a question set for a given indexed repo.

    Args:
        repo_url:    Canonical GitHub URL of an already-indexed repo.
        questions:   List of {"question": str, "ground_truth": str} dicts.
                     Defaults to DEFAULT_QUESTIONS.
        output_path: Where to save results JSON. Auto-generated if None.
        top_k:       Chunks to retrieve per question.

    Returns:
        Dict with per-question results and aggregate scores.
    """
    from config import get_settings
    from rag.pipeline import answer as rag_answer

    cfg = get_settings()
    qs = questions or DEFAULT_QUESTIONS
    logger.info(f"Starting RAGAS eval: {len(qs)} questions on {repo_url}")

    # ── Step 1: Collect RAG outputs ──────────────────────────────────────
    samples = []
    for i, q in enumerate(qs, 1):
        logger.info(f"[{i}/{len(qs)}] Running: {q['question'][:60]}")
        try:
            result = rag_answer(q["question"], repo_url, top_k=top_k)
            samples.append({
                "question":      q["question"],
                "answer":        result["answer"],
                "contexts":      [s.content for s in result["sources"]],
                "ground_truth":  q.get("ground_truth", ""),
                "source_files":  [s.display_location() for s in result["sources"]],
            })
            logger.success(f"  ✓ Answer: {result['answer'][:80]} ...")
        except Exception as exc:
            logger.error(f"  ✗ Failed: {exc}")
            samples.append({
                "question":     q["question"],
                "answer":       f"ERROR: {exc}",
                "contexts":     [],
                "ground_truth": q.get("ground_truth", ""),
                "source_files": [],
            })

    # ── Step 2: Run RAGAS ─────────────────────────────────────────────────
    scores = _run_ragas_metrics(samples, cfg)

    # ── Step 3: Build results dict ────────────────────────────────────────
    results = {
        "repo_url":   repo_url,
        "timestamp":  datetime.utcnow().isoformat(),
        "model":      cfg.ollama_llm_model,
        "embed_model": cfg.ollama_embed_model,
        "num_questions": len(qs),
        "top_k":      top_k,
        "aggregate":  scores,
        "per_question": samples,
    }

    # ── Step 4: Save results ──────────────────────────────────────────────
    if output_path is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"eval_{ts}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    logger.success(f"Results saved to {output_path}")

    # ── Step 5: Print Markdown summary ───────────────────────────────────
    _print_summary(results)

    return results


def _run_ragas_metrics(samples: list[dict], cfg) -> dict:
    """
    Run RAGAS metrics. Returns a dict of metric_name → float score.
    Falls back to placeholder scores if RAGAS or LLM is unavailable.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        # Point RAGAS at local Ollama
        ragas_llm = LangchainLLMWrapper(
            ChatOllama(
                model=cfg.ollama_llm_model,
                base_url=cfg.ollama_base_url,
                temperature=0,
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                model=cfg.ollama_embed_model,
                base_url=cfg.ollama_base_url,
            )
        )

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for m in metrics:
            m.llm = ragas_llm
            if hasattr(m, "embeddings"):
                m.embeddings = ragas_embeddings

        # Build HuggingFace Dataset (RAGAS requirement)
        valid_samples = [s for s in samples if s["contexts"] and "ERROR" not in s["answer"]]
        if not valid_samples:
            logger.warning("No valid samples for RAGAS — skipping metrics")
            return _null_scores()

        dataset = Dataset.from_dict({
            "question":     [s["question"]     for s in valid_samples],
            "answer":       [s["answer"]        for s in valid_samples],
            "contexts":     [s["contexts"]      for s in valid_samples],
            "ground_truth": [s["ground_truth"]  for s in valid_samples],
        })

        logger.info("Running RAGAS evaluation (this may take a few minutes) ...")
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        return {
            "faithfulness":      float(df["faithfulness"].mean()),
            "answer_relevancy":  float(df["answer_relevancy"].mean()),
            "context_precision": float(df["context_precision"].mean()),
            "context_recall":    float(df["context_recall"].mean()),
        }

    except ImportError as exc:
        logger.warning(f"RAGAS not available ({exc}) — returning placeholder scores")
        return _null_scores()
    except Exception as exc:
        logger.error(f"RAGAS evaluation failed: {exc}")
        return _null_scores()


def _null_scores() -> dict:
    return {
        "faithfulness":      None,
        "answer_relevancy":  None,
        "context_precision": None,
        "context_recall":    None,
    }


def _print_summary(results: dict) -> None:
    """Print a Markdown-formatted summary table to stdout."""
    agg = results["aggregate"]
    repo = results["repo_url"]
    ts   = results["timestamp"]

    def _fmt(v) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"""
╔══════════════════════════════════════════════════════╗
║              RAGAS Evaluation Results                ║
╠══════════════════════════════════════════════════════╣
║  Repo:      {repo[:42]:<42} ║
║  Timestamp: {ts[:19]:<42} ║
║  Model:     {results['model']:<42} ║
║  Questions: {results['num_questions']:<42} ║
╠══════════════════════════════════════════════════════╣
║  Metric                Score                         ║
║  ─────────────────────────────────────────────────── ║
║  Faithfulness          {_fmt(agg['faithfulness']):<30} ║
║  Answer Relevancy      {_fmt(agg['answer_relevancy']):<30} ║
║  Context Precision     {_fmt(agg['context_precision']):<30} ║
║  Context Recall        {_fmt(agg['context_recall']):<30} ║
╚══════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Load latest results (used by Streamlit UI)
# ---------------------------------------------------------------------------

def load_latest_results(repo_url: Optional[str] = None) -> Optional[dict]:
    """
    Load the most recent eval results from eval/results/.
    Returns None if no results exist yet.
    """
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None

    files = sorted(results_dir.glob("eval_*.json"), reverse=True)
    if not files:
        return None

    for f in files:
        try:
            data = json.loads(f.read_text())
            if repo_url is None or data.get("repo_url") == repo_url:
                return data
        except Exception:
            continue
    return None


def get_results_markdown_table(results: dict) -> str:
    """Format eval results as a Markdown table for README / UI display."""
    agg = results["aggregate"]

    def _bar(v: Optional[float], width: int = 20) -> str:
        if v is None:
            return "N/A"
        filled = int(v * width)
        return "█" * filled + "░" * (width - filled) + f"  {v:.3f}"

    lines = [
        "| Metric | Score | Bar |",
        "|--------|-------|-----|",
        f"| Faithfulness | {agg['faithfulness'] or 'N/A'} | {_bar(agg['faithfulness'])} |",
        f"| Answer Relevancy | {agg['answer_relevancy'] or 'N/A'} | {_bar(agg['answer_relevancy'])} |",
        f"| Context Precision | {agg['context_precision'] or 'N/A'} | {_bar(agg['context_precision'])} |",
        f"| Context Recall | {agg['context_recall'] or 'N/A'} | {_bar(agg['context_recall'])} |",
        "",
        f"_Evaluated on {results['num_questions']} questions · "
        f"Model: `{results['model']}` · "
        f"Embed: `{results['embed_model']}` · "
        f"Timestamp: {results['timestamp'][:19]}_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on an indexed repo"
    )
    parser.add_argument(
        "--repo-url", required=True,
        help="Canonical GitHub URL of the indexed repo"
    )
    parser.add_argument(
        "--questions", default=None,
        help="Path to JSON file with [{question, ground_truth}] entries"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save JSON results (auto-generated if not set)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve per question (default: 5)"
    )
    args = parser.parse_args()

    custom_questions = None
    if args.questions:
        q_path = Path(args.questions)
        if not q_path.exists():
            print(f"Error: questions file not found: {q_path}", file=sys.stderr)
            sys.exit(1)
        custom_questions = json.loads(q_path.read_text())

    run_evaluation(
        repo_url=args.repo_url,
        questions=custom_questions,
        output_path=Path(args.output) if args.output else None,
        top_k=args.top_k,
    )
