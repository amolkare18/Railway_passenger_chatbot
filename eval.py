from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langsmith import Client

import sys; sys.path.append(".")
from config import GROQ_API_KEY, LANGSMITH_API_KEY
from app.rag import answer_query

client       = Client(api_key=LANGSMITH_API_KEY)
dataset_name = "RAG_DB"

# Groq + json_mode avoids the True/False tool-calling parse bug
_eval_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=GROQ_API_KEY)

# LangSmith normalizes dataset field names to lowercase
def _ref(reference_outputs: dict) -> str:
    return (
        reference_outputs.get("reference_output")
        or reference_outputs.get("Reference_Output")
        or next(iter(reference_outputs.values()), "")
    )


# ── Correctness ───────────────────────────────────────────────────
class CorrectnessGrade(BaseModel):
    explanation: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(description="Score 0.0–1.0: 1.0 = fully correct, 0.0 = completely wrong", ge=0.0, le=1.0)

correctness_instructions = """You are a teacher grading a quiz. You will be given a QUESTION, the GROUND TRUTH ANSWER, and the STUDENT ANSWER.
Grade criteria:
(1) Score based ONLY on factual accuracy relative to the ground truth.
(2) Penalise conflicting statements heavily.
(3) Extra correct information is fine; missing details reduce the score proportionally.

Respond with valid JSON only: {"explanation": "...", "score": <float 0.0-1.0>}"""

grader_llm = _eval_llm.with_structured_output(CorrectnessGrade, method="json_mode")

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
    """Evaluator for RAG answer accuracy (0.0–1.0)."""
    prompt = (
        f"QUESTION: {inputs['input']}\n"
        f"GROUND TRUTH ANSWER: {_ref(reference_outputs)}\n"
        f"STUDENT ANSWER: {outputs['answer']}"
    )
    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user",   "content": prompt},
    ])
    return grade.score


# ── Relevance ─────────────────────────────────────────────────────
class RelevanceGrade(BaseModel):
    explanation: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(description="Score 0.0–1.0: 1.0 = fully relevant, 0.0 = completely irrelevant", ge=0.0, le=1.0)

relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a STUDENT ANSWER.
Grade criteria:
(1) Is the answer concise and directly addressing the question?
(2) Does it actually help answer what was asked?
Score proportionally — a partial answer gets a middle score like 0.5 or 0.6.

Respond with valid JSON only: {"explanation": "...", "score": <float 0.0-1.0>}"""

relevance_llm = _eval_llm.with_structured_output(RelevanceGrade, method="json_mode")

def relevance(inputs: dict, outputs: dict) -> float:
    """Evaluator for RAG answer helpfulness (0.0–1.0)."""
    prompt = f"QUESTION: {inputs['input']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user",   "content": prompt},
    ])
    return grade.score


# ── Groundedness ──────────────────────────────────────────────────
class GroundedGrade(BaseModel):
    explanation: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(description="Score 0.0–1.0: 1.0 = fully grounded in facts, 0.0 = hallucinated", ge=0.0, le=1.0)

grounded_instructions = """You are a teacher grading a quiz. You will be given FACTS and a STUDENT ANSWER.
Grade criteria:
(1) Score how grounded the answer is in the provided FACTS.
(2) Penalise information not supported by the FACTS.
(3) Partial groundedness gets a proportional score like 0.6 or 0.7.

Respond with valid JSON only: {"explanation": "...", "score": <float 0.0-1.0>}"""

grounded_llm = _eval_llm.with_structured_output(GroundedGrade, method="json_mode")

def groundedness(inputs: dict, outputs: dict) -> float:
    """Evaluator for RAG answer groundedness (0.0–1.0)."""
    prompt = f"FACTS: {outputs['context']}\nSTUDENT ANSWER: {outputs['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user",   "content": prompt},
    ])
    return grade.score


# ── Retrieval Relevance ───────────────────────────────────────────
class RetrievalRelevanceGrade(BaseModel):
    explanation: str = Field(description="Step-by-step reasoning for the score")
    score: float = Field(description="Score 0.0–1.0: 1.0 = retrieved facts are completely relevant to the question", ge=0.0, le=1.0)

retrieval_relevance_instructions = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS retrieved for the student.
Grade criteria:
(1) Identify how relevant the FACTS are to the QUESTION.
(2) If the facts contain strong keyword or semantic overlap, score high (0.8–1.0).
(3) Partial relevance (some related, some not) gets a middle score (0.4–0.7).
(4) Completely unrelated facts score 0.0–0.2.

Respond with valid JSON only: {"explanation": "...", "score": <float 0.0-1.0>}"""

retrieval_relevance_llm = _eval_llm.with_structured_output(RetrievalRelevanceGrade, method="json_mode")

def retrieval_relevance(inputs: dict, outputs: dict) -> float:
    """Evaluator for retrieval relevance (0.0–1.0)."""
    prompt = f"FACTS: {outputs['context']}\nQUESTION: {inputs['input']}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user",   "content": prompt},
    ])
    return grade.score


# ── Target & Evaluation Run ───────────────────────────────────────
def target(inputs: dict) -> dict:
    return answer_query(inputs["input"])

experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "groq llama-3.1-8b, pinecone RAG"},
)

# experiment_results.to_pandas()
