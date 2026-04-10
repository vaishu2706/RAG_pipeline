"""
Step 8: Evaluation using RAGAS
Metrics: faithfulness, answer_relevancy, context_recall (retrieval quality)
"""

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
from rag import build_rag_chain

load_dotenv()


def run_evaluation(eval_samples: list[dict]) -> dict:
    """
    eval_samples: list of {"question": str, "ground_truth": str}
    Returns RAGAS metric scores.
    """
    if not eval_samples:
        raise ValueError("No valid evaluation samples provided")

    chain = build_rag_chain()

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in eval_samples:
        result = chain.invoke({"query": sample["question"]})
        questions.append(sample["question"])
        answers.append(result["result"])
        contexts.append([doc.page_content for doc in result["source_documents"]])
        ground_truths.append(sample["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall, answer_correctness])
    return scores.to_pandas().to_dict(orient="records")
