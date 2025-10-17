import json
from pathlib import Path

from llama_index.core import PromptTemplate
from llama_index.core.llama_dataset.generator import RagDatasetGenerator

from config import settings
from retrieval.rag_helper import RAGHelper


def generate_question_answer_pairs(
    num_question: int,
    rag_helper: RAGHelper | None = None,
) -> str:
    """
    Apply the RAG helper query engine to answer a user question.

    Args:
        question (str): The user's question

    Returns:
        str: The answer from the RAG system or an error message
    """

    if rag_helper is None or rag_helper._index is None:
        return "❌ Error: No index has been built yet. Please build an index first."

    try:

        dataset_generator = RagDatasetGenerator(
            nodes=rag_helper._nodes[:num_question],
            llm=rag_helper.llm,  # Explicitly providing the LLM here
            num_questions_per_chunk=settings.rag.nb_questions_per_chunk,
            text_question_template=PromptTemplate(
                """
### CONTEXT ###
{context_str}
---------------------
### TOPIC ###
{query_str}
---------------------
### INSTRUCTIONS ###
Based exclusively on the provided CONTEXT, formulate a list of questions related to the TOPIC. Each question must be answerable using only the information given in the text.
""",
            ),
        )

        qa_set = dataset_generator.generate_dataset_from_nodes()
        qa_pairs = [(r.query, r.reference_answer) for r in qa_set.examples]

        temp_dir: Path = Path("temp_qa")
        temp_dir.mkdir(exist_ok=True)

        output_file: str = Path(temp_dir / "qa_pairs_list.json").as_posix()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        return output_file

    except Exception as e:
        return f"❌ Error processing your question: {str(e)}"
