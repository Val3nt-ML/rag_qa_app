import json
from pathlib import Path
import re

from llama_index.core import PromptTemplate
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.schema import ImageNode, TextNode
import numpy as np

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
        nl_lst = []
        for i in range(len(rag_helper._nodes)):
            if isinstance(rag_helper._nodes[i], TextNode):
                nl_lst.append(len(rag_helper._nodes[i].text))

        # Filter out nodes that don't have "enough" text, that may correspond to only metadata information
        txt_filtered_nodes: list[TextNode] = [
            n
            for n in [m for m in rag_helper._nodes if isinstance(m, TextNode)]
            if len(n.text)
            >= np.percentile(nl_lst, settings.qa_generator.percentile_threshold)
        ]

        final_nodes: list[TextNode | ImageNode] = txt_filtered_nodes.copy()
        final_nodes.extend([n for n in rag_helper._nodes if isinstance(n, ImageNode)])

        if num_question > len(final_nodes) * settings.rag.nb_questions_per_chunk:
            return "❌ Nb questions too High ! Reduce either the number of questions to generate or increase the number of questions per chunk."

        dataset_generator = RagDatasetGenerator(
            nodes=final_nodes[:num_question],
            llm=rag_helper.llm,  # Explicitly providing the LLM here
            num_questions_per_chunk=settings.rag.nb_questions_per_chunk,
            question_gen_query="You are an engineer specialized in building domain. Your role is to generate {num_questions_per_chunk} technical questions about information provided in the **CONTEXT** section below.",
            text_question_template=PromptTemplate(
                """
### TASK ###
{query_str}

### CONTEXT ###
{context_str}

### INSTRUCTIONS ###
- Restrict the questions to the **CONTEXT** section information provided
- Each question **must be in French**.
- Each question must be answerable using **only** the information given in the text.
- **Don't generate** question related to metadata information like a number of pages, document name,  document file path, size or author.

### EXAMPLES ###
Here are some examples of good questions:
- "Quelles sont les exigences minimum requises pour les dispositifs d’isolation?"
- "Fournit un récapitulatif des dispositifs réglementaires applicables dans le document (de préférence sous forme de matrice) ?"
- "A quel phase d’un projet d’isolation doit être remis le rapport d’étanchéité de l’air ? "
- "Quels sont les différents livrables du management de la qualité en phase DCE?"
""",
            ),
        )

        qa_set = dataset_generator.generate_dataset_from_nodes()
        qa_pairs = [
            {
                "question": r.query,
                "answer": r.reference_answer,
                "context": r.reference_contexts,
            }
            for r in qa_set.examples
        ]

        temp_dir: Path = Path("temp_qa")
        temp_dir.mkdir(exist_ok=True)

        output_file: str = Path(temp_dir / "qa_pairs_list.json").as_posix()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        return output_file

    except Exception as e:
        return f"❌ Error processing your question: {str(e)}"
