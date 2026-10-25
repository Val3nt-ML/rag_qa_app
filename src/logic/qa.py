import json
import re
from pathlib import Path

import numpy as np
from llama_index.core import PromptTemplate
from llama_index.core.schema import BaseNode, ImageNode, TextNode

from config import settings
from retrieval.rag_helper import RAGHelper

QUESTION_PROMPT = PromptTemplate(
    """<role>Tu es un ingénieur spécialisé dans le bâtiment.
</role>

<task>
Génère {num_questions} questions techniques en français à partir du CONTEXTE.
</task>

<constraints>
- Chaque question doit être répondable uniquement à partir de ce contexte.
- Ne pose pas de question sur les métadonnées (pages, nom ou chemin du document).
- Retourne uniquement une liste numérotée de questions, une question par ligne.
</constraints>

<context>
{context_str}
</context>
"""
)


def generate_question_answer_pairs(
    num_question: int,
    rag_helper: RAGHelper | None = None,
) -> str:
    """Export French question/answer pairs and their source contexts as JSON."""
    if rag_helper is None or rag_helper._index is None:
        return "❌ Error: No index has been built yet. Please build an index first."

    questions_per_chunk = settings.rag.nb_questions_per_chunk
    if num_question < 1 or questions_per_chunk < 1:
        return "❌ The number of questions and questions per chunk must be positive."

    try:
        text_nodes = [
            node
            for node in rag_helper._nodes
            if isinstance(node, TextNode)
            and not isinstance(node, ImageNode)
            and node.text.strip()
        ]
        final_nodes: list[BaseNode] = []
        if text_nodes:
            threshold = np.percentile(
                [len(node.text) for node in text_nodes],
                settings.qa_generator.percentile_threshold,
            )
            final_nodes.extend(
                node for node in text_nodes if len(node.text) >= threshold
            )
        final_nodes.extend(
            node
            for node in rag_helper._nodes
            if isinstance(node, ImageNode) and node.get_content().strip()
        )

        if not final_nodes:
            return "❌ No text content is available to generate questions."
        if num_question > len(final_nodes) * questions_per_chunk:
            return "❌ Nb questions too High ! Reduce either the number of questions to generate or increase the number of questions per chunk."

        # Use the public LLM API instead of the removed llama_dataset module.
        # Generate per node so each exported answer retains its exact source context.
        qa_pairs = []
        for node in final_nodes:
            remaining = num_question - len(qa_pairs)
            if remaining == 0:
                break
            context = node.get_content()
            generated = rag_helper.llm.predict(
                QUESTION_PROMPT,
                num_questions=min(questions_per_chunk, remaining),
                context_str=context,
            )
            questions = [
                re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
                for line in generated.splitlines()
                if line.strip()
            ]
            for question in questions[: min(questions_per_chunk, remaining)]:
                answer = rag_helper.llm.predict(
                    rag_helper.text_qa_template,
                    context_str=context,
                    query_str=question,
                )
                qa_pairs.append(
                    {
                        "question": question,
                        "answer": answer,
                        "context": [context],
                    }
                )

        if len(qa_pairs) != num_question:
            return "❌ The model generated fewer questions than requested. Please try again."

        temp_dir = Path("temp_qa")
        temp_dir.mkdir(exist_ok=True)
        output_file = temp_dir / "qa_pairs_list.json"
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(qa_pairs, file, indent=2, ensure_ascii=False)
        return output_file.as_posix()

    except Exception as e:  # noqa: BLE001 - Surface provider errors in the Gradio UI.
        return f"❌ Error processing your question: {e!s}"
