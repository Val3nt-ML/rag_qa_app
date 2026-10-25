import importlib
import json
from unittest.mock import Mock

import pytest
from llama_index.core.schema import Document, TextNode


@pytest.fixture
def qa(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    return importlib.import_module("logic.qa")


def test_qa_module_imports_with_current_llamaindex(qa):
    assert callable(qa.generate_question_answer_pairs)


def test_export_preserves_source_context_and_requested_count(qa, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qa.settings.rag, "nb_questions_per_chunk", 2)
    helper = Mock()
    helper._index = object()
    helper._nodes = [
        TextNode(text="Le bâtiment utilise une isolation en laine de bois.")
    ]
    helper.llm.predict.side_effect = [
        "1. Quel isolant est utilisé ?\n2. Quel matériau compose l’isolant ?",
        "La laine de bois.",
    ]

    output = qa.generate_question_answer_pairs(1, helper)

    with open(output, encoding="utf-8") as file:
        pairs = json.load(file)
    assert pairs == [
        {
            "question": "Quel isolant est utilisé ?",
            "answer": "La laine de bois.",
            "context": [helper._nodes[0].text],
        }
    ]


def test_empty_index_does_not_call_llm(qa):
    helper = Mock()
    helper._index = object()
    helper._nodes = []

    result = qa.generate_question_answer_pairs(1, helper)

    assert result.startswith("❌")
    helper.llm.predict.assert_not_called()


def test_excessive_question_count_does_not_call_llm(qa, monkeypatch):
    monkeypatch.setattr(qa.settings.rag, "nb_questions_per_chunk", 1)
    helper = Mock()
    helper._index = object()
    helper._nodes = [TextNode(text="Une source documentaire.")]

    result = qa.generate_question_answer_pairs(2, helper)

    assert result.startswith("❌")
    helper.llm.predict.assert_not_called()


def test_missing_tokenizer_reports_configuration_error(qa, monkeypatch):
    retrieval = importlib.import_module("retrieval.rag_helper")
    monkeypatch.setattr(retrieval.AutoTokenizer, "from_pretrained", lambda *_: None)
    helper = qa.RAGHelper.__new__(qa.RAGHelper)

    with pytest.raises(ValueError, match="tokenizer"):
        helper.compute_text_nodes([Document(text="Source")])
