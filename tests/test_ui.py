import importlib
from unittest.mock import Mock

import pytest


@pytest.fixture
def buttons(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    return importlib.import_module("logic.button")


def test_new_document_invalidates_previous_index(buttons):
    button, helper, status = buttons.update_button_state("Ready", "new.pdf", Mock())
    assert helper is None
    assert button.interactive is True
    assert "prepare" in status.lower()


def test_missing_document_never_starts_indexing(buttons, monkeypatch):
    build = Mock()
    monkeypatch.setattr(buttons, "build_index_from_file", build)
    updates = list(buttons.building_button_handler(None))
    build.assert_not_called()
    assert updates[-1][1] is None
    assert updates[-1][0].interactive is False


def test_failed_build_allows_retry(buttons, monkeypatch):
    monkeypatch.setattr(
        buttons, "build_index_from_file", Mock(side_effect=RuntimeError("offline"))
    )
    updates = list(buttons.building_button_handler("document.pdf"))
    assert updates[-1][0].interactive is True
    assert updates[-1][1] is None
    assert "failed" in updates[-1][2]


def test_successful_build_enables_document_workspace(buttons, monkeypatch):
    helper = Mock()
    monkeypatch.setattr(
        buttons, "build_index_from_file", Mock(return_value=(helper, "OK"))
    )
    updates = list(buttons.building_button_handler("document.pdf"))
    assert updates[0][0].interactive is False
    assert updates[-1][1] is helper
    assert "ready" in updates[-1][2]


def test_workspace_controls_require_a_prepared_document(buttons):
    interface = importlib.import_module("app")
    assert all(
        control.interactive is False for control in interface.index_controls(None)
    )
    assert all(
        control.interactive is True for control in interface.index_controls(Mock())
    )


def test_failed_export_shows_message_instead_of_invalid_download(buttons, monkeypatch):
    interface = importlib.import_module("app")
    monkeypatch.setattr(
        interface,
        "generate_question_answer_pairs",
        Mock(return_value="❌ Not enough text"),
    )
    download, message = interface.export_questions(5, Mock())
    assert download is None
    assert "Generation stopped" in message
