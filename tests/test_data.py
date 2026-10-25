import importlib

import pytest


@pytest.fixture
def config(monkeypatch):
    # Never use real credentials when validating the configuration.
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "test-token")
    return importlib.import_module("config")


def test_settings_load_token_from_environment(config):
    settings = config.load_settings()

    assert settings.huggingface.token == "test-token"
    assert settings.rag.similarity_top_k > 0


def test_environment_overrides_dotenv(config, monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("HUGGINGFACE_TOKEN=dotenv-token\n")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "environment-token")

    settings = config.HuggingFace(_env_file=dotenv)

    assert settings.token == "environment-token"


def test_missing_configuration_reports_path(config, tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        config.load_settings(missing_path)


def test_invalid_configuration_is_rejected(config, tmp_path):
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("rag: {}\n")

    with pytest.raises(ValueError, match="Configuration error"):
        config.load_settings(invalid_path)
