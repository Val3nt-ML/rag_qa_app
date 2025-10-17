from pathlib import Path

from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings
import yaml


class Rag(BaseModel):
    nb_questions_per_chunk: int
    similarity_top_k: int
    llm_model: str
    llm_model_context_window: int
    embed_model: str


class Azure(BaseModel):
    api_version: str
    endpoint: str
    api_key: str


class Settings(BaseSettings):
    azure: Azure
    rag: Rag


def load_settings(path: Path = Path(Path(__file__).resolve().parent / "config.yaml")) -> Settings:
    """Loads settings from a YAML file and validates them using Pydantic."""
    try:
        with open(path, "r") as file:
            # 1. Load the raw YAML data
            yaml_data = yaml.safe_load(file)

            # 2. Validate the data with the Pydantic model
            return Settings.model_validate(yaml_data)

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at '{path}'")
    except ValidationError as e:
        # Pydantic raises a helpful error if the data is invalid
        raise ValueError(f"Configuration error: {e}")


# Create a single, importable instance of the settings for your app
# This code runs when the module is imported
settings = load_settings()
