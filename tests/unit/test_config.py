from __future__ import annotations

from pathlib import Path

import pytest

from askmydocs.config import AppConfig, load_config
from askmydocs.errors import ConfigError
from conftest import TEST_CONFIG_FILE


def test_loads_yaml_values() -> None:
    config = load_config(TEST_CONFIG_FILE)
    assert config.chunking.chunk_tokens == 60
    assert config.retrieval.rrf_k == 60
    assert config.generation.model == "llama-3.3-70b-versatile"


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKMYDOCS_CHUNKING__CHUNK_TOKENS", "123")
    config = load_config(TEST_CONFIG_FILE)
    assert config.chunking.chunk_tokens == 123


def test_groq_key_read_from_plain_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert load_config(TEST_CONFIG_FILE).groq_api_key == "gsk-test"


def test_relative_paths_are_resolved_absolute() -> None:
    config = load_config(TEST_CONFIG_FILE)
    assert config.paths.chunks_file.is_absolute()


def test_missing_config_file_raises() -> None:
    with pytest.raises(ConfigError):
        load_config(Path("config/does-not-exist.yaml"))


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKMYDOCS_CHUNKING__CHUNK_OVERLAP_TOKENS", "9999")
    with pytest.raises(ConfigError):
        load_config(TEST_CONFIG_FILE)


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKMYDOCS_LOGGING__LEVEL", "LOUD")
    with pytest.raises(ConfigError):
        load_config(TEST_CONFIG_FILE)


def test_config_fixture_is_isolated(config: AppConfig, tmp_path: Path) -> None:
    assert tmp_path in config.paths.chunks_file.parents
