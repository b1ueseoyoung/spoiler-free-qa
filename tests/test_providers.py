"""공급자 전환(docs/deploy-plan.md 1단계): 기본은 로컬, 설정으로 Claude API와 프로세스 안 임베딩을 고른다."""

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from sfqa import claude, config
from sfqa import index as index_mod
from sfqa.claude import AnthropicClient
from sfqa.config import load_settings
from sfqa.llm import LiveClient, LocalEmbeddings, ReplayClient, live_client, make_client, make_embeddings
from tests.test_retrieve import TEXTS


def test_defaults_stay_local():
    s = load_settings({})
    assert (s.llm_provider, s.embed_provider) == ("lmstudio", "lmstudio")
    assert s.llm_model == "qwen/qwen3.6-35b-a3b" and s.embed_model == "text-embedding-kure-v1"


def test_anthropic_defaults_match_claude_client():
    s = load_settings({"LLM_PROVIDER": "anthropic"})
    assert s.llm_model == config.ANTHROPIC_DEFAULT_MODEL
    assert s.reasoning_effort == claude.effort_label(s.llm_model)  # 이 값이 다르면 AnthropicClient가 거부한다


def test_local_embed_default_model_and_override():
    assert load_settings({"EMBED_PROVIDER": "local"}).embed_model == "mykor/KURE-v1-gguf/KURE-v1-Q8_0.gguf"
    assert load_settings({"EMBED_PROVIDER": "local", "EMBED_MODEL": "x/y"}).embed_model == "x/y"


@pytest.mark.parametrize("env", [{"LLM_PROVIDER": "openai"}, {"EMBED_PROVIDER": "voyage"}])
def test_unknown_provider_rejected(env):
    with pytest.raises(ValueError):
        load_settings(env)


def test_loopback_enforced_while_any_lm_studio_use():
    remote = {"LLM_BASE_URL": "https://example.com/v1"}
    with pytest.raises(ValueError):  # 임베딩이 아직 LM Studio면 주소 검사를 한다
        load_settings({**remote, "LLM_PROVIDER": "anthropic"})
    s = load_settings({**remote, "LLM_PROVIDER": "anthropic", "EMBED_PROVIDER": "local"})
    assert s.base_url == "https://example.com/v1"  # LM Studio를 안 쓰면 이 주소는 쓰이지 않는다


def test_clients_follow_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # 키 값만 있으면 된다(호출하지 않는다)
    s = load_settings({"LLM_PROVIDER": "anthropic", "CASSETTE_MODE": "record", "CASSETTE_PATH": str(tmp_path / "c.jsonl")})
    c = make_client(s)
    assert isinstance(c, AnthropicClient) and c.cassette == tmp_path / "c.jsonl"
    replay = load_settings({"LLM_PROVIDER": "anthropic", "CASSETTE_MODE": "replay", "CASSETTE_PATH": str(tmp_path / "c.jsonl")})
    assert isinstance(make_client(replay), ReplayClient)
    assert isinstance(live_client(load_settings({})), LiveClient)


def test_anthropic_client_needs_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(claude, "load_key", lambda env_file=None: False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        live_client(load_settings({"LLM_PROVIDER": "anthropic"}))


class FakeLlama:
    """llama_cpp.Llama 대역: pooling_type()과 embed()만."""

    def __init__(self, pooling=2):
        self.pooling = pooling

    def pooling_type(self):
        return self.pooling

    def embed(self, text):
        return [float(len(text)), 1.0]


def test_local_embeddings_use_gguf_model_and_check_cls_pooling():
    emb = LocalEmbeddings("mykor/KURE-v1-gguf/KURE-v1-Q8_0.gguf", model=FakeLlama())
    assert emb.embed_documents(["가나", "다"]) == [[2.0, 1.0], [1.0, 1.0]]
    assert emb.embed_query("가나다") == [3.0, 1.0]
    with pytest.raises(ValueError, match="pooling이 mean"):
        LocalEmbeddings("x", model=FakeLlama(pooling=1)).embed_query("가")


def test_resolve_gguf_prefers_models_dir(tmp_path, monkeypatch):
    from sfqa.llm import resolve_gguf

    f = tmp_path / "mykor" / "KURE-v1-gguf" / "KURE-v1-Q8_0.gguf"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"gguf")
    monkeypatch.setenv("GGUF_MODELS_DIR", str(tmp_path))
    assert resolve_gguf("mykor/KURE-v1-gguf/KURE-v1-Q8_0.gguf") == f


def test_make_embeddings_local_is_lazy():
    emb = make_embeddings(load_settings({"EMBED_PROVIDER": "local"}))
    assert isinstance(emb, LocalEmbeddings) and emb._model is None  # 처음 쓸 때 올린다


def test_index_dirs_are_separate_per_runtime(tmp_path):
    from sfqa.works import Work

    work = Work(id="w", title="w", author="a", label="w", chapters=19, dir=tmp_path / "w")
    emb = DeterministicFakeEmbedding(size=8)
    name = "mykor/KURE-v1-gguf/KURE-v1-Q8_0.gguf"
    meta = index_mod.build_index(work, name, emb, TEXTS, provider="local")
    assert meta["embed_provider"] == "local"
    assert index_mod.index_dir(work, name, "local").name == "local__mykor__KURE-v1-gguf__KURE-v1-Q8_0.gguf"
    index_mod.load_index(work, name, TEXTS, provider="local")
    with pytest.raises(index_mod.IndexMismatch):  # LM Studio 쪽에서는 이 색인을 보지 않는다
        index_mod.load_index(work, name, TEXTS)
    assert index_mod.index_dir(work, "text-embedding-kure-v1").name == "text-embedding-kure-v1"


def test_retrieve_k_setting_and_gate_k():
    assert load_settings({}).retrieve_k == 18 and load_settings({"RETRIEVE_K": "20"}).retrieve_k == 20
