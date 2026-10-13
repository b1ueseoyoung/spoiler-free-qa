import pytest

from sfqa.config import check_loopback, load_settings


@pytest.mark.parametrize("url", ["http://127.0.0.1:1234/v1", "http://localhost:1234/v1", "http://[::1]:1234/v1"])
def test_loopback_allowed(url):
    check_loopback(url)


@pytest.mark.parametrize(
    "url",
    ["http://192.168.0.10:1234/v1", "https://api.openai.com/v1", "http://127.0.0.1.example.com/v1", "ftp://127.0.0.1/"],
)
def test_non_loopback_rejected(url):
    with pytest.raises(ValueError):
        check_loopback(url)


def test_settings_reject_remote_base_url():
    with pytest.raises(ValueError):
        load_settings({"LLM_BASE_URL": "https://api.openai.com/v1"})


def test_settings_defaults_are_local():
    s = load_settings({})
    assert s.base_url.startswith("http://127.0.0.1")
    assert s.reasoning_effort == "none"
    assert s.cassette_mode == "off"


def test_cassette_mode_needs_path():
    with pytest.raises(ValueError):
        load_settings({"CASSETTE_MODE": "replay"})
    with pytest.raises(ValueError):
        load_settings({"CASSETTE_MODE": "tape", "CASSETTE_PATH": "x.jsonl"})
