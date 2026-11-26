import tomllib
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models.toml"
KINDS = {"openai", "openai-embeddings", "anthropic"}
EMBED_USES = {"embed_semantic", "embed_style"}


def test_models_toml_parses_and_judges_name_endpoints():
    cfg = tomllib.loads(MODELS.read_text())
    endpoints = cfg["endpoints"]
    for name, ep in endpoints.items():
        assert ep["kind"] in KINDS, name
        assert ep["model"], name
        assert ep["kind"] == "anthropic" or ep["base_url"], name
    for profile, uses in cfg["judges"].items():
        for use, endpoint in uses.items():
            assert endpoint in endpoints, f"judges.{profile}.{use} -> unknown endpoint {endpoint!r}"
            is_embed = endpoints[endpoint]["kind"] == "openai-embeddings"
            assert is_embed == (use in EMBED_USES), f"judges.{profile}.{use} -> {endpoint!r}"
    for name, cast in cfg["cast"].items():
        assert set(cast.values()) <= set(endpoints) and not {"subject", "scripted"} & set(cast), f"cast.{name}"
    for order in ("web", "papers"):
        for provider in cfg["search"][order]:
            assert provider in cfg["search"]["rates"], f"search.{order}: {provider!r} has no rate"
