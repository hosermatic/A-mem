"""Test that A-MEM routes LLM calls through LiteLLM."""
import os
import pytest


def test_amem_config_uses_litellm_base_url():
    """Verify A-MEM config points to LiteLLM, not directly to provider."""
    config_paths = [
        "config.yaml", "config.yml", ".env", "src/config.py",
        "amem/config.py", "settings.py"
    ]
    amem_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    found_config = False
    for cp in config_paths:
        full_path = os.path.join(amem_dir, cp)
        if os.path.exists(full_path):
            content = open(full_path).read()
            found_config = True
            # Should NOT have direct provider URLs
            assert "api.openai.com" not in content, \
                f"A-MEM config {cp} calls OpenAI directly — should route through LiteLLM"
            assert "api.anthropic.com" not in content, \
                f"A-MEM config {cp} calls Anthropic directly — should route through LiteLLM"

    if not found_config:
        pytest.skip("No config file found to verify")


def test_litellm_base_url_env():
    """Verify OPENAI_API_BASE or equivalent points to LiteLLM."""
    base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", ""))
    if base_url:
        assert "localhost:4000" in base_url or "litellm" in base_url.lower(), \
            f"OPENAI_API_BASE should point to LiteLLM (localhost:4000), got: {base_url}"


def test_http_service_main_defaults_to_litellm():
    """Verify http_service/main.py defaults LLM_BACKEND to litellm."""
    amem_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_path = os.path.join(amem_dir, "http_service", "main.py")

    assert os.path.exists(main_path), f"main.py not found at {main_path}"
    content = open(main_path).read()

    assert '"openai"' not in content.split("LLM_BACKEND")[1].split("\n")[0] \
        if "LLM_BACKEND" in content else True, \
        "http_service/main.py still defaults LLM_BACKEND to 'openai'"

    assert 'litellm' in content, \
        "http_service/main.py should reference litellm as the default backend"


def test_env_template_defaults_to_litellm():
    """Verify .env.template sets LLM_BACKEND=litellm."""
    amem_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(amem_dir, ".env.template")

    if not os.path.exists(template_path):
        pytest.skip(".env.template not found")

    content = open(template_path).read()
    assert "LLM_BACKEND=litellm" in content, \
        f".env.template should set LLM_BACKEND=litellm"
    assert "LITELLM_GATEWAY_URL=http://localhost:4000" in content, \
        ".env.template should include LITELLM_GATEWAY_URL"


def test_http_service_env_example_uses_litellm():
    """Verify http_service/.env.example defaults to litellm."""
    amem_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(amem_dir, "http_service", ".env.example")

    if not os.path.exists(env_path):
        pytest.skip("http_service/.env.example not found")

    content = open(env_path).read()
    assert "LLM_BACKEND=litellm" in content, \
        "http_service/.env.example should set LLM_BACKEND=litellm"
