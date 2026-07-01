from openai import OpenAI
from environment.config.config import config


def get_client(model_prefix=None):
    """Return the unified DashScope OpenAI-compatible client.

    VideoAgent's benchmark reproduction uses one DashScope endpoint/key for all
    OpenAI-compatible LLM and VLM calls. ``model_prefix`` is accepted for
    backwards-compatible call sites, but credential selection is intentionally
    unified through ``dashscope_api_key`` and ``dashscope_base_url``.
    """
    llm_config = config.get("llm") or {}
    api_key = llm_config.get("dashscope_api_key")
    base_url = llm_config.get("dashscope_base_url")
    if not api_key or not base_url:
        raise RuntimeError(
            "Missing DashScope config: set llm.dashscope_api_key and "
            "llm.dashscope_base_url in environment/config/config.yml"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_messages(system=None, user=None, messages=None):
    if messages is not None:
        return messages
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    if user is not None:
        messages.append({"role": "user", "content": user})
    return messages


def deepseek(model="qwen3.7-max", system=None, user=None, messages=None):
    client = get_client("deepseek")
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(system=system, user=user, messages=messages),
        temperature=1,
        response_format={"type": "json_object"},
    )
    return response


def claude(model="qwen3.7-max", system=None, user=None, messages=None):
    client = get_client("claude")
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(system=system, user=user, messages=messages),
        temperature=1,
    )
    return response


def gemini(model="qwen3.7-plus", system=None, user=None, messages=None):
    client = get_client("gemini")
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(system=system, user=user, messages=messages),
        temperature=1,
    )
    return response


def gpt(model="qwen3.7-max", system=None, user=None, messages=None):
    client = get_client("gpt")
    response = client.chat.completions.create(
        model=model,
        messages=_build_messages(system=system, user=user, messages=messages),
        temperature=1,
    )
    return response
