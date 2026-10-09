"""The chat model used by the agent.

``llm`` is built from the config on import (no network call until it is
used); ``build_llm()`` builds others, e.g. for the eval script. Embeddings
and the vector store live in ``retrieval``, on the MCP server side.
"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from .config import (
    CHAT_BACKEND,
    CHAT_MODEL,
    LLAMA_SERVER_URL,
    MAX_TOKENS,
    OLLAMA_CHAT_MODEL,
    SAMPLING,
    THINKING,
)


class ChatLlamaServer(ChatOpenAI):
    """ChatOpenAI that keeps llama-server's ``reasoning_content``.

    llama-server returns Qwen's thinking in a separate ``reasoning_content``
    field, which ChatOpenAI drops. Qwen3.8 is trained to see its earlier
    reasoning again when it continues after a tool call, so this subclass
    stores the field in ``additional_kwargs`` (streamed pieces concatenate
    when chunks merge) and sends it back with each assistant message.
    """

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for gen, choice in zip(result.generations, data.get("choices") or []):
            reasoning = (choice.get("message") or {}).get("reasoning_content")
            if reasoning:
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info,
    ):
        gen = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info,
        )
        choices = chunk.get("choices") or []
        reasoning = choices and (choices[0].get("delta") or {}).get("reasoning_content")
        if gen is not None and reasoning:
            gen.message.additional_kwargs["reasoning_content"] = reasoning
        return gen

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for msg, out in zip(messages, payload.get("messages") or []):
            if isinstance(msg, AIMessage):
                reasoning = msg.additional_kwargs.get("reasoning_content")
                if reasoning:
                    out["reasoning_content"] = reasoning
        return payload


def build_llm(
    backend: str | None = None,
    model: str | None = None,
    thinking: bool | None = None,
):
    """Construct the chat model. Arguments default to the config values.

    Args:
        backend: ``"llama_server"`` or ``"ollama"``.
        model: Model name for that backend.
        thinking: Whether the model reasons before answering.

    Returns:
        A LangChain chat model instance (ChatOllama or ChatOpenAI).
    """
    backend = backend or CHAT_BACKEND
    thinking = THINKING if thinking is None else thinking

    if backend == "ollama":
        # Simple local path. Uses a model from `ollama list`. Less
        # reliable than Qwen3.8-27B, but needs no llama.cpp setup — good
        # for confirming the app works end to end.
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or OLLAMA_CHAT_MODEL,
            temperature=0.1,
            reasoning=None if thinking else False,
        )

    if backend == "llama_server":
        # Qwen3.8-27B served by llama.cpp's llama-server (see
        # SETUP_NOTES.md) on its OpenAI-compatible endpoint. Thinking
        # arrives in reasoning_content, which ChatLlamaServer passes back
        # between tool calls; the message content stays clean.
        sampling = SAMPLING[thinking]
        return ChatLlamaServer(
            model=model or CHAT_MODEL,
            base_url=LLAMA_SERVER_URL,
            api_key="not-needed",      # llama-server ignores this
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            presence_penalty=sampling["presence_penalty"],
            max_tokens=MAX_TOKENS[thinking],
            extra_body={
                "top_k": sampling["top_k"],
                "min_p": sampling["min_p"],
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )

    raise ValueError(
        f"Unknown backend {backend!r} — use 'ollama' or 'llama_server'."
    )


llm = build_llm()