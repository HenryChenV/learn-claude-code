""" 
Utility functions for the myagents package.
"""


from anthropic.types import Usage
import tiktoken


def truncate(output: str, max_length: int = 50000) -> str:
    if len(output) > max_length:
        return f"{output[:max_length]}..."
    else:
        return output


def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


def estimate_tokens(model: str, text: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # en ~4chars/token, zh ~1-2char/token)
        return len(text) // 3 + 1


def estimate_next_context(
    model: str,
    prev_usage: Usage,
    new_inputs: list[str]) -> int:

    cache_tokens = prev_usage.cache_read_input_tokens or 0
    input_tokens = prev_usage.input_tokens
    output_tokens = prev_usage.output_tokens

    new_tokens = sum([estimate_tokens(model, i) for i in new_inputs])

    return sum([cache_tokens, input_tokens, output_tokens, new_tokens])