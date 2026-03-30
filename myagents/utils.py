""" 
Utility functions for the myagents package.
"""


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
