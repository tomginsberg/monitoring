from typing import Any


def clean_config(config: dict[Any, Any], decimals: int = 3) -> dict[Any, Any]:
    # convert any float atoms to strings
    for k, v in config.items():
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, float):
            config[k] = round(v, decimals)
        elif isinstance(v, tuple):
            config[k] = list(v)
        elif isinstance(v, dict):
            config[k] = clean_config(v)
    return config
