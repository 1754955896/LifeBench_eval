"""
Configuration loading utilities.

Supports YAML configuration file loading with environment variable substitution.
"""
import http.client
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        file_path: YAML file path

    Returns:
        Parsed configuration dictionary
    """
    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = _replace_env_vars(config)
    return config


def _replace_env_vars(obj: Any) -> Any:
    """
    Recursively replace environment variables in configuration.

    Supported format: ${VAR_NAME} or ${VAR_NAME:default_value}
    """
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        pattern = r"\$\{([^:}]+)(?::([^}]+))?\}"

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replacer, obj)
    else:
        return obj


def save_yaml(config: Dict[str, Any], file_path: str):
    """Save configuration to YAML file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def normalize_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize system configuration fields that are expected to be numeric.
    """
    from copy import deepcopy

    normalized = deepcopy(config)
    normalizers = {
        ("llm", "temperature"): float,
        ("llm", "max_tokens"): int,
        ("answer", "temperature"): float,
        ("answer", "max_retries"): int,
    }

    for path, coercer in normalizers.items():
        _normalize_config_scalar(normalized, path, coercer)

    return normalized


def _normalize_config_scalar(
    config: Dict[str, Any], path: tuple, coercer: type
) -> None:
    current = config
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
        if current is None:
            return

    if not isinstance(current, dict):
        return

    key = path[-1]
    if key not in current:
        return

    value = current[key]
    if value is None:
        return

    if coercer is int and isinstance(value, bool):
        raise ValueError(f"{'.'.join(path)} must be an integer, got boolean")

    if coercer is int and isinstance(value, int):
        return

    if coercer is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        current[key] = float(value)
        return

    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return
        try:
            current[key] = coercer(stripped)
        except ValueError:
            pass


def get_deepseek_balance(env_path: str = ".env") -> Dict[str, Any]:
    """
    Query DeepSeek API balance.

    Args:
        env_path: Path to .env file containing LLM_API_KEY

    Returns:
        Balance information as dict
    """
    env_file = Path(env_path)
    if not env_file.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    api_key = None
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LLM_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

    if not api_key:
        raise ValueError("LLM_API_KEY not found in .env file")

    conn = http.client.HTTPSConnection("api.deepseek.com")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    conn.request("GET", "/user/balance", "", headers)
    res = conn.getresponse()
    data = res.read()
    conn.close()

    return json.loads(data.decode("utf-8"))
