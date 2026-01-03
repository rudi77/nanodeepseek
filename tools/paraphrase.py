"""Paraphrase client using Azure OpenAI if configured; deterministic fallback."""

from __future__ import annotations

import json
import os
import random
import urllib.request
from typing import Dict


def _azure_chat_json(prompt: str, temperature: float = 0.7) -> Dict[str, str]:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    if not endpoint or not api_key or not deployment:
        raise RuntimeError("Azure OpenAI env vars missing")

    url = (
        f"{endpoint.rstrip('/')}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={api_version}"
    )
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You output JSON only with a single key 'instruction'. "
                    "Do not include any other text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return json.loads(content)


def paraphrase_instruction(prompt: str, temperature: float = 0.7) -> str:
    try:
        result = _azure_chat_json(prompt, temperature=temperature)
        instruction = result.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
    except Exception:
        pass

    # Deterministic fallback: simple templating with small variation.
    templates = [
        "Erstelle den Buchungssatz: {task}",
        "Bilden Sie die Er?ffnungsbuchung: {task}",
        "Bitte buche folgenden Sachverhalt: {task}",
    ]
    return random.choice(templates).format(task=prompt)
