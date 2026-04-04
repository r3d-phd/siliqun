"""
LLM Ensemble for AlphaEvolve -- Ollama-Only Edition.

Uses the local Ollama GPU server on the user's RTX 2070 as the sole
LLM backend.  Zero rate limits, zero external API dependencies.

DEHB-learned parameters (temperature, top_p, num_predict, repeat_penalty)
are applied as defaults and can be overridden per-call.

Auto-reconnects the SSH tunnel if the connection drops.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("aedb.llm_ensemble")

# ──────────────────────────────────────────────────────────────────────
# Ollama configuration
# ──────────────────────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11435")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-gpu")

# SSH tunnel parameters for auto-reconnect
SSH_HOST = os.environ.get("GPU_SSH_HOST", "6.tcp.eu.ngrok.io")
SSH_PORT = os.environ.get("GPU_SSH_PORT", "11654")
SSH_USER = os.environ.get("GPU_SSH_USER", "raad")
SSH_PASS = os.environ.get("GPU_SSH_PASS", "r3d@29e")
GPU_REMOTE_PORT = 11435


# ──────────────────────────────────────────────────────────────────────
# SSH tunnel management
# ──────────────────────────────────────────────────────────────────────

def _check_tunnel_alive() -> bool:
    """Check if the SSH tunnel to the GPU server is alive."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _reconnect_tunnel() -> bool:
    """Attempt to re-establish the SSH tunnel to the GPU server."""
    logger.info("Attempting SSH tunnel reconnect...")
    subprocess.run(["pkill", "-f", f"ssh.*{GPU_REMOTE_PORT}"], capture_output=True)
    time.sleep(1)

    try:
        cmd = [
            "sshpass", "-p", SSH_PASS,
            "ssh",
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", SSH_PORT,
            "-L", f"{GPU_REMOTE_PORT}:localhost:{GPU_REMOTE_PORT}",
            "-N",
            f"{SSH_USER}@{SSH_HOST}",
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)

        if _check_tunnel_alive():
            logger.info("SSH tunnel reconnected successfully")
            return True
        else:
            logger.warning("SSH tunnel reconnect failed")
            return False
    except Exception as e:
        logger.warning(f"SSH tunnel reconnect error: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────
# LLM Ensemble (Ollama-only, DEHB-parameterised)
# ──────────────────────────────────────────────────────────────────────

class LLMEnsemble:
    """Ollama-only LLM backend for AlphaEvolve.

    Uses the local RTX 2070 GPU via SSH tunnel.
    Zero rate limits, ~38 tok/s (7B) or ~20 tok/s (14B).

    DEHB-learned parameters are stored as defaults and applied
    to every LLM call unless overridden.

    Parameters
    ----------
    flash_ratio : float
        Ignored (single-model ensemble).
    seed : int
        Random seed.
    """

    def __init__(self, flash_ratio: float = 0.7, seed: int = 42):
        self.total_calls = 0
        self.total_tokens = 0
        self.failures = 0
        self.consecutive_failures = 0
        self._last_reconnect = 0.0

        # DEHB-learned defaults (updated by orchestrator after DEHB runs)
        self.default_temperature: float = 0.8
        self.default_top_p: float = 0.95
        self.default_num_predict: int = 800
        self.default_repeat_penalty: float = 1.1

        logger.info(f"LLM Ensemble: Ollama-only ({OLLAMA_MODEL} @ {OLLAMA_URL})")

    def _ensure_tunnel(self):
        """Ensure the SSH tunnel is alive, reconnect if needed."""
        now = time.time()
        if now - self._last_reconnect < 30:
            return
        if not _check_tunnel_alive():
            self._last_reconnect = now
            _reconnect_tunnel()

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1500,
        prefer_tier: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ) -> Tuple[Optional[str], str]:
        """Call Ollama and return the response.

        Uses DEHB-learned defaults unless overridden by explicit params.

        Parameters
        ----------
        system_prompt : str
            System-level instructions.
        user_prompt : str
            User-level prompt with code context.
        max_tokens : int
            Maximum tokens to generate.
        prefer_tier : str, optional
            Ignored (single model).
        temperature : float, optional
            Override DEHB-learned temperature.
        top_p : float, optional
            Override DEHB-learned top_p.
        repeat_penalty : float, optional
            Override DEHB-learned repeat_penalty.

        Returns
        -------
        response : str or None
        model_name : str
        """
        self.total_calls += 1

        # Apply DEHB-learned defaults with optional overrides
        temp = temperature if temperature is not None else self.default_temperature
        tp = top_p if top_p is not None else self.default_top_p
        rp = repeat_penalty if repeat_penalty is not None else self.default_repeat_penalty
        np_ = max_tokens if max_tokens > 0 else self.default_num_predict

        # Try up to 3 times with reconnect
        for attempt in range(3):
            try:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                resp = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": temp,
                            "top_p": tp,
                            "num_predict": np_,
                            "repeat_penalty": rp,
                        },
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    if text:
                        eval_count = data.get("eval_count", 0)
                        eval_dur = data.get("eval_duration", 1) / 1e9
                        tok_per_s = eval_count / max(eval_dur, 0.001)
                        self.total_tokens += eval_count
                        self.consecutive_failures = 0
                        logger.debug(
                            f"Ollama: {eval_count} tok in {eval_dur:.1f}s "
                            f"({tok_per_s:.1f} tok/s) "
                            f"[temp={temp:.2f}, top_p={tp:.2f}]"
                        )
                        return text, "ollama-gpu"
                # Empty response
                logger.warning(f"Ollama empty response (attempt {attempt+1}/3)")

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout) as e:
                logger.warning(
                    f"Ollama connection error (attempt {attempt+1}/3): {e}"
                )

            # Reconnect and retry
            self.failures += 1
            self.consecutive_failures += 1
            if attempt < 2:
                logger.info("Reconnecting SSH tunnel...")
                self._ensure_tunnel()
                time.sleep(2)

        logger.warning("Ollama failed after 3 attempts")
        return None, "none"

    def get_stats(self) -> Dict:
        """Return usage statistics including DEHB-learned params."""
        return {
            "ollama-gpu": {
                "tier": "flash",
                "calls": self.total_calls,
                "tokens": self.total_tokens,
                "failures": self.failures,
                "consecutive_failures": self.consecutive_failures,
                "dehb_params": {
                    "temperature": self.default_temperature,
                    "top_p": self.default_top_p,
                    "num_predict": self.default_num_predict,
                    "repeat_penalty": self.default_repeat_penalty,
                },
            }
        }
