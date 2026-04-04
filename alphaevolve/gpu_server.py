#!/usr/bin/env python3
"""GPU-accelerated LLM inference server using llama-cpp-python.

Runs on the user's RTX 2070 machine and serves an Ollama-compatible API.
Uses create_chat_completion for proper chat template handling.
"""

import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from llama_cpp import Llama

# Model path (Ollama's downloaded GGUF file)
MODEL_PATH = "/home/raad/.ollama/models/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"

print("Loading model with GPU offload...")
t0 = time.time()
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,  # Offload ALL layers to GPU
    n_ctx=4096,
    n_batch=512,
    verbose=True,
    chat_format="chatml",  # Qwen2.5 uses ChatML format
)
print(f"Model loaded in {time.time()-t0:.1f}s")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))

            prompt = body.get("prompt", "")
            options = body.get("options", {})
            max_tokens = options.get("num_predict", body.get("num_predict", 1500))
            temperature = options.get("temperature", body.get("temperature", 0.7))
            repeat_penalty = options.get("repeat_penalty", 1.1)

            # Split prompt into system + user if it contains both
            # The LLMEnsemble sends "system\n\nuser" format
            system_msg = "You are a helpful coding assistant."
            user_msg = prompt
            if "\n\n" in prompt:
                parts = prompt.split("\n\n", 1)
                if len(parts[0]) > 50:  # Looks like a system prompt
                    system_msg = parts[0]
                    user_msg = parts[1]

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            t0 = time.time()
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                repeat_penalty=repeat_penalty,
            )
            t1 = time.time()

            text = output["choices"][0]["message"]["content"]
            eval_count = output["usage"]["completion_tokens"]
            eval_duration = int((t1 - t0) * 1e9)

            response = {
                "response": text,
                "eval_count": eval_count,
                "eval_duration": eval_duration,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        elif self.path == "/api/tags":
            response = {
                "models": [
                    {
                        "name": "qwen2.5-coder:7b-gpu",
                        "model": "qwen2.5-coder:7b-gpu",
                        "details": {"parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
                    }
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/api/tags":
            self.do_POST()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"GPU LLM Server running")

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {format % args}")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11435
    print(f"Starting GPU LLM server on port {port}...")
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()
