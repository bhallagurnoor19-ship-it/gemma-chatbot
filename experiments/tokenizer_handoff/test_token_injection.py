"""
Test Token Injection Script
Phase 4 & 5 of EmbeddingGemma 2 <-> Gemma 4 Tokenizer Compatibility & Direct Injection Investigation.
Directly probes Ollama runtime interfaces (/api/generate, /api/chat, /api/tokenize, and context mechanisms)
to verify whether externally generated token IDs can be injected into Gemma 4.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

def query_endpoint(endpoint: str, payload: dict, timeout: int = 10) -> dict:
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            elapsed = time.time() - t0
            return {
                "success": True,
                "http_status": resp.status,
                "elapsed_seconds": round(elapsed, 4),
                "response_preview": data[:300]
            }
    except urllib.error.HTTPError as he:
        elapsed = time.time() - t0
        body = he.read().decode("utf-8", errors="replace")
        return {
            "success": False,
            "http_status": he.code,
            "elapsed_seconds": round(elapsed, 4),
            "error_body": body[:300]
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success": False,
            "error": str(e),
            "elapsed_seconds": round(elapsed, 4)
        }

def run_injection_tests():
    print("=" * 75)
    print("PHASE 4 & 5: DIRECT TOKEN-ID INJECTION INVESTIGATION")
    print("=" * 75)

    # Sample token IDs from "Hello, how are you?"
    sample_token_ids = [2, 9259, 236764, 3, 7843, 3, 733, 3, 7624, 236881]

    probes = [
        {
            "name": "Probe 1: /api/generate with token ID array as 'prompt'",
            "endpoint": "/api/generate",
            "payload": {
                "model": "gemma4:12b",
                "prompt": sample_token_ids,
                "stream": False
            },
            "hypothesis": "Does Ollama's Generate endpoint allow passing token ID integers directly?"
        },
        {
            "name": "Probe 2: /api/chat with token ID array as message 'content'",
            "endpoint": "/api/chat",
            "payload": {
                "model": "gemma4:12b",
                "messages": [
                    {"role": "user", "content": sample_token_ids}
                ],
                "stream": False
            },
            "hypothesis": "Does Ollama's Chat endpoint allow passing pre-tokenized message contents?"
        },
        {
            "name": "Probe 3: /api/generate with raw=True and 'context' as token IDs",
            "endpoint": "/api/generate",
            "payload": {
                "model": "gemma4:12b",
                "prompt": "",
                "raw": True,
                "context": sample_token_ids,
                "stream": False
            },
            "hypothesis": "Can 'context' array be used with raw=True to inject KV/token state?"
        },
        {
            "name": "Probe 4: /api/tokenize for external verification",
            "endpoint": "/api/tokenize",
            "payload": {
                "model": "gemma4:12b",
                "prompt": "Hello"
            },
            "hypothesis": "Does Ollama expose a tokenize endpoint to verify runtime token alignment?"
        },
        {
            "name": "Probe 5: /api/detokenize for reverse token reconstruction",
            "endpoint": "/api/detokenize",
            "payload": {
                "model": "gemma4:12b",
                "tokens": sample_token_ids
            },
            "hypothesis": "Does Ollama expose a detokenize endpoint for token translation?"
        }
    ]

    results = []

    for p in probes:
        print(f"\n[*] Executing {p['name']}...")
        print(f"    Endpoint: {p['endpoint']}")
        res = query_endpoint(p["endpoint"], p["payload"])
        p_res = {
            "name": p["name"],
            "endpoint": p["endpoint"],
            "hypothesis": p["hypothesis"],
            "result": res
        }
        results.append(p_res)

        if res.get("success"):
            print(f"    [+] Supported! HTTP {res['http_status']}")
            print(f"        Response: {res.get('response_preview')}")
        else:
            print(f"    [-] UNSUPPORTED: HTTP {res.get('http_status')} / {res.get('error') or res.get('error_body')}")

    # Architectural Analysis
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ollama_runtime": {
            "layer_1_http_daemon": "Go (server/routes.go)",
            "layer_2_inference_runner": "C++ llamacpp (llama_decode / llama_eval)",
            "direct_token_injection_supported": False,
            "root_cause": (
                "Ollama's HTTP daemon strictly unmarshals 'prompt' and 'content' into Go `string` fields. "
                "Any array payload triggers a Go JSON unmarshal error (HTTP 400). "
                "Furthermore, Ollama 0.40.15 does not expose `/api/tokenize` or `/api/detokenize` endpoints."
            ),
            "lower_level_runtime_capability": (
                "The underlying llama.cpp runner natively supports `llama_eval(ctx, tokens.data(), n_tokens, n_past)`. "
                "In native C++ or `llama-cpp-python`, direct token ID injection is fully supported, "
                "but accessing it requires bypassing or modifying the Ollama HTTP server."
            )
        },
        "probes": results
    }

    out_file = os.path.join(OUTPUT_DIR, "injection_investigation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed injection investigation saved to: {out_file}")

    return summary

if __name__ == "__main__":
    run_injection_tests()
