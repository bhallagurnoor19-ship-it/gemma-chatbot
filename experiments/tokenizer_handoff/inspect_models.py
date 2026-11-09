"""
Inspect Models & Environment Discovery Script
Phase 1 of EmbeddingGemma 2 <-> Gemma 4 Tokenizer Compatibility & Direct Injection Investigation.
Gathers runtime versions, model tags, manifests, architecture details, and RAG configuration.
"""

import json
import os
import struct
import sys
import urllib.request

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
BLOBS_DIR = os.path.expanduser(r"~\.ollama\models\blobs")

MODELS_TO_INSPECT = [
    "embeddinggemma-2:270m",
    "embeddinggemma:latest",
    "gemma4:12b",
]

def query_ollama(endpoint, data=None):
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def inspect_gguf_header(filepath):
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return {"error": f"Not GGUF format: {magic}"}
        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        file_size = os.path.getsize(filepath)
        return {
            "magic": magic.decode("latin-1"),
            "gguf_version": version,
            "tensor_count": tensor_count,
            "kv_count": kv_count,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }

def inspect_environment():
    report = {}

    # 1. Ollama Version
    try:
        ver_data = query_ollama("/api/version")
        report["ollama_version"] = ver_data.get("version", "unknown")
    except Exception as e:
        report["ollama_version"] = f"Error: {e}"

    # 2. Installed Model Tags
    try:
        tags_data = query_ollama("/api/tags")
        report["installed_models"] = [
            {
                "name": m.get("name"),
                "size_mb": round(m.get("size", 0) / (1024 * 1024), 2),
                "modified_at": m.get("modified_at"),
                "digest": m.get("digest")
            }
            for m in tags_data.get("models", [])
        ]
    except Exception as e:
        report["installed_models"] = f"Error: {e}"

    # 3. Model Deep Metadata from /api/show
    models_meta = {}
    for model_name in MODELS_TO_INSPECT:
        try:
            show_data = query_ollama("/api/show", {"model": model_name})
            details = show_data.get("details", {})
            model_info = show_data.get("model_info", {})
            modelfile = show_data.get("modelfile", "")

            # Extract blob references from Modelfile
            blobs = []
            for line in modelfile.splitlines():
                line = line.strip()
                if line.startswith("FROM ") or line.startswith("DRAFT "):
                    parts = line.split(maxsplit=1)
                    if len(parts) > 1 and parts[1].startswith("C:\\"):
                        blobs.append(parts[1])

            # Filter relevant tokenizer and architecture keys
            tok_info = {}
            for k, v in model_info.items():
                if any(x in k for x in ["tokenizer", "vocab", "arch", "family", "context", "embedding"]):
                    tok_info[k] = v

            # Inspect GGUF header for main blob
            main_blob_header = {}
            if blobs:
                main_blob_header = inspect_gguf_header(blobs[0])

            models_meta[model_name] = {
                "format": details.get("format"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
                "template": show_data.get("template"),
                "parameters": show_data.get("parameters"),
                "blobs": blobs,
                "main_blob_header": main_blob_header,
                "filtered_model_info": tok_info
            }
        except Exception as e:
            models_meta[model_name] = {"error": str(e)}

    report["models_metadata"] = models_meta

    # 4. Check Tokenize Endpoint Support
    tok_endpoint_support = {}
    for model_name in MODELS_TO_INSPECT:
        try:
            tok_res = query_ollama("/api/tokenize", {"model": model_name, "prompt": "test"})
            tok_endpoint_support[model_name] = {"supported": True, "res": tok_res}
        except urllib.error.HTTPError as he:
            tok_endpoint_support[model_name] = {"supported": False, "status": he.code, "msg": str(he)}
        except Exception as e:
            tok_endpoint_support[model_name] = {"supported": False, "msg": str(e)}
    report["tokenize_endpoint_support"] = tok_endpoint_support

    return report

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 1: ENVIRONMENT & MODEL DISCOVERY")
    print("=" * 70)
    rep = inspect_environment()
    print(f"\nOllama Version: {rep.get('ollama_version')}")
    print(f"\nTarget Models Identified:")
    for m, d in rep.get("models_metadata", {}).items():
        if "error" in d:
            print(f"  - {m}: ERROR ({d['error']})")
        else:
            print(f"  - {m}:")
            print(f"      Family: {d.get('family')}")
            print(f"      Param Size: {d.get('parameter_size')}")
            print(f"      Quant: {d.get('quantization_level')}")
            print(f"      Main Blob Size: {d.get('main_blob_header', {}).get('file_size_mb')} MB")
            print(f"      Tokenizer ggml.model: {d.get('filtered_model_info', {}).get('tokenizer.ggml.model')}")
            print(f"      Tokenizer bos_token_id: {d.get('filtered_model_info', {}).get('tokenizer.ggml.bos_token_id')}")
            print(f"      Tokenizer eos_token_id: {d.get('filtered_model_info', {}).get('tokenizer.ggml.eos_token_id')}")

    print(f"\nOllama /api/tokenize Support:")
    for m, s in rep.get("tokenize_endpoint_support", {}).items():
        print(f"  - {m}: {s}")

    out_json = os.path.join(os.path.dirname(__file__), "discovery_findings.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nDetailed discovery saved to: {out_json}")
