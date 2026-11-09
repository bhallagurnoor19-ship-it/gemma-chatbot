"""
Extract Tokenizers Script
Phase 2 of EmbeddingGemma 2 <-> Gemma 4 Tokenizer Compatibility & Direct Injection Investigation.
Directly extracts tokenizer vocabularies, scores, types, and special token configurations
from raw GGUF blobs for embeddinggemma-2:270m, gemma4:12b, and embeddinggemma:latest.
"""

import os
import struct
import sys
import time
import json

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLOBS_DIR = os.path.expanduser(r"~\.ollama\models\blobs")

MODEL_BLOBS = {
    "embeddinggemma-2:270m": os.path.join(BLOBS_DIR, "sha256-111e84992a0406bacb70d11dddd8874001c783cc56b82c462ec82a2f5b8d08b1"),
    "gemma4:12b": os.path.join(BLOBS_DIR, "sha256-bb722270d54346adc198f213851dbe0207e87c3ecf2a0aff4d92262726215391"),
    "embeddinggemma:latest": os.path.join(BLOBS_DIR, "sha256-0800cbac9c2064dde519420e75e512a83cb360de3ad5df176185dc69652fc515"),
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_gguf_tokenizer(filepath, extract_full_tokens=True):
    """
    Parses GGUF header and extracts all tokenizer-related key-value pairs.
    Skips weight tensors entirely for maximum performance.
    """
    t0 = time.time()
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Blob file does not exist: {filepath}")

    tokenizer_data = {}

    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic in {filepath}: {magic}")

        version, tensor_count, kv_count = struct.unpack("<IQQ", f.read(20))
        tokenizer_data["_gguf_version"] = version
        tokenizer_data["_tensor_count"] = tensor_count
        tokenizer_data["_kv_count"] = kv_count

        def read_str():
            length, = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8", errors="replace")

        def skip_val(vtype):
            if vtype in (0, 1, 7): # uint8, int8, bool
                f.seek(1, 1)
            elif vtype in (2, 3): # uint16, int16
                f.seek(2, 1)
            elif vtype in (4, 5, 6): # uint32, int32, float32
                f.seek(4, 1)
            elif vtype in (10, 11, 12): # uint64, int64, float64
                f.seek(8, 1)
            elif vtype == 8: # string
                sl, = struct.unpack("<Q", f.read(8))
                f.seek(sl, 1)
            elif vtype == 9: # array
                item_type, arr_len = struct.unpack("<IQ", f.read(12))
                if item_type in (0, 1, 7):
                    f.seek(arr_len, 1)
                elif item_type in (2, 3):
                    f.seek(arr_len * 2, 1)
                elif item_type in (4, 5, 6):
                    f.seek(arr_len * 4, 1)
                elif item_type in (10, 11, 12):
                    f.seek(arr_len * 8, 1)
                elif item_type == 8:
                    for _ in range(arr_len):
                        sl, = struct.unpack("<Q", f.read(8))
                        f.seek(sl, 1)
                else:
                    raise ValueError(f"Unsupported array item type {item_type}")
            else:
                raise ValueError(f"Unknown GGUF type {vtype}")

        for _ in range(kv_count):
            k_len, = struct.unpack("<Q", f.read(8))
            key = f.read(k_len).decode("utf-8", errors="replace")
            vtype, = struct.unpack("<I", f.read(4))

            # Only parse keys of interest
            if key == "tokenizer.ggml.tokens" and extract_full_tokens:
                item_type, arr_len = struct.unpack("<IQ", f.read(12))
                tokens = []
                for _ in range(arr_len):
                    sl, = struct.unpack("<Q", f.read(8))
                    tokens.append(f.read(sl).decode("utf-8", errors="replace"))
                tokenizer_data["tokens"] = tokens
            elif key == "tokenizer.ggml.scores":
                item_type, arr_len = struct.unpack("<IQ", f.read(12))
                scores = struct.unpack(f"<{arr_len}f", f.read(arr_len * 4))
                tokenizer_data["scores"] = list(scores)
            elif key == "tokenizer.ggml.token_type":
                item_type, arr_len = struct.unpack("<IQ", f.read(12))
                token_types = struct.unpack(f"<{arr_len}i", f.read(arr_len * 4))
                tokenizer_data["token_types"] = list(token_types)
            elif key == "tokenizer.ggml.merges":
                item_type, arr_len = struct.unpack("<IQ", f.read(12))
                merges = []
                for _ in range(arr_len):
                    sl, = struct.unpack("<Q", f.read(8))
                    merges.append(f.read(sl).decode("utf-8", errors="replace"))
                tokenizer_data["merges"] = merges
            elif key in [
                "tokenizer.ggml.model",
                "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id",
                "tokenizer.ggml.padding_token_id",
                "tokenizer.ggml.unknown_token_id",
                "tokenizer.ggml.mask_token_id",
                "tokenizer.ggml.add_bos_token",
                "tokenizer.ggml.add_eos_token",
                "tokenizer.ggml.add_space_prefix",
                "tokenizer.chat_template",
                "general.architecture",
                "general.name"
            ]:
                if vtype == 8: # string
                    sl, = struct.unpack("<Q", f.read(8))
                    tokenizer_data[key] = f.read(sl).decode("utf-8", errors="replace")
                elif vtype == 4: # uint32
                    tokenizer_data[key], = struct.unpack("<I", f.read(4))
                elif vtype == 5: # int32
                    tokenizer_data[key], = struct.unpack("<i", f.read(4))
                elif vtype == 7: # bool
                    tokenizer_data[key], = struct.unpack("<?", f.read(1))
                elif vtype == 10: # uint64
                    tokenizer_data[key], = struct.unpack("<Q", f.read(8))
                elif vtype == 11: # int64
                    tokenizer_data[key], = struct.unpack("<q", f.read(8))
                else:
                    skip_val(vtype)
            else:
                skip_val(vtype)

    elapsed = time.time() - t0
    tokenizer_data["_extract_time_seconds"] = round(elapsed, 3)
    return tokenizer_data

def main():
    print("=" * 70)
    print("PHASE 2: TOKENIZER EXTRACTION FROM GGUF BLOBS")
    print("=" * 70)

    extracted_tokenizers = {}

    for model_name, blob_path in MODEL_BLOBS.items():
        print(f"\n[+] Extracting tokenizer from {model_name}...")
        print(f"    Blob: {blob_path}")
        try:
            tok = parse_gguf_tokenizer(blob_path, extract_full_tokens=True)
            vocab_size = len(tok.get("tokens", []))
            scores_size = len(tok.get("scores", []))
            types_size = len(tok.get("token_types", []))
            merges_size = len(tok.get("merges", []))

            print(f"    Extracted in: {tok['_extract_time_seconds']}s")
            print(f"    Tokenizer Model: {tok.get('tokenizer.ggml.model')}")
            print(f"    Vocab Size: {vocab_size} tokens")
            print(f"    Scores Size: {scores_size}")
            print(f"    Types Size: {types_size}")
            print(f"    Merges Size: {merges_size}")
            print(f"    BOS Token ID: {tok.get('tokenizer.ggml.bos_token_id')}")
            print(f"    EOS Token ID: {tok.get('tokenizer.ggml.eos_token_id')}")
            print(f"    PAD Token ID: {tok.get('tokenizer.ggml.padding_token_id')}")
            print(f"    UNK Token ID: {tok.get('tokenizer.ggml.unknown_token_id')}")
            print(f"    MASK Token ID: {tok.get('tokenizer.ggml.mask_token_id')}")
            print(f"    Add BOS: {tok.get('tokenizer.ggml.add_bos_token')}")

            # Save summary info (without entire 256k array in summary JSON to keep it lightweight)
            summary = {k: v for k, v in tok.items() if k not in ("tokens", "scores", "token_types", "merges")}
            summary["vocab_size"] = vocab_size
            summary["scores_count"] = scores_size
            summary["token_types_count"] = types_size
            summary["merges_count"] = merges_size
            summary["first_10_tokens"] = [tok["tokens"][i] for i in range(min(10, vocab_size))]
            summary["last_10_tokens"] = [tok["tokens"][i] for i in range(max(0, vocab_size - 10), vocab_size)]
            
            extracted_tokenizers[model_name] = tok

            # Save individual model vocabulary data for Phase 3
            model_slug = model_name.replace(":", "_").replace("-", "_")
            model_vocab_file = os.path.join(OUTPUT_DIR, f"vocab_{model_slug}.json")
            with open(model_vocab_file, "w", encoding="utf-8") as vf:
                json.dump({
                    "model": model_name,
                    "metadata": summary,
                    "tokens": tok.get("tokens", []),
                    "scores": tok.get("scores", []),
                    "token_types": tok.get("token_types", []),
                    "merges": tok.get("merges", [])
                }, vf, ensure_ascii=False)
            print(f"    Saved full vocabulary artifact to: {model_vocab_file}")

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Save summary metadata
    summary_path = os.path.join(OUTPUT_DIR, "extracted_tokenizers_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({m: {k: v for k, v in t.items() if k not in ("tokens", "scores", "token_types", "merges")} 
                   for m, t in extracted_tokenizers.items()}, f, indent=2)
    print(f"\nExtraction summary written to: {summary_path}")

if __name__ == "__main__":
    main()
