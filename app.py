"""
Autonomous Multi-Model Chatbot & Coding Pipeline
- General Chat & Vision: Gemma 4 (gemma4:12b / gemma4)
- Architecture Design: DeepSeek-Coder-V2 16B (deepseek-coder-v2:16b)
- Code Generation: Qwen2.5-Coder 7B (Qwen2.5-Coder:7B)
- Aider Pair-Programming Integration
"""

import base64
import binascii
from enum import Enum
import json
import os
import re
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger("app")

# Force UTF-8 on Windows stdout/stderr to prevent charmap/cp1252 emoji crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import requests


class _HttpRequestsWrapper:
    """Wrapper around requests to allow test patching of app.http_requests without mutating global requests."""
    def __getattr__(self, name: str) -> Any:
        return getattr(requests, name)


http_requests = _HttpRequestsWrapper()
from dotenv import load_dotenv

from model_lifecycle import (
    get_model_manager,
    ModelLease,
    ModelLifecycleError,
    ModelLeaseTimeoutError,
    ModelLeaseCancelledError,
    ModelEvictionError,
    StateReconciliationError,
)
from rag.service import get_rag_service
from rag.memory import (
    get_conversation_store,
    ConversationMessage,
    ConversationMemoryStore,
)
from message_store import (
    get_message_store,
    MessageStore,
    MessageStatus,
    MessageRecord,
)

app = Flask(__name__)

# ── Environment & Settings ───────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(APP_DIR, ".env")
load_dotenv(ENV_FILE)

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_MODELS = [
    {"id": "gemini-flash-lite-latest", "name": "Gemini Flash-Lite (Default for Free Tier - Instant & High Quota)"},
    {"id": "gemini-flash-latest", "name": "Gemini Flash (Default Flash - High Intelligence)"},
    {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite (Fast & Lightweight)"},
    {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Latest Flash Generation)"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro (Deep Complex Reasoning - Requires Pro Quota)"}
]

def get_gemini_api_key() -> str:
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    return os.environ.get("GEMINI_API_KEY", "").strip()

def get_gemini_model() -> str:
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

def is_search_grounding_enabled() -> bool:
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    val = os.environ.get("GEMINI_SEARCH_GROUNDING", "false").strip().lower()
    return val in ("1", "true", "yes", "on")

def is_cloud_multimodal_enabled() -> bool:
    """Check if user has explicitly authorized uploading images to Gemini Cloud API (defaults to False)."""
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    val = os.environ.get("ALLOW_CLOUD_MULTIMODAL", "false").strip().lower()
    return val in ("1", "true", "yes", "on")

def save_env_config(
    key: str | None = None,
    model: str | None = None,
    grounding: bool | None = None,
    cloud_multimodal: bool | None = None,
):
    """Safely write configuration keys to project .env file."""
    lines = []
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            pass

    kv = {}
    for line in lines:
        line_s = line.strip()
        if line_s and not line_s.startswith("#") and "=" in line_s:
            k, v = line_s.split("=", 1)
            kv[k.strip()] = v.strip()

    if key is not None:
        kv["GEMINI_API_KEY"] = key.strip()
        os.environ["GEMINI_API_KEY"] = key.strip()
    if model is not None:
        kv["GEMINI_MODEL"] = model.strip()
        os.environ["GEMINI_MODEL"] = model.strip()
    if grounding is not None:
        kv["GEMINI_SEARCH_GROUNDING"] = "true" if grounding else "false"
        os.environ["GEMINI_SEARCH_GROUNDING"] = kv["GEMINI_SEARCH_GROUNDING"]
    if cloud_multimodal is not None:
        kv["ALLOW_CLOUD_MULTIMODAL"] = "true" if cloud_multimodal else "false"
        os.environ["ALLOW_CLOUD_MULTIMODAL"] = kv["ALLOW_CLOUD_MULTIMODAL"]

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        for k, v in kv.items():
            f.write(f"{k}={v}\n")

# ── Configuration & Models ───────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
# Ollama timeouts: (connect_timeout, read_timeout)
# connect_timeout (10s): fail fast if Ollama daemon is offline or unreachable
# read_timeout (360s): accommodate local cold model loading (~60-90s) plus CPU prompt eval (~150-220s)
OLLAMA_CONNECT_TIMEOUT = 10
OLLAMA_MODEL_READ_TIMEOUT = 360
OLLAMA_TIMEOUT = (OLLAMA_CONNECT_TIMEOUT, OLLAMA_MODEL_READ_TIMEOUT)
# Ollama options: expand num_ctx from default 2048 to 8192 to prevent premature truncation
# on large RAG context and detailed multi-section responses. num_predict set to 4096.
OLLAMA_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 4096,
}

CHAT_MODELS = ["gemma4:12b", "gemma4:latest", "gemma4", "gemma3:latest", "gemma:latest"]
ARCHITECT_MODELS = ["deepseek-coder-v2:16b", "deepseek-coder-v2:latest"]
CODER_MODELS = ["Qwen2.5-Coder:7B", "qwen:latest"]

# Set global max request body size to 45 MB to safely accommodate 25 MB raw binary
# images with base64 expansion (~33.3 MB base64) plus conversation history envelope
app.config["MAX_CONTENT_LENGTH"] = 45 * 1024 * 1024

# System Prompts
CHAT_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and knowledgeable AI assistant powered by Gemma 4. "
    "Provide clear, concise, and structured answers. When given images, analyze them thoroughly."
)

ARCHITECT_SYSTEM_PROMPT = (
    "You are DeepSeek Coder V2, an elite Principal Software Architect. "
    "Your objective is to analyze the user's coding request and formulate a comprehensive, "
    "bulletproof architectural design and technical specification. "
    "Structure your design as follows:\n"
    "1. 🎯 Overview & Tech Stack\n"
    "2. 🏗️ Architecture & Module Decomposition (File/Folder Structure)\n"
    "3. 🔑 Key Data Structures, APIs & Flow\n"
    "4. ⚠️ Edge Cases, Security & Performance Considerations\n"
    "5. 📋 Implementation Blueprint for the Coding Agent\n"
    "Focus purely on high-level design, architecture, and step-by-step logic. "
    "The downstream coding agent (Qwen2.5-Coder) will write the full code."
)

CODER_SYSTEM_PROMPT = (
    "You are Qwen2.5-Coder, a world-class Senior Software Engineer. "
    "You write pristine, complete, production-ready code based on the user's requirements "
    "and the Principal Architect's specification. "
    "Rules:\n"
    "- Write complete, functional, runnable code (no placeholders or 'TODOs').\n"
    "- Include informative docstrings, comments, and proper error handling.\n"
    "- Provide clear usage instructions or unit tests.\n"
    "- Follow the Architect's specification faithfully."
)


def get_available_ollama_models() -> list[str]:
    """Retrieve list of currently installed models in Ollama."""
    try:
        resp = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=4)
        if resp.status_code == 200:
            return [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        pass
    return []


def resolve_model(preference_list: list[str], available: list[str]) -> str:
    """Find the first matching installed model or fallback to first preference."""
    for candidate in preference_list:
        if candidate in available:
            return candidate
        # case-insensitive search
        for m in available:
            if candidate.lower() == m.lower():
                return m
    # prefix match (e.g. gemma4:* if gemma4 is requested)
    for candidate in preference_list:
        prefix = candidate.split(":")[0].lower()
        for m in available:
            if m.lower().startswith(prefix + ":") or m.lower() == prefix:
                return m
    return preference_list[0]


def get_aider_path() -> str | None:
    """Find local aider executable."""
    custom_path = r"C:\Users\Gurnoor\.local\bin\aider.EXE"
    if os.path.exists(custom_path):
        return custom_path
    return shutil.which("aider")


# ── Model Capability Registry & Dynamic Verification ────────────────────────
class ModelModality(str, Enum):
    TEXT = "text"
    CODE = "code"
    VISION = "vision"


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    GEMINI = "gemini"


class ModelCapability:
    def __init__(
        self,
        name: str,
        provider: ModelProvider,
        modalities: Set[ModelModality],
        max_context: int,
        verified_vision: bool = False,
    ):
        self.name = name
        self.provider = provider
        self.modalities = modalities
        self.max_context = max_context
        self.verified_vision = verified_vision

    @property
    def supports_vision(self) -> bool:
        return ModelModality.VISION in self.modalities and self.verified_vision


# Authoritative static registry: Any model not listed here defaults to TEXT only (fail closed)
STATIC_CAPABILITY_REGISTRY: Dict[str, ModelCapability] = {
    # Gemini Cloud Models (Multimodal vision & code)
    "gemini-flash-lite-latest": ModelCapability("gemini-flash-lite-latest", ModelProvider.GEMINI, {ModelModality.TEXT, ModelModality.CODE, ModelModality.VISION}, 1048576, verified_vision=True),
    "gemini-flash-latest": ModelCapability("gemini-flash-latest", ModelProvider.GEMINI, {ModelModality.TEXT, ModelModality.CODE, ModelModality.VISION}, 1048576, verified_vision=True),
    "gemini-3.8-flash": ModelCapability("gemini-3.8-flash", ModelProvider.GEMINI, {ModelModality.TEXT, ModelModality.CODE, ModelModality.VISION}, 1048576, verified_vision=True),
    "gemini-3.5-flash-lite": ModelCapability("gemini-3.5-flash-lite", ModelProvider.GEMINI, {ModelModality.TEXT, ModelModality.CODE, ModelModality.VISION}, 1048576, verified_vision=True),
    "gemini-3.1-pro-preview": ModelCapability("gemini-3.1-pro-preview", ModelProvider.GEMINI, {ModelModality.TEXT, ModelModality.CODE, ModelModality.VISION}, 2097152, verified_vision=True),

    # Verified Local Vision Models (When installed in Ollama)
    "llama3.2-vision:11b": ModelCapability("llama3.2-vision:11b", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 8192, verified_vision=True),
    "llama3.2-vision:latest": ModelCapability("llama3.2-vision:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 8192, verified_vision=True),
    "llama3.2-vision:90b": ModelCapability("llama3.2-vision:90b", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 8192, verified_vision=True),
    "llava:7b": ModelCapability("llava:7b", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 4096, verified_vision=True),
    "llava:13b": ModelCapability("llava:13b", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 4096, verified_vision=True),
    "llava:latest": ModelCapability("llava:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 4096, verified_vision=True),
    "moondream:latest": ModelCapability("moondream:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 2048, verified_vision=True),
    "minicpm-v:latest": ModelCapability("minicpm-v:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.VISION}, 4096, verified_vision=True),

    # Verified Local Text/Code Models (Strictly Non-Vision)
    "deepseek-coder-v2:16b": ModelCapability("deepseek-coder-v2:16b", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.CODE}, 16384, verified_vision=False),
    "deepseek-coder-v2:latest": ModelCapability("deepseek-coder-v2:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.CODE}, 16384, verified_vision=False),
    "Qwen2.5-Coder:7B": ModelCapability("Qwen2.5-Coder:7B", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.CODE}, 32768, verified_vision=False),
    "qwen:latest": ModelCapability("qwen:latest", ModelProvider.OLLAMA, {ModelModality.TEXT, ModelModality.CODE}, 32768, verified_vision=False),
    "gemma4:12b": ModelCapability("gemma4:12b", ModelProvider.OLLAMA, {ModelModality.TEXT}, 8192, verified_vision=False),
    "gemma3:latest": ModelCapability("gemma3:latest", ModelProvider.OLLAMA, {ModelModality.TEXT}, 8192, verified_vision=False),
    "embeddinggemma:latest": ModelCapability("embeddinggemma:latest", ModelProvider.OLLAMA, {ModelModality.TEXT}, 2048, verified_vision=False),
}

KNOWN_VISION_FAMILIES = {"mllama", "clip", "vision", "moondream", "llava"}


def get_extra_vision_models() -> set[str]:
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
    val = os.environ.get("EXTRA_VISION_MODELS", "").strip()
    if not val:
        return set()
    return {m.strip().lower() for m in val.split(",") if m.strip()}


def verify_ollama_model_vision(base_url: str, model_name: str) -> bool:
    """
    Two-stage runtime verification against Ollama POST /api/show.
    Fails closed on connection error or missing vision components.
    """
    try:
        resp = http_requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=3.0,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        model_info = data.get("model_info", {})
        arch = str(model_info.get("general.architecture", "")).lower()
        if any(v in arch for v in KNOWN_VISION_FAMILIES):
            return True
        details = data.get("details", {})
        families = [str(f).lower() for f in details.get("families", [])]
        if any(v in families for v in KNOWN_VISION_FAMILIES):
            return True
        params = str(data.get("parameters", "")).lower()
        if "projector" in params or "clip" in params:
            return True
        return False
    except Exception:
        return False


def is_model_vision_capable(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Determines whether a model is vision capable (fail closed).
    Returns True only if:
    1. Present in STATIC_CAPABILITY_REGISTRY with verified_vision=True, OR
    2. Present in EXTRA_VISION_MODELS and verified via verify_ollama_model_vision().
    """
    if not model_name:
        return False
    norm_name = model_name.strip()
    if norm_name in STATIC_CAPABILITY_REGISTRY:
        return STATIC_CAPABILITY_REGISTRY[norm_name].supports_vision
    # Case-insensitive match in registry
    for reg_name, cap in STATIC_CAPABILITY_REGISTRY.items():
        if reg_name.lower() == norm_name.lower():
            return cap.supports_vision

    # Check extra registered vision models with two-stage verification
    extra_vision = get_extra_vision_models()
    if norm_name.lower() in extra_vision or any(norm_name.lower().startswith(x) for x in extra_vision):
        return verify_ollama_model_vision(base_url, norm_name)

    # Fail closed for all unknown models
    return False


def resolve_vision_model(available_models: list[str]) -> Optional[str]:
    """Find the best verified local vision model available in Ollama."""
    PREFERRED_VISION_MODELS = [
        "llama3.2-vision:11b",
        "llama3.2-vision:latest",
        "llama3.2-vision:90b",
        "llava:13b",
        "llava:7b",
        "llava:latest",
        "moondream:latest",
        "minicpm-v:latest",
    ]
    for preferred in PREFERRED_VISION_MODELS:
        for m in available_models:
            if m.lower() == preferred.lower() or m.lower().startswith(preferred.lower()):
                if is_model_vision_capable(m):
                    return m
    for extra in get_extra_vision_models():
        for m in available_models:
            if m.lower() == extra.lower() or m.lower().startswith(extra.lower()):
                if is_model_vision_capable(m):
                    return m
    return None


# ── Image Quota & Binary Sniffing ─────────────────────────────────────────────
MAX_IMAGES_PER_PROMPT = 10
MAX_IMAGE_RAW_BYTES = 10 * 1024 * 1024        # 10 MB raw binary
MAX_TOTAL_IMAGE_RAW_BYTES = 25 * 1024 * 1024  # 25 MB raw binary aggregate

MAGIC_SIGNATURES = [
    (b"\xFF\xD8\xFF", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def sniff_image_mime(header_bytes: bytes) -> Optional[str]:
    """Detect image MIME type from raw magic bytes (JPEG, PNG, WebP, GIF)."""
    for magic, mime in MAGIC_SIGNATURES:
        if header_bytes.startswith(magic):
            return mime
    # WebP: RIFF....WEBP
    if header_bytes.startswith(b"RIFF") and len(header_bytes) >= 12 and header_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_and_decode_images(raw_images_b64: list[str]) -> Tuple[list[dict], Optional[str]]:
    """
    Validates base64 strings, decodes raw bytes, checks magic bytes, and enforces quotas.
    Returns: (list_of_validated_image_dicts, error_message_if_any)
    Each validated dict contains:
    {"raw_bytes": bytes, "b64": str, "mime_type": str, "size": int}
    """
    if len(raw_images_b64) > MAX_IMAGES_PER_PROMPT:
        return [], f"Image limit exceeded: received {len(raw_images_b64)} images (maximum allowed is {MAX_IMAGES_PER_PROMPT})."

    validated = []
    total_raw_bytes = 0

    for idx, b64_str in enumerate(raw_images_b64):
        if not isinstance(b64_str, str) or not b64_str.strip():
            continue
        clean_b64 = b64_str.strip()
        if "," in clean_b64 and clean_b64.startswith("data:"):
            clean_b64 = clean_b64.split(",", 1)[1]

        try:
            raw_bytes = base64.b64decode(clean_b64, validate=True)
        except (binascii.Error, ValueError):
            return [], f"Image #{idx + 1} contains corrupted or invalid base64 encoding."

        img_len = len(raw_bytes)
        if img_len > MAX_IMAGE_RAW_BYTES:
            size_mb = round(img_len / (1024 * 1024), 2)
            return [], f"Image #{idx + 1} exceeds 10 MB limit ({size_mb} MB)."

        total_raw_bytes += img_len
        if total_raw_bytes > MAX_TOTAL_IMAGE_RAW_BYTES:
            total_mb = round(total_raw_bytes / (1024 * 1024), 2)
            return [], f"Total image payload exceeds 25 MB aggregate limit ({total_mb} MB)."

        detected_mime = sniff_image_mime(raw_bytes[:16])
        if not detected_mime:
            return [], f"Image #{idx + 1} has an unsupported format or corrupted header. Allowed: JPEG, PNG, WebP, GIF."

        validated.append({
            "raw_bytes": raw_bytes,
            "b64": clean_b64,
            "mime_type": detected_mime,
            "size": img_len,
        })

    return validated, None


def extract_attachment_summary(messages: list[dict]) -> dict:
    """Inspects conversation history for attached image modalities."""
    total_images = []
    latest_has_images = False
    if messages:
        latest_msg = messages[-1]
        if isinstance(latest_msg, dict):
            latest_imgs = latest_msg.get("images", [])
            if isinstance(latest_imgs, list) and len(latest_imgs) > 0:
                latest_has_images = True
    for m in messages:
        if isinstance(m, dict):
            imgs = m.get("images", [])
            if isinstance(imgs, list):
                for img in imgs:
                    if isinstance(img, str) and img.strip():
                        total_images.append(img.strip())

    return {
        "has_images": len(total_images) > 0,
        "image_count": len(total_images),
        "raw_images_b64": total_images,
        "latest_has_images": latest_has_images,
    }


# ── Autonomous Intent Classifier ─────────────────────────────────────────────
CODING_PATTERNS = re.compile(
    r"\b("
    r"code|python|javascript|typescript|java|c\+\+|c#|rust|golang|go|html|css|sql|bash|powershell|"
    r"function|class|def|async|await|return|import|export|algorithm|api|endpoint|route|rest|"
    r"bug|debug|error|exception|traceback|fix|refactor|optimize|test|unit\s*test|pytest|"
    r"docker|dockerfile|kubernetes|database|schema|table|query|react|vue|angular|svelte|"
    r"flask|django|fastapi|express|spring|git|repo|repository|script|cli|package"
    r")\b|"
    r"(write\s+a\s+|create\s+a\s+|build\s+a\s+|develop\s+a\s+|implement\s+|how\s+to\s+code|"
    r"how\s+to\s+build|design\s+a\s+database|make\s+a\s+program|solve\s+this\s+leetcode)",
    re.IGNORECASE,
)

CURRENT_INFO_PATTERNS = re.compile(
    r"\b("
    r"today|yesterday|tomorrow|tonight|current|currently|latest|recent|recently|now|right now|"
    r"news|breaking news|headline|headlines|price of|stock|crypto|bitcoin|weather|forecast|"
    r"who won|match score|election|live score|release date|upcoming|trending|events in|what happened|"
    r"2024|2025|2026|this week|this month|this year"
    r")\b|"
    r"(search\s+for|look\s+up|browse|find\s+out\s+about|google\s+|who\s+is\s+the\s+current|what\s+is\s+the\s+current)",
    re.IGNORECASE,
)

DIFFICULT_REASONING_PATTERNS = re.compile(
    r"\b("
    r"prove|proof|theorem|lemma|derive|derivative|integral|differential|calculus|"
    r"step-by-step\s+proof|mathematical\s+induction|complex\s+analysis|matrix\s+eigen|"
    r"puzzle|riddle|syllogism|logical\s+fallacy|paradox|game\s+theory|quantum\s+mechanics|"
    r"advanced\s+physics|thermodynamics|deep\s+reasoning|counterfactual|philosophical\s+argument"
    r")\b|"
    r"(think\s+step\s+by\s+step|solve\s+this\s+complex|analyze\s+in\s+depth|rigorous\s+proof)",
    re.IGNORECASE,
)


ARCHITECTURAL_ASSESSMENT_PATTERNS = re.compile(
    r"\b("
    r"architectural\s+assessment|architecture\s+assessment|architectural\s+review|"
    r"architecture\s+review|system\s+design\s+review|architectural\s+evaluation|"
    r"architecture\s+evaluation|architectural\s+audit|system\s+architecture\s+assessment|"
    r"evaluate\s+(?:the\s+)?architecture|assess\s+(?:the\s+)?architecture|"
    r"review\s+(?:the\s+)?architecture|audit\s+(?:the\s+)?architecture|"
    r"analyze\s+(?:the\s+)?architecture|architectural\s+analysis|"
    r"architecture\s+trade-?offs?|architectural\s+trade-?offs?"
    r")\b",
    re.IGNORECASE,
)

CODE_IMPLEMENTATION_INTENT_PATTERNS = re.compile(
    r"\b("
    r"write\s+(?:the\s+|a\s+|some\s+)?code|implement\s+(?:the\s+|a\s+)?(?:code|function|class|method|module|service|api)|"
    r"generate\s+(?:the\s+|a\s+)?code|code\s+(?:this|the|a)|create\s+(?:the\s+|a\s+)?(?:script|file|program|implementation)|"
    r"build\s+(?:the\s+|a\s+)?(?:script|app|program|implementation)|write\s+a\s+program"
    r")\b",
    re.IGNORECASE,
)


def classify_intent(
    messages: list[dict],
    explicit_mode: str = "auto",
    has_images: bool = False,
    local_vision_model: Optional[str] = None,
) -> str:
    """
    Autonomously classifies user prompt according to the architecture:
    - Normal / local lightweight -> Local Gemma 4 ('chat')
    - Local Vision Model -> Local Ollama Vision ('local_vision')
    - Current / difficult / real-time / cloud multimodal -> Gemini API ('gemini')
    - Architectural assessment / analysis (without code generation request) -> DeepSeek Architect ('deepseek')
    - Coding -> DeepSeek Architect + Qwen Coder ('pipeline')
    - Explicit modes: 'gemini', 'chat', 'pipeline', 'deepseek', 'qwen'
    """
    if has_images:
        # Explicit Mode Checks (Option B: Strict Rejection)
        if explicit_mode in ("pipeline", "deepseek", "qwen"):
            return f"incompatible_mode:{explicit_mode}"

        if explicit_mode in ("chat", "gemma"):
            if local_vision_model and is_model_vision_capable(local_vision_model):
                return "local_vision"
            return "incompatible_mode:chat"

        if explicit_mode == "gemini":
            if not is_cloud_multimodal_enabled():
                return "cloud_multimodal_disabled"
            return "gemini"

        # Auto Mode Checks (Local-First Priority)
        if explicit_mode == "auto":
            # Priority 1: Verified local vision model
            if local_vision_model and is_model_vision_capable(local_vision_model):
                return "local_vision"
            # Priority 2: Gemini API fallback if configured AND explicitly permitted
            if bool(get_gemini_api_key()) and is_cloud_multimodal_enabled():
                return "gemini"
            # Priority 3: Clear failure (fail closed)
            return "no_vision_available"

    # Text-Only Routing: 100% identical preservation of existing behavior
    if explicit_mode in ("gemini", "pipeline", "chat", "gemma", "deepseek", "qwen"):
        if explicit_mode == "gemma":
            return "chat"
        return explicit_mode

    # If explicit_mode is auto, analyze the latest user message
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return "chat"

    latest = user_msgs[-1]
    content = latest.get("content", "")

    # 1. Pure architectural assessment without code generation request -> DeepSeek Architect
    if ARCHITECTURAL_ASSESSMENT_PATTERNS.search(content) and not CODE_IMPLEMENTATION_INTENT_PATTERNS.search(content):
        return "deepseek"

    # 2. Code intent -> Coding Pipeline (DeepSeek + Qwen)
    if "```" in content or CODING_PATTERNS.search(content):
        return "pipeline"

    # 3. Current info or difficult reasoning -> Gemini API
    if CURRENT_INFO_PATTERNS.search(content) or DIFFICULT_REASONING_PATTERNS.search(content):
        return "gemini"

    # 4. Normal / Local / Lightweight -> Local Gemma 4
    return "chat"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Check Ollama availability and report active models."""
    available = get_available_ollama_models()
    chat_m = resolve_model(CHAT_MODELS, available)
    arch_m = resolve_model(ARCHITECT_MODELS, available)
    coder_m = resolve_model(CODER_MODELS, available)
    aider_bin = get_aider_path()

    return {
        "ok": len(available) > 0,
        "models": available,
        "chat_model": chat_m,
        "architect_model": arch_m,
        "coder_model": coder_m,
        "gemini_model": get_gemini_model(),
        "has_gemini_key": bool(get_gemini_api_key()),
        "search_grounding": is_search_grounding_enabled(),
        "aider_available": aider_bin is not None,
        "aider_path": aider_bin,
    }


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Return user configuration status."""
    raw_key = get_gemini_api_key()
    masked_key = ""
    if raw_key:
        masked_key = raw_key[:6] + "..." + raw_key[-4:] if len(raw_key) > 10 else "******"

    return jsonify({
        "ok": True,
        "has_gemini_key": bool(raw_key),
        "masked_key": masked_key,
        "gemini_api_key_masked": masked_key,
        "gemini_model": get_gemini_model(),
        "available_gemini_models": GEMINI_MODELS,
        "search_grounding": is_search_grounding_enabled(),
        "allow_cloud_multimodal": is_cloud_multimodal_enabled(),
    })


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update and persist API settings."""
    data = request.get_json() or {}
    key = data.get("api_key") or data.get("gemini_api_key")
    model = data.get("model") or data.get("gemini_model")
    grounding = data.get("search_grounding")
    cloud_multimodal = data.get("allow_cloud_multimodal")
    if cloud_multimodal is None:
        cloud_multimodal = data.get("cloud_multimodal")

    save_env_config(
        key=key if key is not None and key.strip() else None,
        model=model if model is not None else None,
        grounding=grounding if grounding is not None else None,
        cloud_multimodal=cloud_multimodal if cloud_multimodal is not None else None,
    )
    return jsonify({"ok": True, "message": "Settings saved successfully."})


@app.route("/api/settings/verify-gemini", methods=["POST"])
def verify_gemini():
    """Validate Gemini API key and model connectivity."""
    data = request.get_json() or {}
    test_key = data.get("api_key") or data.get("gemini_api_key") or get_gemini_api_key()
    test_model = data.get("model") or data.get("gemini_model") or get_gemini_model()

    if not test_key:
        return jsonify({"ok": False, "error": "No Gemini API key provided."}), 400

    try:
        from google import genai
        client = genai.Client(api_key=test_key)

        candidates = [test_model]
        for fb in ("gemini-flash-lite-latest", "gemini-3.5-flash-lite"):
            if fb != test_model and fb not in candidates:
                candidates.append(fb)

        last_error = None
        for candidate in candidates:
            try:
                resp = client.models.generate_content(
                    model=candidate,
                    contents="Say 'Connection verified.'"
                )
                msg = resp.text.strip() if hasattr(resp, "text") and resp.text else "Verified"
                notice = ""
                if candidate != test_model:
                    notice = f" (Note: {test_model} was busy, verified with {candidate})"
                return jsonify({
                    "ok": True,
                    "model": candidate,
                    "message": f"Successfully connected to {candidate}! ({msg}){notice}"
                })
            except Exception as ce:
                last_error = ce
                err_str = str(ce).lower()
                if "api_key_invalid" in err_str or "invalid api key" in err_str or "unauthenticated" in err_str:
                    break

        raise last_error
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/models")
def list_models():
    """List available models."""
    available = get_available_ollama_models()
    return {"models": available}


@app.route("/api/aider/info")
def aider_info():
    """Inspect local Aider installation."""
    aider_bin = get_aider_path()
    if not aider_bin:
        return {"installed": False, "error": "Aider not found in path"}

    try:
        ver_res = subprocess.run([aider_bin, "--version"], capture_output=True, text=True, timeout=4)
        ver = ver_res.stdout.strip() or ver_res.stderr.strip()
    except Exception as e:
        ver = f"Installed (could not query version: {e})"

    available = get_available_ollama_models()
    coder_m = resolve_model(CODER_MODELS, available)
    arch_m = resolve_model(ARCHITECT_MODELS, available)

    cmd = f"aider --model ollama/{coder_m} --architect --editor-model ollama/{arch_m}"

    return {
        "installed": True,
        "path": aider_bin,
        "version": ver,
        "recommended_command": cmd,
    }


# ── Request Cancellation Registry ────────────────────────────────────────────

class RequestCancellationRegistry:
    """
    Thread-safe registry mapping active chat request IDs to their cancellation events.
    Guarantees:
    - Atomically registers unique request IDs.
    - Rejects duplicate active request IDs without overwriting existing events.
    - Scoped abort signaling: aborting request A never impacts request B.
    - Idempotent cleanup and abort calls.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._registry: Dict[str, threading.Event] = {}

    def register(self, request_id: str) -> Tuple[threading.Event, bool]:
        """
        Atomically register a cancellation event for request_id.
        Returns (event, True) if newly registered.
        Returns (existing_event, False) if request_id is already active (duplicate).
        """
        with self._lock:
            if request_id in self._registry:
                return self._registry[request_id], False
            ev = threading.Event()
            self._registry[request_id] = ev
            return ev, True

    def unregister(self, request_id: str) -> Optional[threading.Event]:
        """Atomically remove request_id from registry."""
        with self._lock:
            return self._registry.pop(request_id, None)

    def abort(self, request_id: str) -> bool:
        """
        Signal cancellation for request_id.
        Returns True if request_id was found and event was set, False otherwise.
        """
        with self._lock:
            ev = self._registry.get(request_id)
        if ev is not None:
            ev.set()
            return True
        return False

    def is_active(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._registry

    def get_event(self, request_id: str) -> Optional[threading.Event]:
        with self._lock:
            return self._registry.get(request_id)

    def active_count(self) -> int:
        with self._lock:
            return len(self._registry)

    def clear(self) -> None:
        """Clear all registered events (for testing isolation)."""
        with self._lock:
            self._registry.clear()


cancellation_registry = RequestCancellationRegistry()


def is_valid_uuid(val: Any) -> bool:
    """
    Validate that val is a strictly canonical RFC 4122 UUID v4 string:
    - Must be a string of exactly 36 lowercase characters with standard hyphenation:
      xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (where y in [8, 9, a, b]).
    - Rejects prefixed identifiers (e.g. req_, req-).
    - Rejects 32-character unhyphenated hex representations.
    - Rejects uppercase characters, whitespace, or non-v4 UUIDs.
    """
    if not val or not isinstance(val, str) or len(val) != 36:
        return False
    try:
        parsed = uuid.UUID(val, version=4)
        return str(parsed) == val
    except (ValueError, AttributeError, TypeError):
        return False


class ClosableStream:
    """
    Iterator wrapper around an SSE generator that guarantees invoking on_close
    when iteration completes, raises an error, response.close() is called, or the object is GC'd.
    """
    def __init__(self, generator, on_close: Optional[Callable[[], None]] = None):
        self._gen = generator
        self._on_close = on_close
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._gen)
        except StopIteration:
            self.close()
            raise
        except Exception:
            self.close()
            raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            if hasattr(self._gen, "close"):
                self._gen.close()
        finally:
            if self._on_close:
                try:
                    self._on_close()
                except Exception as e:
                    logger.warning(f"Error in ClosableStream on_close: {e}")

    def __del__(self):
        self.close()


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Stream chat response with Autonomous Model Switching:
    - Chat/Vision -> Gemma
    - Coding Pipeline -> DeepSeek-Coder-V2 16B (Architect) -> Qwen2.5-Coder 7B (Coder)
    - Cloud AI -> Gemini API
    Unified RAG integration provides grounded context when appropriate without pollutive history accumulation.
    """
    data = request.get_json(silent=True) or {}
    raw_id = data.get("request_id")
    if raw_id:
        if not is_valid_uuid(raw_id):
            return jsonify({"error": "Invalid request_id format; must be a valid UUID"}), 400
        request_id = str(raw_id)
    else:
        request_id = str(uuid.uuid4())

    cancel_event, registered = cancellation_registry.register(request_id)
    if not registered:
        return jsonify({"error": f"Duplicate active request_id '{request_id}'"}), 409

    try:
        messages = data.get("messages", [])
        mode = data.get("mode", "auto")  # 'auto', 'pipeline', 'gemma', 'deepseek', 'qwen', 'gemini'
        rag_mode = data.get("rag_mode", "auto")

        # Conversational Memory & Session Identifier
        raw_conv_id = data.get("conversation_id") or data.get("session_id")
        if raw_conv_id and isinstance(raw_conv_id, str) and raw_conv_id.strip():
            conversation_id = raw_conv_id.strip()
        else:
            conversation_id = str(uuid.uuid4())

        # Message Identification & Voice-Ready Source
        raw_msg_id = data.get("message_id")
        has_explicit_message_id = bool(raw_msg_id and isinstance(raw_msg_id, str) and raw_msg_id.strip())
        if has_explicit_message_id:
            user_message_id = raw_msg_id.strip()
        else:
            user_message_id = str(uuid.uuid4())

        raw_source = data.get("source", "text")
        source = "voice" if raw_source == "voice" else "text"

        message_store = get_message_store()
        existing_msg = message_store.get_message(conversation_id, user_message_id)

        # Concurrency / In-Flight check
        if existing_msg and existing_msg.status == MessageStatus.PROCESSING.value:
            cancellation_registry.unregister(request_id)
            return jsonify({
                "error": f"Message '{user_message_id}' is currently being processed",
                "status": "processing",
                "conversation_id": conversation_id,
                "message_id": user_message_id,
            }), 409

        # Idempotent replay of completed message
        if existing_msg and existing_msg.status == MessageStatus.COMPLETED.value:
            cached_asst = message_store.get_assistant_response_for_user_message(conversation_id, user_message_id)
            if cached_asst:
                def stream_cached_completion():
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'persisted', 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'route', 'route': 'cached', 'request_id': request_id, 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'cached': True, 'rag_mode': 'off', 'rag_status': 'cached', 'rag_sources': [], 'pdf_sources': []})}\n\n"
                    yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'chat', 'model': cached_asst.model_id or 'cached', 'title': 'Cached Response'})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'stage': 'chat', 'token': cached_asst.content})}\n\n"
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'chat'})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': cached_asst.message_id, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'cached', 'cached': True})}\n\n"

                resp = Response(
                    stream_with_context(stream_cached_completion()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
                resp.call_on_close(lambda: cancellation_registry.unregister(request_id))
                return resp

        # 1. Attachment inspection & strict validation
        attach_summary = extract_attachment_summary(messages)
        has_images = attach_summary["has_images"]
        validated_images = []

        if has_images:
            validated_images, val_err = validate_and_decode_images(attach_summary["raw_images_b64"])
            if val_err:
                cancellation_registry.unregister(request_id)
                return jsonify({"error": val_err}), 400

        available = get_available_ollama_models()
        chat_m = resolve_model(CHAT_MODELS, available)
        arch_m = resolve_model(ARCHITECT_MODELS, available)
        coder_m = resolve_model(CODER_MODELS, available)
        vision_m = resolve_vision_model(available)

        chosen_route = classify_intent(
            messages,
            mode,
            has_images=has_images,
            local_vision_model=vision_m,
        )

        # 2. Strict rejection of incompatible or disabled modes (Option B)
        if chosen_route.startswith("incompatible_mode:"):
            cancellation_registry.unregister(request_id)
            incompat = chosen_route.split(":", 1)[1]
            model_label = "DeepSeek-Coder-V2 + Qwen2.5-Coder" if incompat == "pipeline" else (
                "DeepSeek-Coder-V2" if incompat == "deepseek" else (
                    "Qwen2.5-Coder" if incompat == "qwen" else "Local Gemma 4"
                )
            )
            return jsonify({
                "error": (
                    f"The selected mode '{incompat}' ({model_label}) is a text/code model and cannot process image attachments. "
                    f"Please select 'Auto' or a vision-capable mode (such as 'Gemini API' or a verified local vision model) to analyze images."
                )
            }), 400

        if chosen_route == "cloud_multimodal_disabled":
            cancellation_registry.unregister(request_id)
            return jsonify({
                "error": (
                    "Cloud image processing is disabled. To analyze images with Gemini, "
                    "enable 'Cloud Multimodal Processing' in Settings (⚙️) or configure ALLOW_CLOUD_MULTIMODAL=true in .env."
                )
            }), 400

        if chosen_route == "no_vision_available":
            cancellation_registry.unregister(request_id)
            return jsonify({
                "error": (
                    "Image processing requires a vision-capable model. "
                    "No verified local vision model (e.g. Llama-3.2-Vision or LLaVA) was found in Ollama, "
                    "and cloud multimodal processing is currently disabled for privacy. "
                    "To analyze images, install a supported local vision model or enable 'Cloud Multimodal Processing' in Settings (⚙️)."
                )
            }), 400

        # Extract latest user message content for retrieval query and storage
        user_msgs = [m for m in messages if m.get("role") == "user"]
        latest_user_text = user_msgs[-1].get("content", "") if user_msgs else ""

        # Durable persistence of user turn (status: 'persisted' -> 'processing')
        if latest_user_text:
            message_store.persist_user_message(
                conversation_id=conversation_id,
                message_id=user_message_id,
                content=latest_user_text,
                source=source,
            )
            message_store.update_status(conversation_id, user_message_id, MessageStatus.PROCESSING.value)
        else:
            message_store.register_message(
                conversation_id=conversation_id,
                message_id=user_message_id,
                role="user",
                content="",
                source=source,
                status=MessageStatus.PERSISTED.value,
            )
            message_store.update_status(conversation_id, user_message_id, MessageStatus.PROCESSING.value)

        # Store user turn into conversational memory asynchronously (non-blocking)
        memory_store = get_conversation_store()
        if latest_user_text:
            user_mem_msg = ConversationMessage(
                conversation_id=conversation_id,
                message_id=user_message_id,
                role="user",
                content=latest_user_text,
                source=source,
                agent_role="user",
                model_id="user",
            )
            memory_store.store_message_async(user_mem_msg)

        # Context pruning: partition into recent conversational turns vs older memory
        recent_messages, older_messages = ConversationMemoryStore.filter_recent_context(messages)
        if older_messages:
            for om in older_messages:
                om_text = om.get("content", "")
                if om_text:
                    om_role = om.get("role", "user")
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=str(uuid.uuid4()),
                            role=om_role,
                            content=om_text,
                            agent_role="user" if om_role == "user" else "assistant",
                            model_id="user" if om_role == "user" else "assistant",
                        )
                    )

        # Execute non-blocking RAG context retrieval (documents, codebase, and/or conversational memory)
        rag_service = get_rag_service()
        rag_result = rag_service.retrieve_context(
            query=latest_user_text,
            mode=rag_mode,
            conversation_id=conversation_id,
        )
        context_block = rag_result.context_text if rag_result.has_context else ""

        # Retrieve grounded context for user-selected PDF documents
        pdf_doc_ids = data.get("pdf_doc_ids", [])
        pdf_sources = []
        if pdf_doc_ids:
            from pdf_intelligence.registry import get_document_registry
            pdf_reg = get_document_registry()
            pdf_context, pdf_sources, pdf_err = pdf_reg.get_document_context(
                pdf_doc_ids, query=latest_user_text
            )
            if pdf_err:
                cancellation_registry.unregister(request_id)
                return jsonify({"error": pdf_err}), 400
            if pdf_context:
                if context_block:
                    context_block = f"{context_block}\n\n{pdf_context}"
                else:
                    context_block = pdf_context
    except Exception:
        cancellation_registry.unregister(request_id)
        raise

    def stream_pipeline():
        active_stage: str | None = None
        model_manager = get_model_manager()

        def stream_ollama_stage(
            stage_name: str,
            model_name: str,
            stage_messages: list[dict],
            max_continuations: int = 2,
            emit_done_attr: bool = False,
            heartbeat_interval: float = 2.0,
            lease: Optional[ModelLease] = None,
        ):
            accumulated_text = ""
            last_done_reason = "stop"
            is_truncated = False

            chunk_queue: queue.Queue = queue.Queue()
            stage_cancel_event = threading.Event()
            active_resp: list[Any] = [None]
            resp_lock = threading.Lock()

            def is_cancelled() -> bool:
                return cancel_event.is_set() or stage_cancel_event.is_set()

            def worker():
                nonlocal accumulated_text, last_done_reason, is_truncated
                current_messages = list(stage_messages)
                attempt = 0

                try:
                    while not is_cancelled():
                        payload = {
                            "model": model_name,
                            "messages": current_messages,
                            "stream": True,
                            "options": OLLAMA_OPTIONS,
                        }

                        resp = None
                        try:
                            resp = http_requests.post(
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json=payload,
                                stream=True,
                                timeout=OLLAMA_TIMEOUT,
                            )
                        except Exception as req_err:
                            if not is_cancelled():
                                chunk_queue.put(("error", req_err))
                            return

                        with resp_lock:
                            if is_cancelled():
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                                return
                            active_resp[0] = resp

                        try:
                            resp.raise_for_status()
                            for line in resp.iter_lines(decode_unicode=True):
                                if is_cancelled():
                                    return
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except Exception:
                                    continue

                                token = chunk.get("message", {}).get("content", "")
                                is_done = chunk.get("done", False)
                                done_reason = chunk.get("done_reason", "stop") if is_done else None

                                if token:
                                    accumulated_text += token
                                if is_done:
                                    last_done_reason = done_reason or "stop"

                                if token or is_done:
                                    chunk_queue.put(("chunk", token, is_done, done_reason))

                                if is_done:
                                    break
                        except Exception as stream_err:
                            if not is_cancelled():
                                chunk_queue.put(("error", stream_err))
                            return
                        finally:
                            with resp_lock:
                                active_resp[0] = None
                            if resp is not None:
                                try:
                                    resp.close()
                                except Exception:
                                    pass

                        if last_done_reason == "length" and attempt < max_continuations and not is_cancelled():
                            attempt += 1
                            current_messages = list(stage_messages)
                            current_messages.append({"role": "assistant", "content": accumulated_text})
                            current_messages.append({
                                "role": "user",
                                "content": "Please continue directly from where you stopped. Do not repeat any preceding sentences, headings, or thoughts."
                            })
                            continue

                        break

                    is_truncated = (last_done_reason == "length")
                    if not is_cancelled():
                        chunk_queue.put(("stage_done", accumulated_text, is_truncated, last_done_reason))
                except Exception as fatal_err:
                    if not is_cancelled():
                        chunk_queue.put(("error", fatal_err))
                finally:
                    # Atomic worker completion: signal state machine.
                    # mark_worker_finished() returns True iff the generator already
                    # detached (DETACHED state) and this worker must release the lease.
                    if lease is not None:
                        should_release = lease.mark_worker_finished()
                        if should_release:
                            lease.release_from_worker()

            worker_thread = threading.Thread(
                target=worker,
                name=f"ollama-worker-{stage_name}-{model_name}",
                daemon=True,
            )
            if lease is not None:
                lease.transfer_to_worker(worker_thread)
            worker_thread.start()

            try:
                while True:
                    if is_cancelled():
                        raise ModelLeaseCancelledError(f"Stage '{stage_name}' cancelled by client")
                    try:
                        item = chunk_queue.get(timeout=heartbeat_interval)
                    except queue.Empty:
                        if is_cancelled():
                            raise ModelLeaseCancelledError(f"Stage '{stage_name}' cancelled by client")
                        if not worker_thread.is_alive():
                            break
                        # Heartbeat comment to keep connection alive and detect client disconnect
                        yield ": heartbeat\n\n"
                        continue

                    tag = item[0]
                    if tag == "chunk":
                        if is_cancelled():
                            raise ModelLeaseCancelledError(f"Stage '{stage_name}' cancelled by client")
                        _, token, is_done, _ = item
                        if token:
                            evt = {"type": "token", "stage": stage_name, "token": token}
                            if emit_done_attr:
                                evt["done"] = is_done
                            yield f"data: {json.dumps(evt)}\n\n"
                    elif tag == "error":
                        raise item[1]
                    elif tag == "stage_done":
                        if is_cancelled():
                            raise ModelLeaseCancelledError(f"Stage '{stage_name}' cancelled by client")
                        _, accumulated_text, is_truncated, last_done_reason = item
                        break
            finally:
                stage_cancel_event.set()
                with resp_lock:
                    if active_resp[0] is not None:
                        try:
                            active_resp[0].close()
                        except Exception:
                            pass
                        active_resp[0] = None
                worker_thread.join(timeout=5.0)
                if worker_thread.is_alive():
                    logger.critical(
                        f"Inference worker {worker_thread.name} did not terminate within 5.0s join timeout. "
                        f"Lease {lease.id if lease else 'none'} remains active under worker ownership."
                    )
                    if lease is not None:
                        lease.mark_stuck()

            return accumulated_text, is_truncated, last_done_reason

        try:
            # 0. Acknowledgement of persistence to client (when client tracks message_id)
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'persisted', 'source': source})}\n\n"

            # 1. Announce routing decision to client
            yield f"data: {json.dumps({'type': 'route', 'route': chosen_route, 'request_id': request_id, 'conversation_id': conversation_id, 'message_id': user_message_id, 'source': source, 'status': 'persisted', 'architect': arch_m, 'coder': coder_m, 'chat_model': chat_m, 'vision_model': vision_m, 'rag_mode': rag_result.mode, 'rag_status': rag_result.status, 'rag_sources': rag_result.sources, 'pdf_sources': pdf_sources})}\n\n"

            if chosen_route == "pipeline":
                # ── STAGE 1: ARCHITECT (DeepSeek-Coder-V2 16B) ─────────────
                active_stage = "architect"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'architect', 'model': arch_m, 'title': 'System Architecture & Technical Design'})}\n\n"

                arch_user_text = (
                    f"{context_block}\n\n---\n{latest_user_text}"
                    if context_block
                    else latest_user_text
                )
                arch_messages = [
                    {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                    *[m for m in recent_messages[:-1] if m.get("role") != "system"],
                    {"role": "user", "content": arch_user_text},
                ]

                with model_manager.acquire_generation_lease(arch_m, is_cancelled=cancel_event.is_set) as arch_lease:
                    arch_response, arch_truncated, arch_done_reason = yield from stream_ollama_stage(
                        "architect", arch_m, arch_messages, max_continuations=2, lease=arch_lease
                    )

                if arch_truncated:
                    active_stage = None
                    err_msg = "Architectural assessment reached maximum generation length. Implementation stage was paused to prevent incomplete code generation."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'architect', 'truncated': True, 'done_reason': arch_done_reason})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'architect', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'pipeline', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Check if prompt is purely an architectural assessment
                is_pure_architectural = (
                    ARCHITECTURAL_ASSESSMENT_PATTERNS.search(latest_user_text)
                    and not CODE_IMPLEMENTATION_INTENT_PATTERNS.search(latest_user_text)
                )

                # Record completed Stage 1 response into message_store and memory store
                arch_msg_id = str(uuid.uuid4())
                if arch_response and arch_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=arch_msg_id,
                        parent_message_id=user_message_id,
                        content=arch_response,
                        model_id=arch_m,
                        agent_role="architect",
                        pipeline_stage="architect",
                        source=source,
                        status="completed" if is_pure_architectural else "persisted",
                    )
                    if is_pure_architectural:
                        message_store.mark_completed(conversation_id, user_message_id, arch_msg_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=arch_msg_id,
                            role="assistant",
                            content=arch_response,
                            source=source,
                            model_id=arch_m,
                            agent_role="architect",
                            pipeline_stage="architect",
                        )
                    )

                if is_pure_architectural:
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'architect'})}\n\n"
                    active_stage = None
                    aider_cmd = f"aider --model ollama/{arch_m} --architect"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': arch_msg_id, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'pipeline', 'analytical_only': True, 'aider_cmd': aider_cmd})}\n\n"
                    return

                yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'architect'})}\n\n"
                active_stage = None

                # Pipeline handoff: independently verify all 3 Stage 1 release conditions before Stage 2
                model_manager.verify_pipeline_handoff(
                    arch_lease,
                    arch_m,
                    timeout=5.0,
                    is_cancelled=cancel_event.is_set,
                )

                # ── STAGE 2: CODER (Qwen2.5-Coder 7B) ──────────────────────
                active_stage = "coder"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'coder', 'model': coder_m, 'title': 'Implementation & Source Code'})}\n\n"

                # Coder receives original prompt + Architect's blueprint + Repository reference context
                coder_prompt_parts = [
                    f"## ARCHITECTURAL BLUEPRINT (from Principal Architect DeepSeek-Coder-V2):\n\n{arch_response}\n\n",
                ]
                if context_block:
                    coder_prompt_parts.append(f"## REPOSITORY REFERENCE CONTEXT:\n\n{context_block}\n\n")
                coder_prompt_parts.append(
                    f"---\n## USER CODING SPECIFICATION:\n{latest_user_text}\n\n"
                    f"Please write the complete, clean, production-grade source code now adhering strictly to the architecture above."
                )
                coder_prompt = "".join(coder_prompt_parts)

                coder_messages = [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    {"role": "user", "content": coder_prompt},
                ]

                with model_manager.acquire_generation_lease(coder_m, is_cancelled=cancel_event.is_set) as coder_lease:
                    coder_response, coder_truncated, coder_done_reason = yield from stream_ollama_stage(
                        "coder", coder_m, coder_messages, max_continuations=2, lease=coder_lease
                    )

                if coder_truncated:
                    active_stage = None
                    err_msg = "Code implementation reached maximum generation length."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'coder', 'truncated': True, 'done_reason': coder_done_reason})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'coder', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'pipeline', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Record completed Stage 2 response into message_store and memory store
                coder_msg_id = str(uuid.uuid4())
                if coder_response and coder_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=coder_msg_id,
                        parent_message_id=user_message_id,
                        content=coder_response,
                        model_id=coder_m,
                        agent_role="coder",
                        pipeline_stage="coder",
                        source=source,
                        status="completed",
                    )
                    message_store.mark_completed(conversation_id, user_message_id, coder_msg_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=coder_msg_id,
                            role="assistant",
                            content=coder_response,
                            source=source,
                            model_id=coder_m,
                            agent_role="coder",
                            pipeline_stage="coder",
                        )
                    )

                yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'coder'})}\n\n"
                active_stage = None

                # Finished pipeline
                aider_cmd = f"aider --model ollama/{coder_m} --architect --editor-model ollama/{arch_m}"
                if has_explicit_message_id:
                    yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': coder_msg_id, 'source': source})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'pipeline', 'aider_cmd': aider_cmd})}\n\n"

            elif chosen_route == "deepseek":
                # Standalone Architect
                active_stage = "architect"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'architect', 'model': arch_m, 'title': 'DeepSeek Architect'})}\n\n"
                arch_user_text = (
                    f"{context_block}\n\n---\n{latest_user_text}"
                    if context_block
                    else latest_user_text
                )
                arch_messages = [
                    {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                    *[m for m in recent_messages[:-1] if m.get("role") != "system"],
                    {"role": "user", "content": arch_user_text},
                ]
                with model_manager.acquire_generation_lease(arch_m, is_cancelled=cancel_event.is_set) as arch_lease:
                    arch_response, arch_truncated, arch_done_reason = yield from stream_ollama_stage(
                        "architect", arch_m, arch_messages, max_continuations=2, lease=arch_lease
                    )
                if arch_truncated:
                    active_stage = None
                    err_msg = "Architect response truncated due to maximum length."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'architect', 'truncated': True, 'done_reason': arch_done_reason})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'architect', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'deepseek', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Record completed response into message_store and memory store
                deepseek_asst_id = str(uuid.uuid4())
                if arch_response and arch_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=deepseek_asst_id,
                        parent_message_id=user_message_id,
                        content=arch_response,
                        model_id=arch_m,
                        agent_role="architect",
                        pipeline_stage="architect",
                        source=source,
                        status="completed",
                    )
                    message_store.mark_completed(conversation_id, user_message_id, deepseek_asst_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=deepseek_asst_id,
                            role="assistant",
                            content=arch_response,
                            source=source,
                            model_id=arch_m,
                            agent_role="architect",
                            pipeline_stage="architect",
                        )
                    )

                active_stage = None
                if has_explicit_message_id:
                    yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': deepseek_asst_id, 'source': source})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'deepseek'})}\n\n"

            elif chosen_route == "qwen":
                # Standalone Coder
                active_stage = "coder"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'coder', 'model': coder_m, 'title': 'Qwen2.5 Coder'})}\n\n"
                coder_user_text = (
                    f"{context_block}\n\n---\n{latest_user_text}"
                    if context_block
                    else latest_user_text
                )
                coder_messages = [
                    {"role": "system", "content": CODER_SYSTEM_PROMPT},
                    *[m for m in recent_messages[:-1] if m.get("role") != "system"],
                    {"role": "user", "content": coder_user_text},
                ]
                with model_manager.acquire_generation_lease(coder_m, is_cancelled=cancel_event.is_set) as coder_lease:
                    coder_response, coder_truncated, coder_done_reason = yield from stream_ollama_stage(
                        "coder", coder_m, coder_messages, max_continuations=2, lease=coder_lease
                    )
                if coder_truncated:
                    active_stage = None
                    err_msg = "Coder response truncated due to maximum length."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'coder', 'truncated': True, 'done_reason': coder_done_reason})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'coder', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'qwen', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Record completed response into message_store and memory store
                qwen_asst_id = str(uuid.uuid4())
                if coder_response and coder_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=qwen_asst_id,
                        parent_message_id=user_message_id,
                        content=coder_response,
                        model_id=coder_m,
                        agent_role="coder",
                        pipeline_stage="coder",
                        source=source,
                        status="completed",
                    )
                    message_store.mark_completed(conversation_id, user_message_id, qwen_asst_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=qwen_asst_id,
                            role="assistant",
                            content=coder_response,
                            source=source,
                            model_id=coder_m,
                            agent_role="coder",
                            pipeline_stage="coder",
                        )
                    )

                yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'coder'})}\n\n"
                active_stage = None
                if has_explicit_message_id:
                    yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': qwen_asst_id, 'source': source})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'qwen'})}\n\n"

            elif chosen_route == "local_vision":
                # ── LOCAL VISION (Ollama Verified Vision Model) ───────────
                target_model = vision_m
                active_stage = "vision"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'vision', 'model': target_model, 'title': f'Local Vision Assistant ({target_model})'})}\n\n"

                vision_user_content = (
                    f"{context_block}\n\n---\nBased on the reference material above (if relevant), please analyze the image(s) and answer the user's question:\n{latest_user_text}"
                    if context_block
                    else latest_user_text
                )

                sanitized_history = []
                for m in recent_messages[:-1]:
                    if not isinstance(m, dict) or m.get("role") == "system":
                        continue
                    entry: Dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content", "")}
                    if m.get("images"):
                        clean_imgs = []
                        for img in m["images"]:
                            if isinstance(img, str) and img.strip():
                                c = img.strip()
                                if "," in c and c.startswith("data:"):
                                    c = c.split(",", 1)[1]
                                clean_imgs.append(c)
                        if clean_imgs:
                            entry["images"] = clean_imgs
                    sanitized_history.append(entry)

                last_msg = messages[-1] if messages else {}
                raw_images = last_msg.get("images", []) if isinstance(last_msg, dict) else []
                clean_images = []
                for img in raw_images:
                    if isinstance(img, str) and img.strip():
                        c_img = img.strip()
                        if "," in c_img and c_img.startswith("data:"):
                            c_img = c_img.split(",", 1)[1]
                        clean_images.append(c_img)

                user_entry: Dict[str, Any] = {"role": "user", "content": vision_user_content}
                if clean_images:
                    user_entry["images"] = clean_images

                vision_messages = [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    *sanitized_history,
                    user_entry,
                ]

                with model_manager.acquire_generation_lease(target_model, is_cancelled=cancel_event.is_set) as vision_lease:
                    vision_response, vision_truncated, vision_done_reason = yield from stream_ollama_stage(
                        "vision", target_model, vision_messages, max_continuations=2, emit_done_attr=True, lease=vision_lease
                    )

                if vision_truncated:
                    active_stage = None
                    err_msg = "Response truncated due to maximum length."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'vision', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'local_vision', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Record completed response into message_store and memory store
                vision_asst_id = str(uuid.uuid4())
                if vision_response and vision_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=vision_asst_id,
                        parent_message_id=user_message_id,
                        content=vision_response,
                        model_id=target_model,
                        agent_role="vision_assistant",
                        pipeline_stage="vision",
                        source=source,
                        status="completed",
                    )
                    message_store.mark_completed(conversation_id, user_message_id, vision_asst_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=vision_asst_id,
                            role="assistant",
                            content=vision_response,
                            source=source,
                            model_id=target_model,
                            agent_role="vision_assistant",
                            pipeline_stage="vision",
                        )
                    )

                active_stage = None
                if has_explicit_message_id:
                    yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': vision_asst_id, 'source': source})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'local_vision'})}\n\n"

            elif chosen_route == "gemini":
                # ── GEMINI API (Current Information & Strong Reasoning) ────
                if has_images and not is_cloud_multimodal_enabled():
                    yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'gemini', 'model': get_gemini_model(), 'title': 'Gemini Cloud Assistant'})}\n\n"
                    err_msg = "Cloud image processing is disabled. Enable 'Cloud Multimodal Processing' in Settings (⚙️) to analyze images with Gemini."
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'gemini', 'error': err_msg})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'gemini', 'error': err_msg})}\n\n"
                    return

                api_key = get_gemini_api_key()
                gemini_model = get_gemini_model()
                search_grounding = is_search_grounding_enabled()

                if not api_key:
                    yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'gemini', 'model': gemini_model, 'title': 'Gemini Cloud Assistant'})}\n\n"
                    yield f"data: {json.dumps({'type': 'token', 'stage': 'gemini', 'token': '⚠️ **Gemini API key is not configured.**\n\nTo enable real-time current information and strong reasoning, click **⚙️ Settings** in the top toolbar to enter your `GEMINI_API_KEY`, or add it to `.env`.'})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'gemini', 'need_key': True})}\n\n"
                    return

                active_stage = "gemini"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'gemini', 'model': gemini_model, 'title': f'Gemini Cloud Assistant ({gemini_model})'})}\n\n"

                try:
                    from google import genai
                    from google.genai import types

                    client = genai.Client(api_key=api_key)

                    system_instruction = (
                        "You are a state-of-the-art AI assistant powered by Google Gemini. "
                        "You excel at providing accurate, real-time, current information and strong step-by-step reasoning. "
                        "When answering, provide well-structured, clear explanations."
                    )

                    config_kwargs = {
                        "system_instruction": system_instruction,
                    }
                    if search_grounding:
                        config_kwargs["tools"] = [{"google_search": {}}]

                    config = types.GenerateContentConfig(**config_kwargs)

                    gemini_contents = []
                    for i, m in enumerate(recent_messages):
                        role = "user" if m.get("role") == "user" else "model"
                        parts = []
                        text = m.get("content", "")
                        if text:
                            # Augment latest user prompt with retrieved context
                            if context_block and i == len(recent_messages) - 1 and role == "user":
                                text = f"{context_block}\n\n---\n{text}"
                            parts.append(types.Part.from_text(text=text))
                        for img in m.get("images", []):
                            if isinstance(img, str) and img.strip():
                                c_img = img.strip()
                                if "," in c_img and c_img.startswith("data:"):
                                    c_img = c_img.split(",", 1)[1]
                                try:
                                    raw_data = base64.b64decode(c_img, validate=True)
                                    detected_mime = sniff_image_mime(raw_data[:16]) or "image/jpeg"
                                    parts.append(types.Part.from_bytes(data=raw_data, mime_type=detected_mime))
                                except Exception:
                                    pass
                        if parts:
                            gemini_contents.append(types.Content(role=role, parts=parts))

                    if not gemini_contents:
                        gemini_contents = [types.Content(role="user", parts=[types.Part.from_text(text="Hello")])]

                    def stream_gemini_with_fallback(primary_model, primary_config):
                        try:
                            for chunk in client.models.generate_content_stream(
                                model=primary_model,
                                contents=gemini_contents,
                                config=primary_config
                            ):
                                yield chunk.text or ""
                        except Exception as e1:
                            err_str = str(e1).lower()
                            # Fallback 1: If search grounding failed (e.g. 429 quota on free tier), retry without tools
                            if primary_config and primary_config.tools:
                                fallback_config = types.GenerateContentConfig(system_instruction=system_instruction)
                                try:
                                    for chunk in client.models.generate_content_stream(
                                        model=primary_model,
                                        contents=gemini_contents,
                                        config=fallback_config
                                    ):
                                        yield chunk.text or ""
                                    return
                                except Exception as e2:
                                    err_str = str(e2).lower()
                            # Fallback 2: If model is busy (503) or exhausted (429), retry with flash-lite
                            fallback_models = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
                            if ("503" in err_str or "unavailable" in err_str or "demand" in err_str or "429" in err_str):
                                for fb_model in fallback_models:
                                    if fb_model == primary_model:
                                        continue
                                    try:
                                        fallback_config = types.GenerateContentConfig(system_instruction=system_instruction)
                                        for chunk in client.models.generate_content_stream(
                                            model=fb_model,
                                            contents=gemini_contents,
                                            config=fallback_config
                                        ):
                                            yield chunk.text or ""
                                        return
                                    except Exception:
                                        continue
                            raise e1

                    gemini_response = ""
                    for token in stream_gemini_with_fallback(gemini_model, config):
                        gemini_response += token
                        yield f"data: {json.dumps({'type': 'token', 'stage': 'gemini', 'token': token})}\n\n"

                    # Record completed response into message_store and memory store
                    gemini_asst_id = str(uuid.uuid4())
                    if gemini_response and gemini_response.strip():
                        message_store.persist_assistant_message(
                            conversation_id=conversation_id,
                            message_id=gemini_asst_id,
                            parent_message_id=user_message_id,
                            content=gemini_response,
                            model_id=gemini_model,
                            agent_role="cloud_assistant",
                            pipeline_stage="gemini",
                            source=source,
                            status="completed",
                        )
                        message_store.mark_completed(conversation_id, user_message_id, gemini_asst_id)
                        memory_store.store_message_async(
                            ConversationMessage(
                                conversation_id=conversation_id,
                                message_id=gemini_asst_id,
                                role="assistant",
                                content=gemini_response,
                                source=source,
                                model_id=gemini_model,
                                agent_role="cloud_assistant",
                                pipeline_stage="gemini",
                            )
                        )

                    active_stage = None
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': gemini_asst_id, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'gemini'})}\n\n"
                except Exception as g_err:
                    err_msg = str(g_err)
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'token', 'stage': 'gemini', 'token': f'\n\n⚠️ **Gemini API Error:** {err_msg}'})}\n\n"
                    if active_stage:
                        yield f"data: {json.dumps({'type': 'stage_end', 'stage': 'gemini', 'error': True})}\n\n"
                        active_stage = None
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'gemini', 'error': err_msg})}\n\n"

            else:
                # ── GENERAL CHAT & VISION (Gemma 4) ──────────────────────────
                target_model = chat_m
                chat_user_content = (
                    f"{context_block}\n\n---\nBased on the reference material above (if relevant), please answer the user's question:\n{latest_user_text}"
                    if context_block
                    else latest_user_text
                )

                last_msg = messages[-1] if messages else {}
                images = last_msg.get("images", []) if isinstance(last_msg, dict) else []
                user_entry: Dict[str, Any] = {"role": "user", "content": chat_user_content}
                if images:
                    user_entry["images"] = images

                chat_messages = [
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    *[m for m in recent_messages[:-1] if m.get("role") != "system"],
                    user_entry,
                ]

                active_stage = "chat"
                yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'chat', 'model': target_model, 'title': 'Gemma 4 Assistant'})}\n\n"

                with model_manager.acquire_generation_lease(target_model, is_cancelled=cancel_event.is_set) as chat_lease:
                    chat_response, chat_truncated, chat_done_reason = yield from stream_ollama_stage(
                        "chat", target_model, chat_messages, max_continuations=2, emit_done_attr=True, lease=chat_lease
                    )

                if chat_truncated:
                    active_stage = None
                    err_msg = "Response truncated due to maximum length."
                    message_store.mark_failed(conversation_id, user_message_id, err_msg)
                    yield f"data: {json.dumps({'type': 'error', 'stage': 'chat', 'error': err_msg, 'truncated': True})}\n\n"
                    if has_explicit_message_id:
                        yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'chat', 'truncated': True, 'interrupted': True, 'error': err_msg})}\n\n"
                    return

                # Record completed response into message_store and memory store
                chat_asst_id = str(uuid.uuid4())
                if chat_response and chat_response.strip():
                    message_store.persist_assistant_message(
                        conversation_id=conversation_id,
                        message_id=chat_asst_id,
                        parent_message_id=user_message_id,
                        content=chat_response,
                        model_id=target_model,
                        agent_role="general_assistant",
                        pipeline_stage="chat",
                        source=source,
                        status="completed",
                    )
                    message_store.mark_completed(conversation_id, user_message_id, chat_asst_id)
                    memory_store.store_message_async(
                        ConversationMessage(
                            conversation_id=conversation_id,
                            message_id=chat_asst_id,
                            role="assistant",
                            content=chat_response,
                            source=source,
                            model_id=target_model,
                            agent_role="general_assistant",
                            pipeline_stage="chat",
                        )
                    )

                active_stage = None
                if has_explicit_message_id:
                    yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'completed', 'assistant_message_id': chat_asst_id, 'source': source})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'done': True, 'mode': 'chat'})}\n\n"

        except ModelLeaseCancelledError as exc:
            err_msg = f"Generation cancelled: {exc}"
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg, 'cancelled': True})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'cancelled': True, 'interrupted': True, 'error': err_msg})}\n\n"
        except ModelLeaseTimeoutError as exc:
            err_msg = f"Timed out waiting for model resources: {exc}"
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'error': err_msg})}\n\n"
        except (http_requests.exceptions.ConnectionError, StateReconciliationError):
            err_msg = "Lost connection to Ollama server"
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'error': err_msg})}\n\n"
        except http_requests.exceptions.Timeout:
            err_msg = "Request timed out waiting for local model."
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'error': err_msg})}\n\n"
        except ModelLifecycleError as exc:
            err_msg = f"Model lifecycle error: {exc}"
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'error': err_msg})}\n\n"
        except Exception as exc:
            err_msg = str(exc)
            message_store.mark_failed(conversation_id, user_message_id, err_msg)
            err_stage = active_stage or ("chat" if chosen_route == "auto" else chosen_route)
            yield f"data: {json.dumps({'type': 'error', 'stage': err_stage, 'error': err_msg})}\n\n"
            if active_stage:
                yield f"data: {json.dumps({'type': 'stage_end', 'stage': active_stage, 'error': True})}\n\n"
                active_stage = None
            if has_explicit_message_id:
                yield f"data: {json.dumps({'type': 'ack', 'conversation_id': conversation_id, 'message_id': user_message_id, 'status': 'failed', 'error': err_msg, 'source': source})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'done': True, 'error': err_msg})}\n\n"
        finally:
            cancellation_registry.unregister(request_id)

    stream_wrapper = ClosableStream(
        stream_pipeline(),
        on_close=lambda: cancellation_registry.unregister(request_id),
    )
    resp = Response(
        stream_with_context(stream_wrapper),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    resp.call_on_close(lambda: cancellation_registry.unregister(request_id))
    return resp


@app.route("/api/chat/abort", methods=["POST"])
def abort_chat():
    """
    Cancel an active in-flight chat request by request_id.
    Guarantees:
    - Scoped strictly to the matching active request ID.
    - Sets the request's cancellation event and wakes any threads waiting in the lifecycle manager.
    - Idempotent: repeated calls for the same request ID return 200 without error.
    - Completed or unknown request IDs return 200 ignored without affecting other requests.
    """
    data = request.get_json(silent=True) or {}
    req_id = data.get("request_id")
    if not req_id or not isinstance(req_id, str):
        return jsonify({"status": "error", "message": "Missing or invalid request_id"}), 400
    if not is_valid_uuid(req_id):
        return jsonify({"status": "error", "message": "Invalid request_id format; must be a canonical UUID v4"}), 400

    aborted = cancellation_registry.abort(req_id)
    if aborted:
        manager = get_model_manager()
        with manager._cv:
            manager._cv.notify_all()
        return jsonify({"status": "aborted", "request_id": req_id}), 200

    return jsonify({"status": "ignored", "message": "Request not found or already completed"}), 200


@app.route("/api/messages/<conversation_id>/<message_id>/status", methods=["GET"])
def get_message_status(conversation_id: str, message_id: str):
    """
    Query backend lifecycle state for a specific message by conversation_id and message_id.
    States: 'received', 'processing', 'persisted', 'completed', 'failed', 'not_found'.
    If completed, includes assistant response data for client recovery without re-running inference.
    """
    store = get_message_store()
    rec = store.get_message(conversation_id, message_id)
    if rec is None:
        return jsonify({
            "conversation_id": conversation_id,
            "message_id": message_id,
            "status": "not_found",
        }), 404

    result = rec.to_dict()
    asst = store.get_assistant_response_for_user_message(conversation_id, message_id)
    if asst:
        asst_dict = asst.to_dict()
        result["assistant_message"] = asst_dict
        result["assistant_response"] = asst_dict
    return jsonify(result), 200


@app.route("/api/messages/status", methods=["POST"])
def query_messages_status():
    """
    Batch or single query of message statuses by conversation_id and message_id.
    Accepts:
      {"conversation_id": "...", "message_id": "..."}
      OR
      {"messages": [{"conversation_id": "...", "message_id": "..."}, ...]}
    """
    data = request.get_json(silent=True) or {}
    store = get_message_store()

    if "messages" in data and isinstance(data["messages"], list):
        results = []
        for item in data["messages"]:
            cid = str(item.get("conversation_id", "")).strip()
            mid = str(item.get("message_id", "")).strip()
            rec = store.get_message(cid, mid)
            if rec:
                item_dict = rec.to_dict()
                asst = store.get_assistant_response_for_user_message(cid, mid)
                if asst:
                    asst_dict = asst.to_dict()
                    item_dict["assistant_message"] = asst_dict
                    item_dict["assistant_response"] = asst_dict
                results.append(item_dict)
            else:
                results.append({
                    "conversation_id": cid,
                    "message_id": mid,
                    "status": "unknown",
                })
        return jsonify({"statuses": results, "messages": results}), 200

    cid = str(data.get("conversation_id", "")).strip()
    mid = str(data.get("message_id", "")).strip()
    if not cid or not mid:
        return jsonify({"error": "Missing conversation_id or message_id"}), 400

    rec = store.get_message(cid, mid)
    if rec is None:
        return jsonify({
            "conversation_id": cid,
            "message_id": mid,
            "status": "unknown",
        }), 404

    result = rec.to_dict()
    asst = store.get_assistant_response_for_user_message(cid, mid)
    if asst:
        asst_dict = asst.to_dict()
        result["assistant_message"] = asst_dict
        result["assistant_response"] = asst_dict
    return jsonify(result), 200


@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Accept an image file upload, validate size/magic bytes, and return its base64 encoding."""
    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    file = request.files["image"]
    if file.filename == "":
        return {"error": "Empty filename"}, 400

    allowed = {"png", "jpg", "jpeg", "gif", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        return {"error": f"Unsupported format '.{ext}'. Use: {', '.join(allowed)}"}, 400

    try:
        image_data = file.read()
    finally:
        file.close()
    if len(image_data) > MAX_IMAGE_RAW_BYTES:
        size_mb = round(len(image_data) / (1024 * 1024), 2)
        return {"error": f"Image exceeds 10 MB limit ({size_mb} MB)."}, 400

    detected_mime = sniff_image_mime(image_data[:16])
    if not detected_mime:
        return {"error": "Unsupported image format or corrupted header. Allowed: JPEG, PNG, WebP, GIF."}, 400

    b64 = base64.b64encode(image_data).decode("utf-8")

    return {
        "base64": b64,
        "filename": file.filename,
        "size": len(image_data),
        "mime_type": detected_mime,
    }


# ── PDF Intelligence Integration Endpoints ────────────────────────────────────

@app.route("/api/pdf/upload", methods=["POST"])
def upload_pdf():
    """
    Dedicated endpoint to accept, validate, and queue PDF documents for inspection.
    Enforces isolated storage, server-generated filenames, size/page limits,
    and asynchronous local background inspection.
    """
    from pdf_intelligence.registry import get_document_registry

    uploaded_files = request.files.getlist("pdf") or request.files.getlist("file")
    if not uploaded_files:
        return jsonify({"error": "No PDF file provided"}), 400

    reg = get_document_registry()
    results = []

    for file in uploaded_files:
        if not file or file.filename == "":
            continue

        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext != "pdf":
            return jsonify({
                "error": f"Invalid file '{file.filename}': Only .pdf documents are supported."
            }), 400

        try:
            file_bytes = file.read()
        finally:
            file.close()

        record, err = reg.validate_and_save_upload(file_bytes, file.filename)
        if err:
            return jsonify({"error": err, "filename": file.filename}), 400

        results.append(record.to_summary_dict())

    if not results:
        return jsonify({"error": "No valid PDF files uploaded"}), 400

    if len(results) == 1:
        return jsonify(results[0]), 201
    return jsonify({"documents": results}), 201


@app.route("/api/pdf/status/<doc_id>", methods=["GET"])
def get_pdf_status(doc_id):
    """Return processing status and inspection metadata for an uploaded PDF."""
    from pdf_intelligence.registry import get_document_registry
    reg = get_document_registry()
    record = reg.get_document(doc_id)
    if not record:
        return jsonify({"error": f"Document '{doc_id}' not found"}), 404
    return jsonify(record.to_summary_dict()), 200


@app.route("/api/pdf/documents", methods=["GET"])
def list_pdf_documents():
    """List all registered PDF documents available for grounded chat."""
    from pdf_intelligence.registry import get_document_registry
    reg = get_document_registry()
    return jsonify({"documents": reg.list_documents()}), 200


@app.route("/api/pdf/<doc_id>", methods=["DELETE"])
def delete_pdf_document(doc_id):
    """Remove an uploaded PDF and its artifacts from registry and disk."""
    from pdf_intelligence.registry import get_document_registry
    reg = get_document_registry()
    deleted = reg.delete_document(doc_id)
    if not deleted:
        return jsonify({"error": f"Document '{doc_id}' not found"}), 404
    return jsonify({"status": "deleted", "doc_id": doc_id}), 200


@app.route("/api/voice/status", methods=["GET"])
def voice_status():
    """Return voice & speech capabilities."""
    return jsonify({
        "ok": True,
        "speech_input": "Browser MediaRecorder + Native Windows Speech Recognition",
        "speech_synthesis": "Web Speech Synthesis API + pyttsx3",
        "pyttsx3_installed": True,
        "transcribe_endpoint": "/api/transcribe"
    })


@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio_endpoint():
    """Transcribe recorded microphone audio via multi-engine speech recognition pipeline."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    import tempfile
    import imageio_ffmpeg
    import speech_recognition as sr

    suffix = os.path.splitext(audio_file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as raw_tmp:
        raw_path = raw_tmp.name
        audio_file.save(raw_path)

    wav_path = raw_path + ".wav"
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        # Convert to 16kHz mono WAV with audio volume boost to capture quiet microphones
        cmd = [
            ffmpeg_exe, "-y", "-i", raw_path,
            "-ar", "16000", "-ac", "1",
            "-af", "volume=2.5",
            "-f", "wav", wav_path
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            return jsonify({"error": "Audio conversion failed"}), 500

        transcribed_text = ""

        # ── Primary Engine: Google Speech Recognition (High accuracy, no API keys)
        try:
            r = sr.Recognizer()
            r.energy_threshold = 250
            r.dynamic_energy_threshold = True

            with sr.AudioFile(wav_path) as source:
                audio_data = r.record(source)

            # Try default English (US)
            try:
                transcribed_text = r.recognize_google(audio_data, language="en-US")
            except sr.UnknownValueError:
                # Try regional English (IN, GB) if accent wasn't matched
                for fallback_lang in ["en-IN", "en-GB"]:
                    try:
                        transcribed_text = r.recognize_google(audio_data, language=fallback_lang)
                        if transcribed_text:
                            break
                    except Exception:
                        pass
            except Exception as g_err:
                print(f"[Voice] Google Speech Recognition error: {g_err}")
        except Exception as e:
            print(f"[Voice] SpeechRecognition engine error: {e}")

        # ── Secondary Fallback: Windows Native System.Speech.Recognition (Offline)
        if not transcribed_text:
            try:
                escaped_wav = wav_path.replace("'", "''")
                ps_code = f"""
                Add-Type -AssemblyName System.Speech
                $eng = New-Object System.Speech.Recognition.SpeechRecognitionEngine
                $eng.SetInputToWaveFile('{escaped_wav}')
                $d = New-Object System.Speech.Recognition.DictationGrammar
                $eng.LoadGrammar($d)
                $out = @()
                while ($true) {{
                    $res = $eng.Recognize([System.TimeSpan]::FromSeconds(2))
                    if ($res -eq $null) {{ break }}
                    if ($res.Text) {{ $out += $res.Text }}
                }}
                if ($out.Count -gt 0) {{ Write-Host ($out -join " ") }}
                """
                ps = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_code],
                    capture_output=True,
                    text=True,
                    creationflags=creationflags,
                    timeout=10
                )
                transcribed_text = ps.stdout.strip()
            except Exception as ps_err:
                print(f"[Voice] Windows native speech error: {ps_err}")

        return jsonify({
            "ok": True,
            "text": transcribed_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [raw_path, wav_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass


if __name__ == "__main__":
    print("\n[AI Assistant] Autonomous Multi-Model Chatbot starting...")
    print("    * General Chat & Vision: Gemma 4 (gemma4:12b)")
    print("    * Software Architect: DeepSeek-Coder-V2 16B")
    print("    * Coding Agent: Qwen2.5-Coder 7B")
    print("    * Voice Input & Read Aloud: Enabled")
    print("    * Aider Pairing Support: Enabled")
    print("    Running on http://127.0.0.1:5000\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
