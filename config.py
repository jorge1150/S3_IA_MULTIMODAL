"""
config.py — Configuración central del proyecto S3 IA Multimodal
Toda constante del sistema vive aquí.
"""

import os
import torch

# ─────────────────────────────────────────────
# DISPOSITIVO DE CÓMPUTO
# macOS Intel Core i9: sin CUDA → CPU siempre
# ─────────────────────────────────────────────
def _detect_device() -> str:
    import platform
    if torch.cuda.is_available():
        return "cuda"
    # MPS solo en Apple Silicon (arm64); en Intel x86_64 es inestable
    # con threading de Gradio y el ONNX Runtime de piper-tts → SIGSEGV
    if (platform.machine() == "arm64"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

DEVICE: str = _detect_device()

# Tipo de cómputo para faster-whisper
WHISPER_COMPUTE_TYPE: str = "float16" if DEVICE == "cuda" else "int8"

# ─────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────
OLLAMA_HOST: str = "localhost"
OLLAMA_PORT: int = 11434
OLLAMA_URL: str = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_TIMEOUT: int = 120          # segundos

LLM_MODEL: str = "tinyllama"       # Modelo de razonamiento
VISION_MODEL: str = "moondream"    # Modelo de visión
LLM_TEMPERATURE: float = 0.1      # Respuestas deterministas

# ─────────────────────────────────────────────
# WHISPER STT (faster-whisper)
# ─────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = "base"
WHISPER_BEAM_SIZE: int = 5
WHISPER_LANGUAGE: str = "es"

# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
SAMPLE_RATE_RECORD: int = 48000    # Hz — grabación por sounddevice
SAMPLE_RATE_WHISPER: int = 16000   # Hz — requerido por Whisper
RECORD_DURATION: int = 5           # segundos de escucha por ciclo
AUDIO_SAFETY_CEILING: float = 0.9  # techo de normalización
MIC_DEVICE_INDEX = None            # None = dispositivo del sistema por defecto
OUTPUT_DEVICE_INDEX = None         # None = dispositivo de salida por defecto

# ─────────────────────────────────────────────
# OPENCLIP — Embeddings para RAG
# Usamos el modelo vía HuggingFace Hub (ya cacheado en ~/.cache/huggingface)
# en lugar del CDN de OpenAI (~338MB descarga extra).
# ─────────────────────────────────────────────
CLIP_MODEL: str = "hf-hub:timm/vit_base_patch32_clip_224.openai"
CLIP_PRETRAINED: str = ""          # ignorado cuando el nombre incluye hf-hub:
CLIP_EMBEDDING_DIM: int = 512
CLIP_MAX_TOKENS: int = 200         # truncar texto antes del tokenizador

# ─────────────────────────────────────────────
# CHROMADB — Base vectorial
# ─────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH: str = os.path.join(BASE_DIR, "vector_db", "db_multimodal_s3")
CHROMA_COLLECTION: str = "manual_tecnico"
RAG_TOP_K: int = 3                 # número de chunks a recuperar
RAG_MIN_SIMILARITY: float = 0.20  # umbral de relevancia

# ─────────────────────────────────────────────
# PIPER TTS
# ─────────────────────────────────────────────
PIPER_MODELS_DIR: str = os.path.join(BASE_DIR, "audio", "piper_models")
PIPER_MODEL_NAME: str = "es_ES-sharvard-medium"
PIPER_MODEL_PATH: str = os.path.join(PIPER_MODELS_DIR, "es_ES-sharvard-medium.onnx")
PIPER_CONFIG_PATH: str = os.path.join(PIPER_MODELS_DIR, "es_ES-sharvard-medium.onnx.json")
TTS_SAMPLE_RATE: int = 22050

# ─────────────────────────────────────────────
# VISIÓN
# ─────────────────────────────────────────────
VISION_TIMEOUT: int = 60
MOONDREAM_PROMPT: str = (
    "Describe brevemente en español qué problema de computadora muestra esta imagen. "
    "Máximo 20 palabras."
)

# ─────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────
VIDEO_FRAME_INTERVAL: int = 30     # procesar 1 frame de cada N
VIDEO_MAX_FRAMES: int = 2          # máximo de frames a analizar (Moondream ~60s/frame en CPU)

# ─────────────────────────────────────────────
# RUTAS DEL SISTEMA
# ─────────────────────────────────────────────
MANUALS_DIR: str = os.path.join(BASE_DIR, "manuals")
LOGS_DIR: str = os.path.join(BASE_DIR, "logs")
TEMP_DIR: str = os.path.join(BASE_DIR, "temp")

# ─────────────────────────────────────────────
# GRADIO
# ─────────────────────────────────────────────
GRADIO_PORT: int = 7864
GRADIO_SERVER: str = "0.0.0.0"
GRADIO_TITLE: str = "IA Multimodal - Soporte Técnico Computadoras Local"

# ─────────────────────────────────────────────
# CREAR DIRECTORIOS NECESARIOS EN IMPORTACIÓN
# ─────────────────────────────────────────────
for _d in [LOGS_DIR, TEMP_DIR, PIPER_MODELS_DIR, os.path.dirname(CHROMA_DB_PATH)]:
    os.makedirs(_d, exist_ok=True)
