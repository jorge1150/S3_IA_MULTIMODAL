"""
vision_agent.py — Agente de Visión
Responsabilidad: analizar imágenes usando Moondream via Ollama.
Convierte imagen PIL a base64 y obtiene descripción visual del problema.
"""

import base64
import io
import requests
from PIL import Image

from config import OLLAMA_URL, VISION_MODEL, VISION_TIMEOUT, MOONDREAM_PROMPT
from .log_agent import LogAgent


class VisionAgent:
    """
    Agente que usa Moondream (LLM de visión) a través de la API de Ollama.
    Acepta imágenes PIL, rutas de archivo o arrays numpy de Gradio.
    """

    def __init__(self, log_agent: LogAgent):
        self.log = log_agent

    # ── API pública ──────────────────────────────────────────────────────────

    def analyze(self, image_input) -> str:
        """
        Analiza una imagen y retorna descripción del problema técnico.
        image_input: PIL.Image, ruta str, o dict de Gradio {"image": ...}
        """
        try:
            pil_img = self._to_pil(image_input)
            if pil_img is None:
                return ""

            b64 = self._to_base64(pil_img)
            self.log.log("VISION", f"Enviando imagen a Moondream ({VISION_MODEL})...")

            payload = {
                "model": VISION_MODEL,
                "prompt": MOONDREAM_PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"temperature": 0.1},
            }

            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=VISION_TIMEOUT,
            )
            resp.raise_for_status()
            description = resp.json().get("response", "").strip()
            self.log.log("VISION", f"Descripción: «{description[:100]}»")
            return description

        except requests.exceptions.ConnectionError:
            self.log.log("ERROR", "No se puede conectar a Ollama. ¿Está ejecutándose?")
            return "[Ollama no disponible]"
        except requests.exceptions.Timeout:
            self.log.log("ERROR", f"Timeout después de {VISION_TIMEOUT}s esperando Moondream.")
            return "[Timeout en visión]"
        except Exception as exc:
            self.log.log("ERROR", f"Falla en visión: {exc}")
            return ""

    # ── Utilidades privadas ──────────────────────────────────────────────────

    def _to_pil(self, image_input) -> Image.Image | None:
        """Convierte cualquier formato de entrada a PIL Image."""
        if image_input is None:
            return None

        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")

        # Gradio puede devolver numpy array
        try:
            import numpy as np
            if isinstance(image_input, np.ndarray):
                return Image.fromarray(image_input).convert("RGB")
        except ImportError:
            pass

        # Gradio a veces devuelve dict con clave 'composite' o 'background'
        if isinstance(image_input, dict):
            for key in ("composite", "background", "layers"):
                val = image_input.get(key)
                if val is not None:
                    return self._to_pil(val)

        self.log.log("ERROR", f"Formato de imagen no reconocido: {type(image_input)}")
        return None

    def _to_base64(self, img: Image.Image) -> str:
        """Convierte PIL Image a string base64 para la API de Ollama."""
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
