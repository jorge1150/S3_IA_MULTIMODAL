"""
log_agent.py — Agente de Trazabilidad y Logs
Responsabilidad: registrar cada etapa del pipeline con timestamp,
nivel y etiqueta de stage. Thread-safe para uso concurrente.
"""

import threading
import logging
import os
from datetime import datetime
from config import LOGS_DIR

# Logger estándar de Python para archivo en disco
logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "system.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_file_logger = logging.getLogger("s3_multimodal")


class LogAgent:
    """
    Agente de logs con dos destinos:
      1. Lista en memoria (para mostrar en Gradio en tiempo real).
      2. Archivo en disco (para auditoría).
    """

    # Emojis por stage para la consola visual
    _STAGE_ICONS = {
        "INICIO":       "🚀",
        "STT":          "🎤",
        "VISION":       "👁️",
        "VIDEO":        "🎬",
        "RAG":          "📚",
        "DIAGNOSTICO":  "🔍",
        "RESPUESTA":    "💡",
        "TTS":          "🔊",
        "FIN":          "✅",
        "ERROR":        "❌",
        "INFO":         "ℹ️",
    }

    def __init__(self):
        self._entries: list[str] = []
        self._lock = threading.Lock()

    # ── API pública ──────────────────────────────────────────────────────────

    def log(self, stage: str, message: str) -> str:
        """Registra una entrada y la devuelve formateada."""
        ts = datetime.now().strftime("%H:%M:%S")
        icon = self._STAGE_ICONS.get(stage.upper(), "•")
        entry = f"[{ts}] {icon} [{stage}] {message}"
        with self._lock:
            self._entries.append(entry)
        _file_logger.info("[%s] %s", stage, message)
        return entry

    def get_all(self) -> str:
        """Devuelve todos los logs concatenados (para Gradio Textbox)."""
        with self._lock:
            return "\n".join(self._entries)

    def clear(self) -> None:
        """Limpia el historial para una nueva consulta."""
        with self._lock:
            self._entries.clear()

    def last(self) -> str:
        """Devuelve la última entrada registrada."""
        with self._lock:
            return self._entries[-1] if self._entries else ""
