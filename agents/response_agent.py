"""
response_agent.py — Agente de Respuesta (+ Diagnóstico)
Responsabilidad: construir el prompt multimodal combinando texto del usuario,
contexto RAG y descripción visual, luego consultar TinyLlama vía Ollama.
"""

import requests

from config import (
    OLLAMA_URL, LLM_MODEL, LLM_TEMPERATURE, OLLAMA_TIMEOUT,
)
from .log_agent import LogAgent


# Prompt del sistema: responde en español al problema real del usuario
_SYSTEM_PROMPT = (
    "Eres un experto en soporte técnico de computadoras. "
    "Responde SIEMPRE en español. "
    "Si hay una descripción de imagen o pantalla, úsala como base principal del diagnóstico. "
    "Responde EXACTAMENTE al problema visual o descrito por el usuario, NO inventes un problema diferente. "
    "Da instrucciones claras, numeradas y paso a paso para resolver el problema específico. "
    "Usa el manual técnico solo como referencia complementaria."
)


class ResponseAgent:
    """
    Genera la respuesta técnica final combinando todas las modalidades:
      - Consulta del usuario (texto o voz transcrita)
      - Contexto del manual técnico (RAG)
      - Descripción visual de la imagen/video
    """

    def __init__(self, log_agent: LogAgent):
        self.log = log_agent

    # ── API pública ──────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        rag_context: list[dict],
        visual_description: str = "",
    ) -> str:
        """
        Genera la respuesta técnica.

        Args:
            query: pregunta o problema del usuario.
            rag_context: lista de chunks RAG [{'text':..., 'similarity':...}].
            visual_description: descripción de imagen/video (puede estar vacía).

        Returns:
            Respuesta técnica en español.
        """
        self.log.log("DIAGNOSTICO", "Enviando contexto a TinyLlama...")
        self.log.log("DIAGNOSTICO", f"Modelo: {LLM_MODEL} | Temp: {LLM_TEMPERATURE}")

        try:
            user_msg = self._build_user_message(query, rag_context, visual_description)
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system",    "content": _SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": "En español, paso a paso:\n"},
                ],
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": 512,
                },
            }
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "").strip()
            self.log.log("RESPUESTA", f"Respuesta generada ({len(answer)} caracteres).")
            return answer

        except requests.exceptions.ConnectionError:
            msg = "Error: Ollama no está ejecutándose. Inicia Ollama y ejecuta: ollama pull tinyllama"
            self.log.log("ERROR", msg)
            return msg
        except requests.exceptions.Timeout:
            msg = f"Error: Timeout después de {OLLAMA_TIMEOUT}s. El modelo tardó demasiado."
            self.log.log("ERROR", msg)
            return msg
        except Exception as exc:
            msg = f"Error al generar respuesta: {exc}"
            self.log.log("ERROR", msg)
            return msg

    # ── Utilidades privadas ──────────────────────────────────────────────────

    def _build_user_message(
        self,
        query: str,
        rag_context: list[dict],
        visual_description: str,
    ) -> str:
        """Construye el mensaje de usuario para /api/chat."""
        parts = []

        # Visual va primero — tiene mayor peso para el diagnóstico
        if visual_description and "[" not in visual_description:
            parts.append(f"Lo que se muestra en pantalla: {visual_description}")

        parts.append(f"Problema reportado: {query}")

        if rag_context:
            refs = " | ".join(c["text"][:300] for c in rag_context[:2])
            parts.append(f"Referencia del manual técnico: {refs}")

        parts.append("Basándote en lo que se muestra en pantalla, da la solución paso a paso en español.")

        return "\n".join(parts)
