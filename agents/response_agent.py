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
    "Responde EXACTAMENTE al problema que describe el usuario. "
    "Da instrucciones claras, numeradas y paso a paso. "
    "Usa el contexto del manual como referencia si es relevante, "
    "pero responde basándote en el problema real del usuario."
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
        prompt = self._build_prompt(query, rag_context, visual_description)
        self.log.log("DIAGNOSTICO", "Enviando contexto a TinyLlama...")
        self.log.log("DIAGNOSTICO", f"Modelo: {LLM_MODEL} | Temp: {LLM_TEMPERATURE}")

        try:
            payload = {
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": 512,
                },
            }
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json().get("response", "").strip()
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

    def _build_prompt(
        self,
        query: str,
        rag_context: list[dict],
        visual_description: str,
    ) -> str:
        """
        Construye el prompt estructurado para TinyLlama.
        Formato simple para evitar que modelos pequeños repitan las instrucciones.
        """
        parts = [f"Sistema: {_SYSTEM_PROMPT}"]

        # La pregunta del usuario va primero para que el LLM la atienda
        parts.append(f"Problema del usuario: {query}")

        # Contexto del manual técnico (RAG) como referencia secundaria
        if rag_context:
            manual_text = " | ".join(c["text"] for c in rag_context)
            parts.append(f"Información de referencia del manual técnico: {manual_text}")

        # Descripción visual si la hay
        if visual_description and "[" not in visual_description:
            parts.append(f"Descripción de imagen: {visual_description}")

        parts.append(f"Responde paso a paso cómo resolver: {query}")

        return "\n".join(parts)
