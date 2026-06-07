"""
coordinator.py — Agente Coordinador (Orquestador)
Responsabilidad: dirigir el pipeline completo de forma secuencial,
emitir actualizaciones de log en tiempo real hacia la UI de Gradio,
y garantizar que cada agente reciba los datos correctos.

Pipeline:
  [INICIO] → [STT] → [VISION] → [VIDEO] → [RAG] → [DIAGNOSTICO] → [TTS] → [FIN]
"""

from typing import Generator

from .log_agent import LogAgent
from .voice_agent import VoiceAgent
from .vision_agent import VisionAgent
from .video_agent import VideoAgent
from .rag_agent import RAGAgent
from .response_agent import ResponseAgent
from .tts_agent import TTSAgent


# Tipo del yield: (stt_text, response, audio_path, logs_str)
_Update = tuple[str, str, str | None, str]


class CoordinatorAgent:
    """
    Orquestador multiagente del sistema S3 IA Multimodal.
    Se instancia UNA sola vez al arrancar app.py (singleton).
    Todos los agentes se inicializan aquí con modelos compartidos.
    """

    def __init__(self):
        # Agente de logs compartido por todos
        self.log_agent = LogAgent()

        # Instanciar agentes — los modelos se cargan de forma lazy al primer uso
        self.voice_agent = VoiceAgent(self.log_agent)
        self.vision_agent = VisionAgent(self.log_agent)
        self.video_agent = VideoAgent(self.log_agent, self.vision_agent)
        self.rag_agent = RAGAgent(self.log_agent)
        self.response_agent = ResponseAgent(self.log_agent)
        self.tts_agent = TTSAgent(self.log_agent)

    # ── API pública ──────────────────────────────────────────────────────────

    def process(
        self,
        image_input,
        audio_input,
        text_input: str,
        video_input,
    ) -> Generator[_Update, None, None]:
        """
        Generador que ejecuta el pipeline y hace yield de actualizaciones
        parciales para que Gradio muestre el progreso en tiempo real.

        Yield: (stt_text, response, audio_path, logs)
        """
        self.log_agent.clear()
        stt_text = ""
        response = ""

        # ── [INICIO] ────────────────────────────────────────────────────────
        self.log_agent.log("INICIO", "Sistema S3 IA Multimodal arrancado.")
        yield stt_text, response, None, self.log_agent.get_all()

        # ── [STT] — Voz a texto ─────────────────────────────────────────────
        if audio_input is not None:
            self.log_agent.log("STT", "Procesando entrada de audio...")
            yield stt_text, response, None, self.log_agent.get_all()

            stt_text = self.voice_agent.transcribe(audio_input)
            if stt_text:
                self.log_agent.log("STT", f"✓ Transcripción completa: «{stt_text[:60]}...»")
            else:
                self.log_agent.log("STT", "No se detectó voz en el audio.")
            yield stt_text, response, None, self.log_agent.get_all()

        # Combinar texto escrito + texto de voz
        full_query = " ".join(
            filter(None, [str(text_input or "").strip(), stt_text.strip()])
        ).strip()

        if not full_query and image_input is None and video_input is None:
            self.log_agent.log("ERROR", "No hay entrada: proporcione texto, voz o imagen.")
            yield stt_text, "Por favor, proporcione al menos texto, voz o una imagen.", None, self.log_agent.get_all()
            return

        # Si solo hay imagen/video, usar descripción genérica como query
        if not full_query:
            full_query = "analiza el problema que se muestra y da solución"

        # ── [VISION] — Análisis de imagen ───────────────────────────────────
        visual_description = ""
        if image_input is not None:
            self.log_agent.log("VISION", "Analizando imagen con Moondream...")
            yield stt_text, response, None, self.log_agent.get_all()

            visual_description = self.vision_agent.analyze(image_input)
            self.log_agent.log("VISION", f"✓ Descripción: «{visual_description[:80]}»")
            yield stt_text, response, None, self.log_agent.get_all()

        # ── [VIDEO] — Análisis de video ──────────────────────────────────────
        video_description = ""
        if video_input is not None:
            self.log_agent.log("VIDEO", "Procesando frames del video...")
            yield stt_text, response, None, self.log_agent.get_all()

            video_description = self.video_agent.process(str(video_input))
            self.log_agent.log("VIDEO", f"✓ Video analizado: {video_description[:60]}...")
            yield stt_text, response, None, self.log_agent.get_all()

        all_visual = " | ".join(filter(None, [visual_description, video_description]))

        # ── [RAG] — Búsqueda en manual técnico ──────────────────────────────
        # Si hay descripción visual, combinarla con la query para que el RAG
        # busque el manual correcto (ej: BSOD → pantalla_azul, no audio)
        rag_query = full_query
        if all_visual and "[" not in all_visual:
            rag_query = f"{full_query} {all_visual}"

        self.log_agent.log("RAG", f"Buscando soluciones para: «{rag_query[:80]}»...")
        yield stt_text, response, None, self.log_agent.get_all()

        rag_chunks = self.rag_agent.retrieve(rag_query)
        n_chunks = len(rag_chunks)
        if n_chunks:
            self.log_agent.log("RAG", f"✓ {n_chunks} fragmento(s) relevante(s) encontrados.")
        else:
            self.log_agent.log("RAG", "⚠ Sin contexto en manual. Usando conocimiento general.")
        yield stt_text, response, None, self.log_agent.get_all()

        # ── [DIAGNOSTICO] — Generación de respuesta ──────────────────────────
        self.log_agent.log("DIAGNOSTICO", "Generando diagnóstico con TinyLlama...")
        yield stt_text, response, None, self.log_agent.get_all()

        response = self.response_agent.generate(
            query=full_query,
            rag_context=rag_chunks,
            visual_description=all_visual,
        )
        self.log_agent.log("RESPUESTA", f"✓ Respuesta lista ({len(response)} caracteres).")
        yield stt_text, response, None, self.log_agent.get_all()

        # ── [TTS] — Síntesis de voz ──────────────────────────────────────────
        self.log_agent.log("TTS", "Sintetizando respuesta en voz...")
        yield stt_text, response, None, self.log_agent.get_all()

        audio_path = self.tts_agent.synthesize(response)
        if audio_path:
            self.log_agent.log("TTS", f"✓ Audio generado: {audio_path}")
        else:
            self.log_agent.log("TTS", "⚠ No se generó audio (ver logs de TTS).")

        # ── [FIN] ────────────────────────────────────────────────────────────
        self.log_agent.log("FIN", "Proceso completado correctamente.")
        yield stt_text, response, audio_path, self.log_agent.get_all()
