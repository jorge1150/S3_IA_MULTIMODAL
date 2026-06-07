"""
tts_agent.py — Agente TTS (Text-to-Speech)
Responsabilidad: sintetizar la respuesta en voz española.

Estrategia de fallback (en orden):
  1. Piper TTS (piper-tts Python package) con modelo es_ES-sharvard-medium
  2. Piper TTS binario del sistema
  3. macOS 'say' command (fallback nativo, sin descargar nada)
"""

import os
import subprocess
import threading
import wave
from config import (
    PIPER_MODEL_PATH, PIPER_CONFIG_PATH, TTS_SAMPLE_RATE, TEMP_DIR,
)
from .log_agent import LogAgent


class TTSAgent:
    """
    Agente de síntesis de voz con tres backends en cascada.
    La respuesta de audio se guarda en un archivo WAV temporal
    para que Gradio lo reproduzca automáticamente.
    """

    def __init__(self, log_agent: LogAgent):
        self.log = log_agent
        self._piper_voice = None
        self._backend: str | None = None
        self._detect_backend()

    # ── Detección de backend disponible ─────────────────────────────────────

    def _detect_backend(self):
        """Detecta el mejor backend disponible al inicializar."""
        # 1. piper-tts Python package
        if os.path.exists(PIPER_MODEL_PATH):
            try:
                from piper import PiperVoice   # noqa: F401
                self._backend = "piper_python"
                self.log.log("TTS", "Backend: piper-tts (Python package)")
                return
            except ImportError:
                pass

        # 2. piper binario en PATH
        if self._command_exists("piper"):
            self._backend = "piper_binary"
            self.log.log("TTS", "Backend: piper (binario del sistema)")
            return

        # 3. macOS say (siempre disponible en macOS)
        if self._command_exists("say"):
            self._backend = "macos_say"
            self.log.log("TTS", "Backend: macOS say (fallback nativo)")
            return

        self.log.log("TTS", "ADVERTENCIA: no se encontró motor TTS. No habrá audio.")
        self._backend = None

    # ── API pública ──────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> str | None:
        """
        Sintetiza texto a voz.
        Retorna ruta al archivo WAV, o None si falla.
        """
        if not text or not text.strip():
            return None

        output_path = os.path.join(TEMP_DIR, "response_audio.wav")
        # Limitar longitud para TTS eficiente
        text_clean = text[:800].strip()

        self.log.log("TTS", f"Sintetizando {len(text_clean)} caracteres...")

        success = False

        if self._backend == "piper_python":
            success = self._synth_piper_python(text_clean, output_path)
        elif self._backend == "piper_binary":
            success = self._synth_piper_binary(text_clean, output_path)
        elif self._backend == "macos_say":
            success = self._synth_macos_say(text_clean, output_path)

        if success and os.path.exists(output_path):
            self.log.log("TTS", f"Audio guardado: {output_path}")
            return output_path

        self.log.log("TTS", "No se pudo generar audio.")
        return None

    def play_async(self, audio_path: str):
        """Reproduce el audio en hilo secundario para no bloquear la UI."""
        if not audio_path:
            return
        t = threading.Thread(target=self._play, args=(audio_path,), daemon=True)
        t.start()

    # ── Backends privados ────────────────────────────────────────────────────

    def _synth_piper_python(self, text: str, output_path: str) -> bool:
        """Usa el paquete piper-tts de Python (API 1.4+: generador de AudioChunk)."""
        try:
            from piper import PiperVoice
            if self._piper_voice is None:
                self._piper_voice = PiperVoice.load(
                    PIPER_MODEL_PATH,
                    config_path=PIPER_CONFIG_PATH,
                    use_cuda=False,
                )
            # piper-tts 1.4+ devuelve un generador de AudioChunk
            chunks = list(self._piper_voice.synthesize(text))
            if not chunks:
                return False
            first = chunks[0]
            with wave.open(output_path, "w") as wf:
                wf.setnchannels(first.sample_channels)
                wf.setsampwidth(first.sample_width)
                wf.setframerate(first.sample_rate)
                for chunk in chunks:
                    wf.writeframes(chunk.audio_int16_bytes)
            return True
        except Exception as exc:
            self.log.log("TTS", f"piper-tts Python falló: {exc}")
            return False

    def _synth_piper_binary(self, text: str, output_path: str) -> bool:
        """Usa el binario 'piper' instalado en el sistema."""
        try:
            result = subprocess.run(
                ["piper", "--model", PIPER_MODEL_PATH, "--output_file", output_path],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as exc:
            self.log.log("TTS", f"piper binario falló: {exc}")
            return False

    def _synth_macos_say(self, text: str, output_path: str) -> bool:
        """
        Usa el comando 'say' nativo de macOS.
        Genera AIFF y lo convierte a WAV con afconvert.
        """
        try:
            aiff_path = output_path.replace(".wav", ".aiff")
            # Generar AIFF con voz Monica (español latinoamericano)
            subprocess.run(
                ["say", "-v", "Monica", "-o", aiff_path, text],
                timeout=30,
                check=True,
            )
            # Convertir AIFF → WAV
            if os.path.exists(aiff_path):
                subprocess.run(
                    ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff_path, output_path],
                    timeout=10,
                    check=True,
                )
                os.remove(aiff_path)
                return True
        except Exception as exc:
            self.log.log("TTS", f"macOS say falló: {exc}. Intentando sin conversión...")
            # Fallback: reproducir AIFF directamente con afplay y no retornar WAV
            try:
                subprocess.run(["say", "-v", "Monica", text], timeout=30)
                return False  # No hay archivo WAV para Gradio
            except Exception:
                pass
        return False

    def _play(self, audio_path: str):
        """Reproduce WAV en macOS con afplay."""
        try:
            subprocess.run(["afplay", audio_path], timeout=60)
        except Exception:
            pass

    # ── Utilidades ──────────────────────────────────────────────────────────

    @staticmethod
    def _command_exists(cmd: str) -> bool:
        """Verifica si un comando está disponible en el PATH del sistema."""
        try:
            subprocess.run(
                ["which", cmd],
                capture_output=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False
