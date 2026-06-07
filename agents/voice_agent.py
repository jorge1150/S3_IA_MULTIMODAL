"""
voice_agent.py — Agente de Voz (STT)
Responsabilidad: convertir audio (micrófono o archivo) a texto en español.
Usa faster-whisper con cuantización int8 para máxima velocidad en CPU.
"""

import os
import wave
import tempfile
import numpy as np
import sounddevice as sd
import scipy.signal

from config import (
    DEVICE, WHISPER_MODEL_SIZE, WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE, WHISPER_LANGUAGE,
    SAMPLE_RATE_RECORD, SAMPLE_RATE_WHISPER,
    RECORD_DURATION, AUDIO_SAFETY_CEILING,
    MIC_DEVICE_INDEX, TEMP_DIR,
)
from .log_agent import LogAgent


class VoiceAgent:
    """
    Agente STT basado en faster-whisper.
    Soporta dos modos:
      - transcribe(audio_path): transcribe archivo existente (Gradio)
      - record_and_transcribe(): graba 5 seg del micrófono y transcribe
    """

    def __init__(self, log_agent: LogAgent):
        self.log = log_agent
        self._model = None

    # ── Carga lazy del modelo ────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return
        self.log.log("STT", f"Cargando Whisper '{WHISPER_MODEL_SIZE}' en {DEVICE} ({WHISPER_COMPUTE_TYPE})...")
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device=DEVICE if DEVICE in ("cuda", "cpu") else "cpu",
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        self.log.log("STT", "Modelo Whisper listo.")

    # ── API pública ──────────────────────────────────────────────────────────

    def transcribe(self, audio_input) -> str:
        """
        Transcribe un archivo de audio.
        audio_input: ruta de archivo (str) o array numpy (sample_rate, data).
        Retorna texto transcrito en español.
        """
        self._load_model()

        # Gradio devuelve (sample_rate, numpy_array) o ruta de string
        if isinstance(audio_input, tuple):
            sr, data = audio_input
            audio_path = self._save_temp_wav(data, sr)
        elif isinstance(audio_input, (str, os.PathLike)) and os.path.exists(str(audio_input)):
            audio_path = str(audio_input)
        else:
            self.log.log("STT", "Entrada de audio inválida.")
            return ""

        self.log.log("STT", f"Transcribiendo: {os.path.basename(str(audio_path))}")

        segments, info = self._model.transcribe(
            audio_path,
            beam_size=WHISPER_BEAM_SIZE,
            language=WHISPER_LANGUAGE,
        )
        text = " ".join(s.text for s in segments).strip()
        self.log.log("STT", f"Transcripción ({info.language}): «{text[:80]}»")
        return text

    def record_and_transcribe(self) -> str:
        """
        Graba desde el micrófono del sistema durante RECORD_DURATION segundos,
        remuestrea de 48kHz a 16kHz y transcribe.
        """
        self.log.log("STT", f"Escuchando {RECORD_DURATION} segundos...")
        try:
            grabacion = sd.rec(
                int(RECORD_DURATION * SAMPLE_RATE_RECORD),
                samplerate=SAMPLE_RATE_RECORD,
                channels=1,
                dtype="float32",
                device=MIC_DEVICE_INDEX,
            )
            sd.wait()
        except Exception as exc:
            self.log.log("ERROR", f"Falla al grabar audio: {exc}")
            return ""

        # Remuestreo 48kHz → 16kHz (requerido por Whisper)
        audio_flat = grabacion.flatten()
        num_samples = int(len(audio_flat) * SAMPLE_RATE_WHISPER / SAMPLE_RATE_RECORD)
        audio_resampled = scipy.signal.resample(audio_flat, num_samples)

        # Normalización con techo de seguridad
        peak = np.max(np.abs(audio_resampled))
        if peak > 0:
            audio_resampled = audio_resampled / peak * AUDIO_SAFETY_CEILING

        audio_path = self._save_temp_wav(audio_resampled, SAMPLE_RATE_WHISPER)
        return self.transcribe(audio_path)

    # ── Utilidades privadas ──────────────────────────────────────────────────

    def _save_temp_wav(self, data: np.ndarray, sample_rate: int) -> str:
        """Guarda array numpy como WAV temporal y retorna la ruta."""
        path = os.path.join(TEMP_DIR, "audio_input.wav")
        data_int16 = (data * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)      # 16 bits = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(data_int16.tobytes())
        return path
