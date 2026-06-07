"""
video_agent.py — Agente de Video
Responsabilidad: extraer frames de un video y analizarlos con VisionAgent.
Procesa 1 frame cada VIDEO_FRAME_INTERVAL y agrega las descripciones.
"""

import os
from PIL import Image

from config import VIDEO_FRAME_INTERVAL, VIDEO_MAX_FRAMES
from .log_agent import LogAgent
from .vision_agent import VisionAgent


class VideoAgent:
    """
    Procesa archivos de video extrayendo frames clave y describiéndolos
    con el agente de visión. Usa OpenCV para decodificar el video.
    """

    def __init__(self, log_agent: LogAgent, vision_agent: VisionAgent):
        self.log = log_agent
        self.vision = vision_agent

    # ── API pública ──────────────────────────────────────────────────────────

    def process(self, video_path: str) -> str:
        """
        Extrae frames del video y retorna descripción consolidada.
        video_path: ruta al archivo de video.
        """
        if not video_path or not os.path.exists(str(video_path)):
            return ""

        try:
            import cv2
        except ImportError:
            self.log.log("VIDEO", "OpenCV no instalado. Instala: pip install opencv-python")
            return "[opencv-python no disponible]"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.log.log("ERROR", f"No se puede abrir el video: {video_path}")
            return ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        duration = total_frames / fps
        self.log.log("VIDEO", f"Video: {total_frames} frames, {fps:.0f}fps, {duration:.1f}s")

        descriptions = []
        frame_idx = 0
        frames_analyzed = 0

        while frames_analyzed < VIDEO_MAX_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV usa BGR, convertir a RGB para PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            self.log.log("VIDEO", f"Analizando frame {frame_idx}/{total_frames}...")
            try:
                desc = self.vision.analyze(pil_img)
                if desc and "[" not in desc:
                    descriptions.append(f"Frame {frames_analyzed+1}: {desc}")
            except Exception as exc:
                self.log.log("VIDEO", f"Frame {frame_idx} omitido: {exc}")

            frame_idx += VIDEO_FRAME_INTERVAL
            frames_analyzed += 1

        cap.release()

        if not descriptions:
            return "No se pudo analizar el contenido del video."

        result = " | ".join(descriptions)
        self.log.log("VIDEO", f"Video procesado: {len(descriptions)} frames analizados.")
        return result
