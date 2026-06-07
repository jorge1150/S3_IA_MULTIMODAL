"""
capture.py — Captura de pantalla y webcam.
Funciones utilitarias para capturar imágenes fuera de Gradio.
"""

import os
import subprocess
from PIL import Image
from config import TEMP_DIR


def capture_screenshot() -> Image.Image | None:
    """
    Captura la pantalla completa en macOS.
    Usa screencapture (nativo de macOS) para máxima compatibilidad.
    """
    path = os.path.join(TEMP_DIR, "screenshot.png")
    try:
        # -x: sin sonido de cámara
        subprocess.run(["screencapture", "-x", path], check=True, timeout=10)
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"[CAPTURA] Error en screenshot: {e}")
        # Fallback: PIL ImageGrab
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(path)
            return img
        except Exception:
            return None


def capture_webcam(device_index: int = 0) -> Image.Image | None:
    """
    Captura un frame de la webcam con OpenCV.
    Retorna imagen PIL o None si falla.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except ImportError:
        print("[CAPTURA] opencv-python no instalado.")
        return None
    except Exception as e:
        print(f"[CAPTURA] Error webcam: {e}")
        return None
