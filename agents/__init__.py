"""Paquete de agentes del sistema S3 IA Multimodal."""
from .log_agent import LogAgent
from .voice_agent import VoiceAgent
from .vision_agent import VisionAgent
from .video_agent import VideoAgent
from .rag_agent import RAGAgent
from .response_agent import ResponseAgent
from .tts_agent import TTSAgent
from .coordinator import CoordinatorAgent

__all__ = [
    "LogAgent",
    "VoiceAgent",
    "VisionAgent",
    "VideoAgent",
    "RAGAgent",
    "ResponseAgent",
    "TTSAgent",
    "CoordinatorAgent",
]
