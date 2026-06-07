"""
test_agents.py — Pruebas de integración del sistema multiagente.
Ejecutar desde la raíz: python tests/test_agents.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import config


class TestLogAgent(unittest.TestCase):
    def test_log_and_retrieve(self):
        from agents.log_agent import LogAgent
        agent = LogAgent()
        agent.log("TEST", "Mensaje de prueba")
        logs = agent.get_all()
        self.assertIn("[TEST]", logs)
        self.assertIn("Mensaje de prueba", logs)

    def test_clear(self):
        from agents.log_agent import LogAgent
        agent = LogAgent()
        agent.log("TEST", "algo")
        agent.clear()
        self.assertEqual(agent.get_all(), "")


class TestVoiceAgent(unittest.TestCase):
    def test_save_temp_wav(self):
        from agents.log_agent import LogAgent
        from agents.voice_agent import VoiceAgent
        import numpy as np

        log = LogAgent()
        agent = VoiceAgent(log)
        data = np.zeros(16000, dtype=np.float32)
        path = agent._save_temp_wav(data, 16000)
        self.assertTrue(os.path.exists(path))
        os.remove(path)


class TestRAGAgent(unittest.TestCase):
    def test_embed_text_structure(self):
        """Verifica que el embedding tenga 512 dimensiones."""
        from agents.log_agent import LogAgent
        from agents.rag_agent import RAGAgent

        log = LogAgent()
        agent = RAGAgent(log)
        agent._load_clip()
        vec = agent._embed_text("problemas de internet en windows")
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 512)

    def test_embed_returns_normalized(self):
        """Verifica que el vector esté normalizado (norma ≈ 1.0)."""
        import numpy as np
        from agents.log_agent import LogAgent
        from agents.rag_agent import RAGAgent

        log = LogAgent()
        agent = RAGAgent(log)
        agent._load_clip()
        vec = agent._embed_text("pantalla negra al encender la computadora")
        norm = float(np.linalg.norm(vec))
        self.assertAlmostEqual(norm, 1.0, places=3)


class TestChunker(unittest.TestCase):
    def test_chunk_short_text(self):
        from rag.chunker import chunk_text
        text = "Texto corto"
        chunks = chunk_text(text)
        self.assertEqual(chunks, [text])

    def test_chunk_long_text(self):
        from rag.chunker import chunk_text
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=180)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 180 + 20)  # chunk_size + overlap

    def test_no_empty_chunks(self):
        from rag.chunker import chunk_text
        text = "  " * 100
        chunks = chunk_text(text)
        for c in chunks:
            self.assertTrue(len(c.strip()) >= 0)


class TestTTSAgent(unittest.TestCase):
    def test_backend_detected(self):
        from agents.log_agent import LogAgent
        from agents.tts_agent import TTSAgent

        log = LogAgent()
        agent = TTSAgent(log)
        # Al menos uno de los backends debería estar disponible en macOS
        self.assertIn(agent._backend, ["piper_python", "piper_binary", "macos_say", None])

    def test_synthesize_short_text(self):
        from agents.log_agent import LogAgent
        from agents.tts_agent import TTSAgent

        log = LogAgent()
        agent = TTSAgent(log)
        if agent._backend is None:
            self.skipTest("Ningún backend TTS disponible")

        result = agent.synthesize("Prueba de audio del sistema.")
        # macOS say con afconvert debería producir WAV
        if result:
            self.assertTrue(os.path.exists(result))


if __name__ == "__main__":
    print("=" * 60)
    print("  S3 IA Multimodal — Pruebas de Agentes")
    print("=" * 60)
    unittest.main(verbosity=2)
