"""
test_rag.py — Pruebas del sistema RAG: ingesta, chunking y recuperación.
Requiere que la base vectorial esté construida: python rag/build_db.py
Ejecutar: python tests/test_rag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import config


class TestIngesta(unittest.TestCase):
    def test_manuals_directory_exists(self):
        self.assertTrue(os.path.isdir(config.MANUALS_DIR))

    def test_manuals_have_content(self):
        import glob
        files = glob.glob(os.path.join(config.MANUALS_DIR, "*.txt"))
        self.assertGreater(len(files), 0, "No hay archivos .txt en manuals/")
        for f in files:
            size = os.path.getsize(f)
            self.assertGreater(size, 100, f"{f} está vacío o muy pequeño")


class TestChromaDB(unittest.TestCase):
    def test_collection_exists(self):
        import chromadb
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        try:
            col = client.get_collection(config.CHROMA_COLLECTION)
            count = col.count()
            self.assertGreater(count, 0, "La colección está vacía. Ejecuta: python rag/build_db.py")
            print(f"\n  ✓ ChromaDB: {count} documentos en '{config.CHROMA_COLLECTION}'")
        except Exception as e:
            self.fail(f"Colección '{config.CHROMA_COLLECTION}' no existe: {e}")


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        from agents.log_agent import LogAgent
        from agents.rag_agent import RAGAgent
        self.log = LogAgent()
        self.rag = RAGAgent(self.log)

    def test_retrieve_internet_problem(self):
        results = self.rag.retrieve("mi computadora no tiene internet")
        self.assertIsInstance(results, list)
        if results:  # puede estar vacío si la DB no existe
            self.assertIn("text", results[0])
            self.assertIn("similarity", results[0])

    def test_retrieve_slow_computer(self):
        results = self.rag.retrieve("computadora muy lenta al arrancar")
        if results:
            # El resultado más relevante debería tener similitud > umbral
            self.assertGreater(results[0]["similarity"], config.RAG_MIN_SIMILARITY)

    def test_retrieve_virus(self):
        results = self.rag.retrieve("creo que tengo un virus en mi computadora")
        if results:
            self.assertTrue(any(
                "virus" in r["text"].lower() or "malware" in r["text"].lower()
                for r in results
            ))

    def test_similarity_scores_in_range(self):
        results = self.rag.retrieve("problemas de audio sin sonido")
        for r in results:
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.1)  # pequeña tolerancia


if __name__ == "__main__":
    print("=" * 60)
    print("  S3 IA Multimodal — Pruebas del Sistema RAG")
    print("=" * 60)
    unittest.main(verbosity=2)
