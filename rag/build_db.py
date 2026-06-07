"""
build_db.py — Script ejecutable para construir la base vectorial.

Uso:
    python rag/build_db.py           # ingesta incremental
    python rag/build_db.py --reset   # borra y reinicia la colección
"""

import sys
import os

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingesta import ingest_all_manuals
import config  # asegura que los directorios existen


def main():
    reset = "--reset" in sys.argv

    print("=" * 60)
    print("  S3 IA MULTIMODAL — Construcción de Base Vectorial")
    print("=" * 60)
    print(f"  Manuales: {config.MANUALS_DIR}")
    print(f"  ChromaDB: {config.CHROMA_DB_PATH}")
    print(f"  Colección: {config.CHROMA_COLLECTION}")
    print(f"  Reset: {reset}")
    print("=" * 60)

    n = ingest_all_manuals(reset=reset)

    print("=" * 60)
    print(f"  ✓ Ingesta completada. {n} chunks nuevos insertados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
