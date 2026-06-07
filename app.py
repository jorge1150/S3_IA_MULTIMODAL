"""
app.py — Punto de entrada principal del sistema S3 IA Multimodal.

Uso:
    python app.py

El sistema arranca en http://localhost:7864
"""

import sys
import os
import faulthandler
faulthandler.enable()   # imprime stack C cuando hay SIGSEGV

# Evita SIGABRT: PyTorch y ONNX (piper-tts) cargan libiomp5.dylib por separado
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Limita OpenMP a 1 hilo — evita conflicto ctranslate2 (Whisper) + torch en macOS Intel
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Deshabilita telemetría de Gradio (requests externos que pueden causar crashes)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Asegurar que el directorio raíz esté en el Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # crea directorios necesarios al importar


def check_ollama() -> bool:
    """Verifica que Ollama esté ejecutándose antes de arrancar."""
    import requests
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"[OK] Ollama activo. Modelos: {models}")

        missing = []
        for model in [config.LLM_MODEL, config.VISION_MODEL]:
            if not any(model in m for m in models):
                missing.append(model)

        if missing:
            print(f"\n[ADVERTENCIA] Modelos faltantes: {missing}")
            for m in missing:
                print(f"  Instalar con: ollama pull {m}")
            print()
        return True
    except Exception:
        print(f"[ERROR] Ollama no está ejecutándose en {config.OLLAMA_URL}")
        print("  Inicialo con: ollama serve")
        print("  O abre la aplicación Ollama.")
        return False


def check_vector_db() -> bool:
    """Verifica que la base vectorial tenga datos."""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        col = client.get_collection(config.CHROMA_COLLECTION)
        count = col.count()
        print(f"[OK] ChromaDB lista: {count} documentos en '{config.CHROMA_COLLECTION}'")
        return count > 0
    except Exception:
        print(f"[ADVERTENCIA] Base vectorial vacía o inexistente.")
        print(f"  Ejecuta: python rag/build_db.py")
        return False


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║    IA MULTIMODAL - SOPORTE TÉCNICO COMPUTADORAS LOCAL    ║
║    Ollama + ChromaDB + Whisper + Moondream + Piper TTS   ║
╠══════════════════════════════════════════════════════════╣
║  Maestría IA Aplicada — UIsrael                          ║
║  Docente: MSc. Ing. Pablo Recalde                        ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()
    print(f"[INFO] Dispositivo de cómputo: {config.DEVICE}")
    print(f"[INFO] Puerto Gradio: {config.GRADIO_PORT}")
    print()

    # Verificaciones previas
    ollama_ok = check_ollama()
    db_ok = check_vector_db()
    print()

    if not db_ok:
        print("[INFO] Construyendo base vectorial desde los manuales...")
        from rag.ingesta import ingest_all_manuals
        n = ingest_all_manuals()
        print(f"[OK] Base vectorial construida: {n} chunks.\n")

    # Inicializar el coordinador (carga lazy de modelos)
    print("[INFO] Inicializando agentes...")
    from agents.coordinator import CoordinatorAgent
    coordinator = CoordinatorAgent()
    print("[OK] Sistema multiagente listo.\n")

    # Construir y lanzar la interfaz Gradio
    from ui.interface import build_interface
    demo = build_interface(coordinator)

    print(f"[INFO] Iniciando interfaz en http://localhost:{config.GRADIO_PORT}")
    print("[INFO] Presiona Ctrl+C para detener el sistema.\n")

    # Gradio 6.x: queue está habilitada por defecto, no llamar .queue() explícitamente
    from ui.interface import GRADIO_THEME
    demo.launch(
        server_name=config.GRADIO_SERVER,
        server_port=config.GRADIO_PORT,
        share=False,
        inbrowser=False,
        theme=GRADIO_THEME,
    )


if __name__ == "__main__":
    main()
