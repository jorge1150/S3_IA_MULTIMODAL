"""
ingesta.py — Módulo de ingesta de documentos a ChromaDB.
Carga manuales de texto, los fragmenta con chunker.py,
vectoriza con OpenCLIP y almacena en ChromaDB.
"""

import os
import glob
import torch
import open_clip
import chromadb

from config import (
    DEVICE, CLIP_MODEL, CLIP_PRETRAINED, CLIP_MAX_TOKENS,
    CHROMA_DB_PATH, CHROMA_COLLECTION, MANUALS_DIR,
)
from .chunker import chunk_document


def get_clip_embedder():
    """Carga OpenCLIP ViT-B-32 para vectorización de texto."""
    print(f"[INGESTA] Cargando OpenCLIP {CLIP_MODEL} en CPU...")
    # CLIP siempre en CPU para compatibilidad con ChromaDB en macOS
    clip_device = "cpu"
    # CLIP_MODEL incluye "hf-hub:" → no se pasa pretrained
    kwargs = {} if CLIP_MODEL.startswith("hf-hub:") else {"pretrained": CLIP_PRETRAINED}
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, **kwargs)
    model = model.to(clip_device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, tokenizer, clip_device


def embed_text(text: str, model, tokenizer, device: str) -> list[float]:
    """Convierte texto a vector L2-normalizado con OpenCLIP."""
    tokens = tokenizer([text[:CLIP_MAX_TOKENS]]).to(device)
    with torch.no_grad():
        vec = model.encode_text(tokens)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().flatten().tolist()


def ingest_all_manuals(manuals_dir: str = MANUALS_DIR, reset: bool = False) -> int:
    """
    Ingesta todos los archivos .txt de manuals_dir a ChromaDB.
    reset=True borra la colección antes de ingestar.
    Retorna el número de chunks insertados.
    """
    # ── Verificar manuales ───────────────────────────────────────────────────
    txt_files = glob.glob(os.path.join(manuals_dir, "*.txt"))
    if not txt_files:
        print(f"[INGESTA] No se encontraron archivos .txt en {manuals_dir}")
        return 0

    print(f"[INGESTA] {len(txt_files)} manuales encontrados.")

    # ── Conectar a ChromaDB ──────────────────────────────────────────────────
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"[INGESTA] Colección '{CHROMA_COLLECTION}' eliminada.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},  # similitud del coseno
    )

    # ── Cargar modelo de embeddings ──────────────────────────────────────────
    model, tokenizer, clip_device = get_clip_embedder()

    # ── Obtener IDs ya existentes (para modo incremental) ───────────────────
    existing_ids: set = set()
    if not reset:
        existing_ids = set(collection.get()["ids"])

    # ── Procesar cada manual — batch upsert por archivo ──────────────────────
    total_chunks = 0
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        print(f"[INGESTA] Procesando: {filename}", flush=True)

        chunks = chunk_document(filepath)
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]
        print(f"           → {len(chunks)} chunks ({len(new_chunks)} nuevos)", flush=True)

        if not new_chunks:
            print(f"           ✓ Sin cambios.", flush=True)
            continue

        # Vectorizar todos los chunks del archivo de una vez
        ids_batch, docs_batch, metas_batch, vecs_batch = [], [], [], []
        for chunk in new_chunks:
            vec = embed_text(chunk["text"], model, tokenizer, clip_device)
            ids_batch.append(chunk["id"])
            docs_batch.append(chunk["text"])
            metas_batch.append({"source": chunk["source"], "id": chunk["id"]})
            vecs_batch.append(vec)

        # Un solo upsert por archivo (mucho más rápido que uno por chunk)
        collection.upsert(
            embeddings=vecs_batch,
            documents=docs_batch,
            metadatas=metas_batch,
            ids=ids_batch,
        )
        total_chunks += len(ids_batch)
        print(f"           ✓ {len(ids_batch)} chunks insertados.", flush=True)

    print(f"\n[INGESTA] Total chunks en ChromaDB: {collection.count()}", flush=True)
    return total_chunks
