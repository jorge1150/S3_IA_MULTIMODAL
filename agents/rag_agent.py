"""
rag_agent.py — Agente RAG (Retrieval-Augmented Generation)
Responsabilidad: vectorizar la consulta con OpenCLIP y buscar los
fragmentos más relevantes en ChromaDB por similitud del coseno.
"""

import torch
import open_clip
import chromadb
from typing import List

from config import (
    DEVICE, CLIP_MODEL, CLIP_PRETRAINED, CLIP_MAX_TOKENS,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    RAG_TOP_K, RAG_MIN_SIMILARITY,
)
from .log_agent import LogAgent


class RAGAgent:
    """
    Motor de recuperación semántica.
    - Usa OpenCLIP ViT-B-32 para vectorizar texto y consultar por imágenes.
    - ChromaDB almacena los embeddings del manual técnico.
    """

    def __init__(self, log_agent: LogAgent):
        self.log = log_agent
        self._clip_model = None
        self._tokenizer = None
        self._preprocessor = None
        self._collection = None
        self._clip_device = "cpu"  # CLIP siempre en CPU en macOS Intel

    # ── Carga lazy de modelos ────────────────────────────────────────────────

    def _load_clip(self):
        if self._clip_model is not None:
            return
        self.log.log("RAG", f"Cargando OpenCLIP {CLIP_MODEL} en CPU...")
        # CLIP_MODEL incluye "hf-hub:" → no se pasa pretrained
        kwargs = {} if CLIP_MODEL.startswith("hf-hub:") else {"pretrained": CLIP_PRETRAINED}
        self._clip_model, _, self._preprocessor = open_clip.create_model_and_transforms(
            CLIP_MODEL, **kwargs
        )
        self._clip_model = self._clip_model.to(self._clip_device)
        self._clip_model.eval()
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.log.log("RAG", "OpenCLIP listo.")

    def _load_chroma(self):
        if self._collection is not None:
            return
        self.log.log("RAG", f"Conectando a ChromaDB en {CHROMA_DB_PATH}...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self._collection = client.get_collection(name=CHROMA_COLLECTION)
            count = self._collection.count()
            self.log.log("RAG", f"Colección '{CHROMA_COLLECTION}' cargada: {count} documentos.")
        except Exception:
            self.log.log("ERROR", (
                f"Colección '{CHROMA_COLLECTION}' no existe. "
                "Ejecuta: python rag/build_db.py"
            ))
            self._collection = None

    # ── API pública ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Busca los fragmentos más relevantes para la consulta.
        Estrategia: CLIP devuelve más candidatos (top_k * 4), luego se
        re-rankean por solapamiento de palabras clave antes de devolver top_k.
        Retorna lista de dicts con {'text', 'similarity', 'id'}.
        """
        self._load_clip()
        self._load_chroma()

        if self._collection is None:
            return []

        # Vectorizar la consulta con OpenCLIP
        query_vector = self._embed_text(query)
        if query_vector is None:
            return []

        # Recuperar todos los chunks (255 total): rápido en CPU y permite
        # que el keyword re-ranking suba los documentos realmente relevantes
        # aunque CLIP (modelo de visión) los haya penalizado en similitud vectorial.
        candidate_k = self._collection.count()
        self.log.log("RAG", f"Buscando top-{top_k} en ChromaDB (candidatos={candidate_k})...")
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
        )

        chunks = []
        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc, dist, doc_id in zip(docs, dists, ids):
            similarity = 1.0 - dist
            if similarity >= RAG_MIN_SIMILARITY:
                chunks.append({"text": doc, "similarity": round(similarity, 3), "id": doc_id})

        # Re-rankear por palabras clave del query (compensa limitaciones de CLIP en texto)
        chunks = self._keyword_rerank(query, chunks)

        # Devolver solo los top_k finales
        chunks = chunks[:top_k]

        for c in chunks:
            self.log.log("RAG", f"  [{c['id']}] sim={c['similarity']:.2f} — {c['text'][:60]}...")

        if not chunks:
            self.log.log("RAG", "Ningún chunk superó el umbral de similitud.")
        else:
            self.log.log("RAG", f"✓ {len(chunks)} fragmento(s) relevante(s) encontrados.")
        return chunks

    def _keyword_rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Re-rankea chunks por solapamiento de palabras clave con la query.
        Boost extra si el nombre del archivo fuente contiene palabras del query.
        Necesario porque CLIP fue entrenado para visión, no búsqueda de texto.
        """
        _STOPWORDS = {
            'para', 'como', 'cuando', 'pero', 'esto', 'esta', 'este', 'que',
            'del', 'los', 'las', 'una', 'unos', 'unas', 'por', 'con', 'sin',
            'sobre', 'entre', 'bajo', 'hace', 'tiene', 'tengo', 'puede',
            'puedo', 'creo', 'bien', 'mal', 'muy', 'hay', 'ser', 'está',
        }
        query_words = {
            w.lower().strip('.,;:!?¡¿')
            for w in query.split()
            if len(w) > 3 and w.lower() not in _STOPWORDS
        }
        if not query_words:
            return chunks

        for chunk in chunks:
            text_lower = chunk["text"].lower()
            id_lower = chunk["id"].lower()  # p.ej. "sin_internet_0003"

            # Hits en el texto del chunk
            text_hits = sum(1 for w in query_words if w in text_lower)
            # Bonus si la palabra clave aparece en el nombre del archivo fuente
            source_hits = sum(1 for w in query_words if w in id_lower)

            boost = 1.0 + 0.10 * text_hits + 0.25 * source_hits
            chunk["similarity"] = round(chunk["similarity"] * boost, 3)

        return sorted(chunks, key=lambda x: x["similarity"], reverse=True)

    def embed_image(self, pil_image) -> list[float] | None:
        """Vectoriza una imagen PIL para buscar por imagen en ChromaDB."""
        self._load_clip()
        try:
            tensor = self._preprocessor(pil_image).unsqueeze(0).to(self._clip_device)
            with torch.no_grad():
                vec = self._clip_model.encode_image(tensor)
                vec /= vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().flatten().tolist()
        except Exception as exc:
            self.log.log("ERROR", f"Error vectorizando imagen: {exc}")
            return None

    # ── Utilidades privadas ──────────────────────────────────────────────────

    def _embed_text(self, text: str) -> list[float] | None:
        """Convierte texto a vector L2-normalizado con OpenCLIP."""
        try:
            # Límite de 77 tokens de CLIP; truncar texto como precaución
            tokens = self._tokenizer([text[:CLIP_MAX_TOKENS]]).to(self._clip_device)
            with torch.no_grad():
                vec = self._clip_model.encode_text(tokens)
                vec /= vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().flatten().tolist()
        except Exception as exc:
            self.log.log("ERROR", f"Error vectorizando texto: {exc}")
            return None
