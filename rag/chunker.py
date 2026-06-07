"""
chunker.py — Utilidades de fragmentación de texto.
Fragmenta documentos largos en chunks que respetan el límite de 77 tokens
de OpenCLIP (~200 caracteres en español).
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 20) -> List[str]:
    """
    Divide texto en fragmentos de chunk_size caracteres con overlap.
    Intenta cortar en el último espacio para no romper palabras.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Buscar último espacio para no cortar palabras
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break  # llegamos al final — evita bucle infinito con overlap

        start = end - overlap

    return chunks


def chunk_document(filepath: str, chunk_size: int = 180, overlap: int = 20) -> List[dict]:
    """
    Lee un archivo de texto y retorna lista de chunks con metadata.
    Returns: [{'id': str, 'text': str, 'source': str}]
    """
    import os
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

    result = []
    for i, chunk in enumerate(raw_chunks):
        result.append({
            "id": f"{name_no_ext}_{i:04d}",
            "text": chunk,
            "source": filename,
        })
    return result
