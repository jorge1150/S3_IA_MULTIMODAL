# S3 IA Multimodal — Asistente de Soporte Técnico Local

**Maestría en IA Aplicada — UIsrael**  
**Docente:** MSc. Ing. Pablo Recalde  
**Módulo:** IA Multimodal — Semana 3

---

## Descripción

Sistema de soporte técnico para computadoras que funciona **100% de forma local** sin internet ni APIs externas. Combina voz, visión, texto y video para diagnosticar y resolver problemas técnicos.

## Pipeline del Sistema

```
🎤 Voz (48kHz)  ──→ [Whisper STT]  ──→ Texto transcrito
📷 Imagen/Webcam ──→ [Moondream]   ──→ Descripción visual
💬 Texto libre   ──→                 ──→ Consulta combinada
🎬 Video         ──→ [Moondream]   ──→ Descripción de frames
                              ↓
                    Consulta unificada
                              ↓
               [OpenCLIP ViT-B-32] ── vectoriza
                              ↓
            [ChromaDB (cosine)]  ── busca top-3 chunks
                              ↓
           [TinyLlama via Ollama] ── razona con contexto
                              ↓
              [Piper TTS / say]  ── sintetiza voz
                              ↓
                 🔊 Respuesta en texto y audio
```

## Arquitectura Multiagente

| Agente | Responsabilidad |
|--------|----------------|
| `CoordinatorAgent` | Orquesta el pipeline completo, gestiona el flujo |
| `VoiceAgent` | STT con faster-whisper base (int8, beam=5, español) |
| `VisionAgent` | Análisis de imágenes con Moondream via Ollama |
| `VideoAgent` | Extrae frames y los analiza con VisionAgent |
| `RAGAgent` | Vectoriza con OpenCLIP y recupera de ChromaDB |
| `ResponseAgent` | Genera respuesta con TinyLlama (temp=0.1) |
| `TTSAgent` | Síntesis de voz con Piper TTS + fallback macOS |
| `LogAgent` | Trazabilidad thread-safe en tiempo real |

## Tecnologías y Justificación

| Tecnología | Elección | Justificación |
|---|---|---|
| **STT** | faster-whisper (base, int8) | El más rápido en CPU; int8 reduce 4x el uso de RAM |
| **LLM** | TinyLlama via Ollama | 1.1B parámetros cuantizados, cabe en 1GB de RAM |
| **Visión** | Moondream via Ollama | Modelo de visión compacto y eficiente en CPU |
| **Embeddings** | OpenCLIP ViT-B-32 | Mismo espacio de representación para texto e imágenes |
| **Vector DB** | ChromaDB | Local, persistente, cosine similarity, sin servidor externo |
| **TTS** | Piper TTS (es_ES-sharvard-medium) | Voz española natural, funciona offline |
| **UI** | Gradio 4.x | UI web con streaming, soporte webcam/micrófono nativo |

## Requisitos del Sistema

- **OS:** macOS (Intel o Apple Silicon), Linux, Windows
- **RAM:** 16GB mínimo, 32GB recomendado
- **Espacio:** ~5GB (modelos + dependencias)
- **Python:** 3.10 o superior
- **Ollama:** instalado y ejecutándose

## Instalación Rápida

```bash
# Clonar o navegar al directorio del proyecto
cd S3_IA_MULTIMODAL

# Instalación automática
bash setup.sh
```

## Instalación Manual

```bash
# 1. Entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. PortAudio (macOS, requerido por sounddevice)
brew install portaudio

# 3. Dependencias Python
pip install -r requirements.txt

# 4. Instalar y configurar Ollama
brew install ollama
ollama serve &          # En otra terminal

ollama pull tinyllama   # ~637MB — modelo de razonamiento
ollama pull moondream   # ~828MB — modelo de visión

# 5. Descargar modelo Piper TTS
python audio/download_piper.py

# 6. Construir base vectorial
python rag/build_db.py

# 7. Lanzar el sistema
python app.py
```

## Uso del Sistema

Abrir en el navegador: **http://localhost:7864**

### Ejemplos de consultas:

- *"Mi computadora no conecta a internet desde ayer"*
- *"La pantalla se pone negra cuando prendo el equipo"*
- Subir foto de un mensaje de error de Windows
- Grabar por voz: "Escucho ruido extraño en el disco duro"

## Estructura del Proyecto

```
S3_IA_MULTIMODAL/
├── app.py                    ← Punto de entrada
├── config.py                 ← Configuración central
├── requirements.txt          ← Dependencias
├── setup.sh                  ← Instalación automática
│
├── agents/                   ← Sistema multiagente
│   ├── coordinator.py        ← Orquestador principal
│   ├── voice_agent.py        ← STT (faster-whisper)
│   ├── vision_agent.py       ← Visión (Moondream/Ollama)
│   ├── video_agent.py        ← Video (OpenCV + VisionAgent)
│   ├── rag_agent.py          ← RAG (OpenCLIP + ChromaDB)
│   ├── response_agent.py     ← LLM (TinyLlama/Ollama)
│   ├── tts_agent.py          ← TTS (Piper + macOS say)
│   └── log_agent.py          ← Trazabilidad
│
├── rag/
│   ├── build_db.py           ← Script de ingesta
│   ├── ingesta.py            ← Carga y vectoriza manuales
│   └── chunker.py            ← Fragmentación de texto
│
├── audio/
│   ├── scanner.py            ← Detectar dispositivos de audio
│   ├── download_piper.py     ← Descargar modelo TTS
│   └── piper_models/         ← Modelos Piper TTS
│
├── vision/
│   └── capture.py            ← Screenshot y webcam
│
├── ui/
│   ├── interface.py          ← Gradio Blocks
│   └── styles.py             ← CSS profesional
│
├── manuals/                  ← Base de conocimiento (.txt)
├── vector_db/                ← ChromaDB persistente
├── logs/                     ← Logs del sistema
├── temp/                     ← Archivos temporales de audio
└── tests/                    ← Pruebas de integración
```

## Diagnóstico de Problemas

| Síntoma | Solución |
|---|---|
| "Ollama no disponible" | Ejecutar `ollama serve` en terminal separada |
| Modelo no encontrado | `ollama pull tinyllama` y `ollama pull moondream` |
| Sin voz de respuesta | `python audio/download_piper.py` |
| Base vectorial vacía | `python rag/build_db.py` |
| Error de micrófono | `python audio/scanner.py` para identificar el índice correcto |
| Error de portaudio | `brew install portaudio` y reinstalar sounddevice |

## Escáner de Dispositivos de Audio

```bash
python audio/scanner.py        # Lista todos los dispositivos
python audio/scanner.py 28     # Prueba el micrófono del índice 28
```

Actualizar `MIC_DEVICE_INDEX` en `config.py` con el índice correcto.

## Ejecutar Pruebas

```bash
python tests/test_agents.py    # Pruebas de agentes
python tests/test_rag.py       # Pruebas del sistema RAG
```

---

*Sistema desarrollado como proyecto final del módulo IA Multimodal.*  
*Todo el procesamiento es local — sin internet, sin APIs externas, sin costos de uso.*
