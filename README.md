# IA Multimodal — Soporte Técnico Computadoras Local

> Sistema inteligente de soporte técnico para computadoras que opera **100% de forma local**, sin conexión a internet ni APIs externas de pago. Integra voz, visión, texto y video para diagnosticar y resolver problemas técnicos en tiempo real.

**Maestría en Inteligencia Artificial Aplicada — Universidad Tecnológica Israel**  
**Docente:** MSc. Ing. Pablo Recalde | Módulo: IA Multimodal

---

## Características Principales

- **Multimodal completo**: acepta texto, voz, imágenes, capturas de pantalla y video simultáneamente
- **100% local**: ningún dato sale del equipo — sin OpenAI, sin Google, sin costos de API
- **Respuesta en voz**: sintetiza la solución en español con Piper TTS
- **Base de conocimiento técnica**: 13 manuales de soporte, 255 fragmentos vectorizados
- **Interfaz web profesional**: accesible desde cualquier navegador en la red local

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRADAS MULTIMODALES                     │
│   🎤 Voz    📷 Imagen    💬 Texto    🎬 Video               │
└──────┬──────────┬──────────┬──────────┬───────────────────-─┘
       │          │          │          │
       ▼          ▼          │          ▼
  [Whisper    [Moondream  │      [OpenCV +
   STT]        Visión]    │       Moondream]
       │          │          │          │
       └──────────┴──────────┴──────────┘
                           │
                    Consulta unificada
                           │
                           ▼
              [OpenCLIP ViT-B-32] — vectoriza
                           │
                           ▼
          [ChromaDB cosine] — recupera top-3 chunks
                           │
                           ▼
         [TinyLlama via Ollama] — genera diagnóstico
                           │
                           ▼
             [Piper TTS] — sintetiza en voz
                           │
                           ▼
             🔊 Respuesta en texto y audio
```

---

## Agentes del Sistema

| Agente | Tecnología | Función |
|---|---|---|
| `CoordinatorAgent` | Python | Orquesta el pipeline completo |
| `VoiceAgent` | faster-whisper (base, int8) | Transcripción de voz a texto |
| `VisionAgent` | Moondream via Ollama | Análisis de imágenes y capturas |
| `VideoAgent` | OpenCV + VisionAgent | Extrae y analiza frames de video |
| `RAGAgent` | OpenCLIP + ChromaDB | Recuperación semántica del manual |
| `ResponseAgent` | TinyLlama via Ollama | Generación de diagnóstico y solución |
| `TTSAgent` | Piper TTS (es_ES) | Síntesis de voz en español |
| `LogAgent` | Thread-safe queue | Trazabilidad en tiempo real |

---

## Tecnologías Utilizadas

| Componente | Tecnología | Versión | Justificación |
|---|---|---|---|
| STT | faster-whisper | ≥1.0.0 | CPU int8, 4× más rápido que openai-whisper |
| LLM | TinyLlama (Ollama) | 1.1B params | Cabe en 1 GB RAM, respuestas coherentes en español |
| Visión | Moondream (Ollama) | — | Modelo de visión compacto, eficiente en CPU |
| Embeddings | OpenCLIP ViT-B-32 | ≥2.24.0 | Espacio semántico compartido texto-imagen |
| Vector DB | ChromaDB | ≥0.5.0 | Local, persistente, sin servidor externo |
| TTS | Piper TTS | ≥1.4.0 | Voz española natural, 100% offline |
| Framework | PyTorch | 2.2.2 | Compatible macOS Intel x86_64 + Python 3.12 |
| UI | Gradio | 6.16.0 | Interfaz web con streaming y soporte multimedia |

---

## Requisitos del Sistema

| Requisito | Mínimo | Recomendado |
|---|---|---|
| Sistema Operativo | macOS 12+, Ubuntu 20.04+, Windows 10+ | macOS 13+ / Ubuntu 22.04+ |
| RAM | 8 GB | 16 GB o más |
| Almacenamiento | 5 GB libres | 10 GB libres |
| Python | 3.12 (obligatorio) | 3.12.x |
| Ollama | Cualquier versión reciente | Última versión |

> **Nota:** Python 3.12 es obligatorio. PyTorch 2.2.x no tiene wheel para Python 3.13 en macOS Intel.

---

## Instalación

### Instalación Automática (Recomendada)

```bash
# 1. Navegar al directorio del proyecto
cd IA_MULTIMODAL

# 2. Dar permisos y ejecutar el instalador
chmod +x setup.sh
bash setup.sh
```

El script realiza automáticamente:
- Verificación de Python 3.12
- Creación del entorno virtual
- Instalación de PortAudio (macOS)
- Instalación de todas las dependencias Python
- Descarga del modelo Piper TTS en español
- Construcción de la base vectorial ChromaDB

### Instalación Manual

```bash
# 1. Crear entorno virtual con Python 3.12
python3.12 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 2. PortAudio (solo macOS)
brew install portaudio

# 3. Instalar dependencias (orden importante)
pip install "numpy<2" scipy sounddevice Pillow requests python-dotenv
pip install "torch==2.2.2" "torchvision==0.17.2"
pip install open_clip_torch faster-whisper "chromadb>=0.5.0"
pip install "opencv-python==4.8.1.78"
pip install "piper-tts>=1.4.0" "gradio==6.16.0"

# 4. Instalar Ollama y descargar modelos LLM
# → Descargar desde https://ollama.com
ollama serve                      # en otra terminal
ollama pull tinyllama             # ~637 MB — razonamiento
ollama pull moondream             # ~828 MB — visión

# 5. Descargar modelo de voz en español
python audio/download_piper.py

# 6. Construir base vectorial
python rag/build_db.py

# 7. Lanzar el sistema
python app.py
```

---

## Uso

1. Asegurarse de que Ollama esté ejecutándose (`ollama serve`)
2. Activar el entorno virtual: `source venv/bin/activate`
3. Iniciar el sistema: `python app.py`
4. Abrir en el navegador: **http://localhost:7864**

### Modos de Consulta

| Modo | Descripción |
|---|---|
| **Solo texto** | Escribe el problema en el campo de descripción |
| **Voz** | Graba con el micrófono; el sistema transcribe y responde |
| **Imagen** | Sube una captura de pantalla o foto del error |
| **Captura de pantalla** | Botón para capturar la pantalla directamente |
| **Video** | Sube un video; se analizan frames automáticamente |
| **Multimodal** | Combina voz + imagen para diagnósticos más precisos |

### Ejemplos de Consultas

- *"Mi computadora no conecta a internet"*
- *"La pantalla se pone negra al encender el equipo"*
- Subir foto de pantalla azul (BSOD) de Windows
- Grabar por voz: *"El ventilador hace mucho ruido"*
- Subir video mostrando el error en pantalla

---

## Estructura del Proyecto

```
IA_MULTIMODAL/
├── app.py                    ← Punto de entrada principal
├── config.py                 ← Configuración central del sistema
├── requirements.txt          ← Dependencias Python
├── setup.sh                  ← Script de instalación automática
├── .gitignore                ← Exclusiones de control de versiones
│
├── agents/                   ← Sistema multiagente
│   ├── coordinator.py        ← Orquestador del pipeline
│   ├── voice_agent.py        ← STT con faster-whisper
│   ├── vision_agent.py       ← Visión con Moondream/Ollama
│   ├── video_agent.py        ← Análisis de video con OpenCV
│   ├── rag_agent.py          ← Recuperación semántica OpenCLIP + ChromaDB
│   ├── response_agent.py     ← Generación de respuesta con TinyLlama
│   ├── tts_agent.py          ← Síntesis de voz con Piper TTS
│   └── log_agent.py          ← Trazabilidad en tiempo real
│
├── rag/
│   ├── build_db.py           ← Script de construcción de ChromaDB
│   ├── ingesta.py            ← Carga y vectoriza manuales
│   └── chunker.py            ← Fragmentación de documentos
│
├── audio/
│   ├── download_piper.py     ← Descarga modelo TTS en español
│   ├── scanner.py            ← Detecta dispositivos de audio
│   └── piper_models/         ← Modelos Piper TTS (excluido de git)
│
├── vision/
│   └── capture.py            ← Captura de pantalla y webcam
│
├── ui/
│   ├── interface.py          ← Interfaz Gradio Blocks
│   └── styles.py             ← Estilos CSS
│
├── manuals/                  ← Base de conocimiento (13 manuales .txt)
├── vector_db/                ← ChromaDB persistente (excluido de git)
├── logs/                     ← Logs del sistema (excluido de git)
├── temp/                     ← Audio temporal (excluido de git)
└── tests/                    ← Pruebas de integración
```

---

## Solución de Problemas

| Síntoma | Causa Probable | Solución |
|---|---|---|
| "Ollama no disponible" | Servicio no iniciado | Ejecutar `ollama serve` en terminal separada |
| Modelo no encontrado | No descargado | `ollama pull tinyllama && ollama pull moondream` |
| Sin voz en respuesta | Modelo Piper faltante | `python audio/download_piper.py` |
| Base vectorial vacía | ChromaDB no construida | `python rag/build_db.py` |
| Error de micrófono | Índice incorrecto | `python audio/scanner.py` para identificar dispositivo |
| Error de PortAudio | Librería no instalada | `brew install portaudio` y reinstalar sounddevice |
| Error de importación torch | Python versión incorrecta | Verificar Python 3.12: `python --version` |

---

## Manuales de Soporte Incluidos

La base de conocimiento contiene 13 manuales técnicos con 255 fragmentos vectorizados:

| Manual | Categoría |
|---|---|
| `sin_internet.txt` | Conectividad y red |
| `pantalla_negra.txt` | Problemas de pantalla |
| `no_enciende.txt` | Arranque del sistema |
| `computadora_lenta.txt` | Rendimiento |
| `audio_problemas.txt` | Sonido y multimedia |
| `impresora.txt` | Periféricos |
| `teclado_mouse.txt` | Dispositivos de entrada |
| `virus_malware.txt` | Seguridad |
| `actualizaciones.txt` | Sistema operativo |
| `almacenamiento.txt` | Disco y almacenamiento |
| `windows_basico.txt` | Windows (incluye BSOD) |
| `linux_basico.txt` | Linux |
| `macos_basico.txt` | macOS |

---

*Desarrollado como proyecto final del módulo IA Multimodal.*  
*Procesamiento 100% local — sin internet, sin APIs externas, sin costos de uso.*
