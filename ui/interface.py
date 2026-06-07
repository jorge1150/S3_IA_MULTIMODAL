"""
interface.py — Interfaz Gradio del sistema S3 IA Multimodal.
Diseño profesional con tabs, inputs multimodales, logs en tiempo real
y reproducción de audio de respuesta.
"""

import gradio as gr
from vision.capture import capture_screenshot
from ui.styles import CSS
from config import GRADIO_PORT, GRADIO_SERVER, GRADIO_TITLE

# Gradio 6.x: theme se pasa a launch(), no a Blocks()
GRADIO_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
)


def build_interface(coordinator) -> gr.Blocks:
    """
    Construye y retorna el objeto gr.Blocks con toda la UI.
    coordinator: instancia de CoordinatorAgent (singleton).
    """

    # ── Handler principal ─────────────────────────────────────────────────

    def analyze(image, audio, text_input, video):
        """
        Generador para el botón 'Analizar'.
        Hace yield de actualizaciones parciales para el streaming en tiempo real.
        """
        for stt, resp, audio_path, logs in coordinator.process(
            image_input=image,
            audio_input=audio,
            text_input=text_input,
            video_input=video,
        ):
            yield stt, resp, audio_path, logs

    def take_screenshot():
        """Captura la pantalla y la inserta en el componente de imagen."""
        img = capture_screenshot()
        return img

    def clear_all():
        """Limpia todos los campos de entrada y salida."""
        return None, None, "", None, "", "", None, ""

    # ── Layout ────────────────────────────────────────────────────────────

    with gr.Blocks(title=GRADIO_TITLE) as demo:
        # CSS inyectado via HTML — compatible con Gradio 6.x (css= deprecado en Blocks)
        gr.HTML(f"<style>{CSS}</style>")

        # Header
        gr.HTML("""
        <div class="s3-header">
            <h1>⚡ IA Multimodal</h1>
            <p>Soporte Técnico Computadoras · 100% Local · Powered by Ollama + ChromaDB</p>
        </div>
        """)

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════
            # TAB 1 — Asistente Principal
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("🤖 Asistente"):
                with gr.Row(equal_height=False):

                    # ── Columna izquierda: ENTRADAS ──────────────────────
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML("<h3 style='color:#00d4ff;margin:0 0 12px'>📥 Entradas</h3>")

                        # Imagen / Webcam
                        img_input = gr.Image(
                            label="📷 Imagen o Captura Webcam",
                            sources=["webcam", "upload"],
                            type="pil",
                            elem_classes=["image-container"],
                            height=200,
                        )

                        # Botón de captura de pantalla
                        btn_screenshot = gr.Button(
                            "🖥️ Capturar Pantalla",
                            variant="secondary",
                            size="sm",
                        )

                        # Audio / Micrófono
                        audio_input = gr.Audio(
                            label="🎤 Voz (Micrófono o Archivo)",
                            sources=["microphone", "upload"],
                            type="filepath",
                        )

                        # Texto libre
                        text_input = gr.Textbox(
                            label="💬 Descripción del Problema",
                            placeholder="Ej: Mi computadora no conecta a internet, pantalla negra al encender...",
                            lines=3,
                        )

                        # Video
                        video_input = gr.Video(
                            label="🎬 Video (opcional)",
                            sources=["upload"],
                            elem_classes=["video-container"],
                        )

                        # Botones de acción
                        with gr.Row():
                            btn_analyze = gr.Button(
                                "🚀 Analizar y Diagnosticar",
                                variant="primary",
                                elem_classes=["btn-analyze"],
                            )
                            btn_clear = gr.Button(
                                "🗑️ Limpiar",
                                variant="secondary",
                                elem_classes=["btn-clear"],
                                size="sm",
                            )

                    # ── Columna derecha: SALIDAS ─────────────────────────
                    with gr.Column(scale=1, min_width=380):
                        gr.HTML("<h3 style='color:#7c3aed;margin:0 0 12px'>📤 Respuesta</h3>")

                        # Texto transcrito (STT)
                        stt_output = gr.Textbox(
                            label="🎙️ Voz Capturada (STT)",
                            lines=2,
                            interactive=False,
                            placeholder="La transcripción de tu voz aparecerá aquí...",
                        )

                        # Respuesta final del LLM
                        response_output = gr.Textbox(
                            label="💡 Diagnóstico y Solución",
                            lines=10,
                            interactive=False,
                            placeholder="El diagnóstico técnico aparecerá aquí...",
                            elem_classes=["response-box"],
                        )

                        # Audio de respuesta TTS
                        audio_output = gr.Audio(
                            label="🔊 Respuesta en Voz",
                            type="filepath",
                            autoplay=True,
                            interactive=False,
                        )

                # Logs en tiempo real (parte inferior)
                gr.HTML("<hr style='border-color:#2d3748;margin:16px 0'>")
                gr.HTML("<h3 style='color:#94a3b8;margin:0 0 8px'>📋 Consola de Trazabilidad</h3>")
                logs_output = gr.Textbox(
                    label="",
                    lines=8,
                    interactive=False,
                    placeholder="[INICIO] → [STT] → [RAG] → [VISION] → [DIAGNOSTICO] → [TTS] → [FIN]",
                    elem_classes=["logs-console"],
                )

            # ═══════════════════════════════════════════════════════════════
            # TAB 2 — Información del Sistema
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("⚙️ Sistema"):
                _build_system_tab()

            # ═══════════════════════════════════════════════════════════════
            # TAB 3 — Guía de Uso
            # ═══════════════════════════════════════════════════════════════
            with gr.Tab("📖 Guía"):
                _build_guide_tab()

        # ── Eventos ───────────────────────────────────────────────────────

        # Botón principal: streaming con generador
        btn_analyze.click(
            fn=analyze,
            inputs=[img_input, audio_input, text_input, video_input],
            outputs=[stt_output, response_output, audio_output, logs_output],
            show_progress="full",
        )

        # Captura de pantalla
        btn_screenshot.click(
            fn=take_screenshot,
            inputs=[],
            outputs=[img_input],
        )

        # Limpiar todo
        btn_clear.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                img_input, audio_input, text_input, video_input,
                stt_output, response_output, audio_output, logs_output,
            ],
        )

    return demo


# ── Tabs auxiliares ───────────────────────────────────────────────────────────

def _build_system_tab():
    """Información del pipeline y estado del sistema."""
    import config, torch

    device_info = f"**Dispositivo de cómputo:** `{config.DEVICE}`"
    cuda_info = f"**CUDA disponible:** `{torch.cuda.is_available()}`"

    gr.Markdown(f"""
## Estado del Sistema

| Componente | Valor |
|---|---|
| Dispositivo | `{config.DEVICE}` |
| CUDA | `{torch.cuda.is_available()}` |
| LLM (Ollama) | `{config.LLM_MODEL}` |
| Visión | `{config.VISION_MODEL}` |
| STT | `faster-whisper {config.WHISPER_MODEL_SIZE}` |
| Embeddings | `OpenCLIP {config.CLIP_MODEL}` |
| Vector DB | `ChromaDB (cosine)` |
| TTS | `Piper TTS / macOS say` |
| Puerto | `{config.GRADIO_PORT}` |
| ChromaDB Path | `{config.CHROMA_DB_PATH}` |

## Pipeline de Procesamiento

```
🎤 Audio → [Whisper STT] → Texto
📷 Imagen → [Moondream] → Descripción visual
💬 Texto + STT + Visual → [OpenCLIP] → Vector
Vector → [ChromaDB] → Contexto RAG
Contexto + Query → [TinyLlama] → Respuesta
Respuesta → [Piper TTS] → Audio
```

## Problemas Cubiertos
- Sin internet · Computadora lenta · Pantalla negra · No enciende
- Problemas de audio · Impresora · Teclado/Mouse · Actualizaciones
- Almacenamiento · Virus/Malware · Windows · Linux · macOS
""")


def _build_guide_tab():
    """Guía rápida de uso del sistema."""
    gr.Markdown("""
## Guía de Uso Rápido

### Opción 1: Solo Texto
1. Escribe tu problema en el campo **"Descripción del Problema"**
2. Haz clic en **"Analizar y Diagnosticar"**
3. Lee la solución y escucha la respuesta en voz

### Opción 2: Voz
1. Haz clic en el ícono del micrófono en **"Voz (Micrófono)"**
2. Habla tu problema claramente
3. Detén la grabación
4. Haz clic en **"Analizar y Diagnosticar"**

### Opción 3: Imagen
1. Usa **"Imagen o Captura Webcam"** para:
   - Subir una captura de pantalla del error
   - Tomar foto de la pantalla con webcam
   - O haz clic en **"Capturar Pantalla"** directamente
2. Combina con texto o voz para mejores resultados

### Opción 4: Video
1. Sube un video del problema en **"Video"**
2. El sistema analiza los frames automáticamente

### Combinación Multimodal (Recomendado)
> Combina voz + imagen para diagnósticos más precisos.
> El sistema unifica todas las entradas antes de consultar el manual.

---

### Solución de Problemas
| Problema | Solución |
|---|---|
| "Ollama no disponible" | Ejecuta `ollama serve` en terminal |
| Sin respuesta de voz | Ejecuta `python audio/download_piper.py` |
| Base de conocimiento vacía | Ejecuta `python rag/build_db.py` |
| Modelo no encontrado | `ollama pull tinyllama && ollama pull moondream` |
""")
