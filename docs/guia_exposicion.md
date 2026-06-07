# Guía de Exposición — S3 IA Multimodal

## Estructura sugerida (20 minutos)

---

### 1. Introducción (3 min)

**Frase de apertura:**
> "Imaginemos un técnico de soporte que puede ver su pantalla, escuchar su voz y leer su problema al mismo tiempo — y que funciona sin internet, sin costos y sin enviar sus datos a ningún servidor externo."

**Puntos clave:**
- El problema: soporte técnico de computadoras es costoso y no siempre disponible
- La solución: IA local que combina voz, visión y texto
- La tecnología: pipeline Whisper → CLIP → ChromaDB → TinyLlama → Piper TTS

---

### 2. Arquitectura del Sistema (4 min)

**Dibujar o mostrar el diagrama:**
```
🎤 VOZ → Whisper (STT) → Texto
📷 IMAGEN → Moondream → Descripción
💬 TEXTO → ─────────────────────┐
                                 ↓
                    [OpenCLIP ViT-B-32]
                                 ↓
                    [ChromaDB cosine]
                                 ↓
                    [TinyLlama + Ollama]
                                 ↓
                    [Piper TTS español]
                                 ↓
                         🔊 RESPUESTA
```

**Explicar cada agente brevemente:**
- LogAgent: trazabilidad en tiempo real (la "caja negra" del sistema)
- VoiceAgent: convierte voz a texto con Whisper base (int8 = más rápido en CPU)
- VisionAgent: "entiende" las imágenes con Moondream
- RAGAgent: "busca en el manual" usando similitud matemática
- ResponseAgent: "razona" la respuesta con TinyLlama
- TTSAgent: "habla" la respuesta con Piper TTS

---

### 3. Demostración en Vivo (8 min)

**Antes de la demo:** asegurarse de que Ollama esté ejecutándose y el sistema abierto en el navegador.

**Demo 1 — Solo texto (2 min):**
1. Escribir: "Mi computadora no conecta a internet"
2. Hacer clic en Analizar
3. Mostrar los logs en tiempo real: [INICIO] → [RAG] → [DIAGNOSTICO] → [TTS]
4. Leer la respuesta técnica generada

**Demo 2 — Voz (2 min):**
1. Hacer clic en el micrófono
2. Decir: "El ventilador de mi computadora hace mucho ruido y el equipo está muy lento"
3. Detener y analizar
4. Mostrar el texto transcrito por Whisper y la solución generada

**Demo 3 — Imagen (2 min):**
1. Capturar una pantalla con un mensaje de error visible (o subir una imagen preparada)
2. Analizar
3. Mostrar cómo Moondream describe la imagen y cómo el sistema genera la solución

**Demo 4 — Multimodal completo (2 min):**
1. Combinar: imagen del error + descripción de voz
2. Mostrar cómo el sistema integra ambas entradas

---

### 4. Conceptos Técnicos Clave (3 min)

**Cuantización (del material del profesor):**
> "TinyLlama tiene 1.1 billones de parámetros. Sin cuantizar necesita ~2.2GB. Con cuantización a 4 bits: ~550MB. Cabe en cualquier laptop moderna."

**RAG vs. LLM solo:**
> "Si usamos solo TinyLlama sin RAG, el modelo no sabe nada sobre el manual de soporte técnico específico. Con RAG, primero buscamos en el manual y le damos el contexto exacto al modelo."

**Similitud del coseno:**
> "OpenCLIP convierte tanto el texto de la pregunta como los párrafos del manual en vectores de 512 números. Luego calculamos cuán 'cerca' están matemáticamente. El más cercano es el más relevante."

---

### 5. Resultados y Métricas (1 min)

| Métrica | Valor |
|---|---|
| Tiempo STT (Whisper base CPU) | ~3-8 segundos |
| Tiempo búsqueda RAG | ~1-2 segundos |
| Tiempo respuesta TinyLlama | ~15-45 segundos (CPU) |
| Tiempo TTS (Piper) | ~2-4 segundos |
| **Total pipeline** | ~25-60 segundos |
| Problemas cubiertos | 15 categorías, ~180 chunks |
| Uso de RAM estimado | ~4-6 GB |

---

### 6. Conclusión (1 min)

**Puntos de cierre:**
1. **100% local** — ningún dato sale de la máquina
2. **Multimodal** — combina texto, voz, imagen y video
3. **Escalable** — agregar manuales = solo copiar archivos .txt y ejecutar `build_db.py`
4. **Educativo** — cada componente implementa los conceptos de las semanas 1 y 2 del módulo

**Frase final:**
> "Este sistema demuestra que la IA no requiere la nube para ser poderosa. Con los modelos correctos y una arquitectura bien diseñada, podemos construir asistentes inteligentes que respetan la privacidad y funcionan sin internet."

---

## Preguntas frecuentes del tribunal

**¿Por qué TinyLlama y no un modelo más grande?**
> En macOS Intel sin GPU, un modelo más grande como Llama 3.2 3B tardaría 3-5 minutos en responder. TinyLlama responde en 15-45 segundos con calidad aceptable para soporte técnico con el contexto RAG.

**¿Por qué OpenCLIP en lugar de un modelo de embeddings de texto puro?**
> OpenCLIP tiene el mismo espacio de representación para texto e imágenes. Esto permite hacer búsquedas con imágenes directamente en la misma base de datos vectorial que el texto, sin componentes adicionales.

**¿Cómo se agrega nueva documentación al sistema?**
> Copiar el archivo .txt a la carpeta `manuals/` y ejecutar `python rag/build_db.py`. El sistema lo ingesta automáticamente y lo hace disponible en segundos.

**¿Podría funcionar con un modelo más potente?**
> Sí. Solo cambiar `LLM_MODEL = "llama3.2:3b"` en `config.py` y ejecutar `ollama pull llama3.2:3b`. El resto del sistema no requiere cambios.
