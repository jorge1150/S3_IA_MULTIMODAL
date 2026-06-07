"""
styles.py — CSS personalizado para la interfaz profesional.
Diseño oscuro tecnológico para presentación universitaria.
"""

CSS = """
/* ── Variables globales ───────────────────────── */
:root {
    --primary: #00d4ff;
    --secondary: #7c3aed;
    --bg-dark: #0f1117;
    --bg-card: #1a1d27;
    --bg-input: #12151e;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #2d3748;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,212,255,0.08);
}

/* ── Reset y body ─────────────────────────────── */
body, .gradio-container {
    background: var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Header principal ─────────────────────────── */
.s3-header {
    text-align: center;
    padding: 28px 20px 16px;
    background: linear-gradient(135deg, #0f1117 0%, #1a1d27 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

.s3-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px 0;
}

.s3-header p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Tabs ─────────────────────────────────────── */
.tabs > .tab-nav {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
}
.tabs > .tab-nav button {
    color: var(--text-muted) !important;
    font-weight: 500;
    padding: 10px 20px;
}
.tabs > .tab-nav button.selected {
    color: var(--primary) !important;
    border-bottom: 2px solid var(--primary) !important;
}

/* ── Paneles / Cards ──────────────────────────── */
.panel-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 16px !important;
    box-shadow: var(--shadow) !important;
}

/* ── Labels ───────────────────────────────────── */
label span, .label-wrap span {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Inputs y Textboxes ───────────────────────── */
textarea, input[type="text"] {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

textarea:focus, input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}

/* ── Logs (consola) ───────────────────────────── */
.logs-console textarea {
    background: #0a0c10 !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    border: 1px solid #1a3a2a !important;
}

/* ── Botón principal ──────────────────────────── */
.btn-analyze {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(0,212,255,0.3) !important;
}
.btn-analyze:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,212,255,0.4) !important;
}

/* ── Botón secundario ─────────────────────────── */
.btn-clear {
    background: var(--bg-card) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Status badges ────────────────────────────── */
.badge-status {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-ok   { background: #064e3b; color: #10b981; }
.badge-warn { background: #451a03; color: #f59e0b; }
.badge-err  { background: #450a0a; color: #ef4444; }

/* ── Imagen / Video ───────────────────────────── */
.image-container, .video-container {
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
}

/* ── Audio player ─────────────────────────────── */
audio {
    width: 100% !important;
    border-radius: 8px !important;
    background: var(--bg-input) !important;
}

/* ── Sección de respuesta ─────────────────────── */
.response-box textarea {
    background: #0d1f0d !important;
    color: #86efac !important;
    border: 1px solid #14532d !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}

/* ── Scrollbars ───────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""
