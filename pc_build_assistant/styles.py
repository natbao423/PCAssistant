"""Custom CSS for the Gradio UI.

Light Notion-inspired theme with an animated aurora-mesh page background
and a frosted-glass container.
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-page: #fafafa;
    --bg-surface: #ffffff;
    --bg-surface-hover: #f5f5f5;
    --bg-muted: #f0f0ef;
    --bg-user-bubble: #f7f6f3;
    --bg-bot-bubble: #ffffff;
    --accent: #2383e2;
    --accent-soft: rgba(35, 131, 226, 0.08);
    --accent-hover: #1b6ec2;
    --text-primary: #1a1a1a;
    --text-secondary: #6b6b6b;
    --text-tertiary: #9b9b9b;
    --border: #e8e8e6;
    --border-focus: #2383e2;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 2px 8px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
    --radius-pill: 24px;
    --code-inline: #d63384;
}

/* ── Dark palette ────────────────────────────────────────────────
   Redefining the same variable names under .dark means every rule
   below that uses var(--bg-surface), var(--text-primary), etc.
   adapts automatically — no duplicated element rules needed. */
.dark {
    --bg-page: #0f0f11;
    --bg-surface: #1a1a1d;
    --bg-surface-hover: #232327;
    --bg-muted: #232327;
    --bg-user-bubble: #26262b;
    --bg-bot-bubble: #1a1a1d;
    --accent: #4a9eff;
    --accent-soft: rgba(74, 158, 255, 0.12);
    --accent-hover: #6bb0ff;
    --text-primary: #ececef;
    --text-secondary: #a0a0a8;
    --text-tertiary: #6e6e78;
    --border: #2e2e34;
    --border-focus: #4a9eff;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.30);
    --shadow-md: 0 2px 8px rgba(0, 0, 0, 0.40);
    --shadow-lg: 0 4px 16px rgba(0, 0, 0, 0.50);
    --code-inline: #ff8ac0;
}

/* ── Gradio's own theme variables — light ─────────────────────── */
:root {
    --body-background-fill: transparent;
    --body-text-color: #1a1a1a;
    --body-text-color-subdued: #6b6b6b;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #f7f6f3;
    --block-background-fill: #ffffff;
    --block-border-color: #e8e8e6;
    --block-label-background-fill: #f0f0ef;
    --block-label-text-color: #6b6b6b;
    --block-title-text-color: #1a1a1a;
    --border-color-primary: #e8e8e6;
    --input-background-fill: #ffffff;
    --input-border-color: #e8e8e6;
    --input-placeholder-color: #9b9b9b;
    --button-secondary-background-fill: #ffffff;
    --button-secondary-text-color: #6b6b6b;
    --button-secondary-border-color: #e8e8e6;
}

/* ── Gradio's own theme variables — dark ──────────────────────── */
.dark {
    --body-background-fill: transparent;
    --body-text-color: #ececef;
    --body-text-color-subdued: #a0a0a8;
    --background-fill-primary: #1a1a1d;
    --background-fill-secondary: #232327;
    --block-background-fill: #1a1a1d;
    --block-border-color: #2e2e34;
    --block-label-background-fill: #232327;
    --block-label-text-color: #a0a0a8;
    --block-title-text-color: #ececef;
    --border-color-primary: #2e2e34;
    --input-background-fill: #1a1a1d;
    --input-border-color: #2e2e34;
    --input-placeholder-color: #6e6e78;
    --button-secondary-background-fill: #1a1a1d;
    --button-secondary-text-color: #a0a0a8;
    --button-secondary-border-color: #2e2e34;
}

/* Header text follows the theme (set as classes, not inline hex). */
.pcba-title   { font-size: 28px; font-weight: 700; color: var(--text-primary); }
.pcba-tagline { font-size: 14px; font-weight: 400; color: var(--text-secondary); }
.pcba-ph-main { font-size: 18px; color: var(--text-tertiary); }
.pcba-ph-sub  { font-size: 13px; color: var(--text-tertiary); }

/* ── Page ───────────────────────────────────────────────────── */
body {
    background: linear-gradient(135deg, #e0e7ff 0%, #f0e6fa 25%, #fce7f3 50%, #dbeafe 75%, #e0f2fe 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Dark: deep slate base instead of pastels. */
body.dark, .dark body, html.dark body {
    background: linear-gradient(135deg, #0d0d12 0%, #141420 25%, #1a1424 50%, #101828 75%, #0c1420 100%) !important;
}

body::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(168, 130, 255, 0.20) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 85% 15%, rgba(96, 165, 250, 0.18) 0%, transparent 55%),
        radial-gradient(ellipse 60% 70% at 50% 80%, rgba(244, 163, 187, 0.16) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 75% 60%, rgba(129, 230, 217, 0.12) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: aurora-drift 20s ease-in-out infinite alternate;
}

/* Dark: richer, more saturated blobs so they read against slate. */
.dark body::before, body.dark::before, html.dark body::before {
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(129, 90, 220, 0.28) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 85% 15%, rgba(45, 110, 200, 0.26) 0%, transparent 55%),
        radial-gradient(ellipse 60% 70% at 50% 80%, rgba(180, 70, 120, 0.20) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 75% 60%, rgba(40, 160, 150, 0.18) 0%, transparent 50%) !important;
}

@keyframes aurora-drift {
    0%   { transform: scale(1)   translate(0, 0); }
    33%  { transform: scale(1.05) translate(-2%, 1%); }
    66%  { transform: scale(0.97) translate(1%, -1.5%); }
    100% { transform: scale(1.02) translate(-1%, 0.5%); }
}

.gradio-container {
    background: rgba(255, 255, 255, 0.70) !important;
    backdrop-filter: blur(24px) saturate(1.4) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.4) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    max-width: 860px !important;
    margin: 24px auto !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid rgba(255, 255, 255, 0.50) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.06),
        0 1px 3px rgba(0, 0, 0, 0.04) !important;
    position: relative;
    z-index: 1;
}

/* Dark: smoked glass instead of frosted white. */
.dark .gradio-container, .gradio-container.dark {
    background: rgba(20, 20, 24, 0.72) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.45),
        0 1px 3px rgba(0, 0, 0, 0.30) !important;
}

/* ── Blocks / panels ────────────────────────────────────────── */
.gr-block.gr-box {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Chat bubbles ───────────────────────────────────────────── */
.message.user {
    background: var(--bg-user-bubble) !important;
    border: none !important;
    border-radius: var(--radius-lg) var(--radius-lg) 6px var(--radius-lg) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    box-shadow: var(--shadow-sm) !important;
}

.message.bot {
    background: var(--bg-bot-bubble) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 6px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Input box ──────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: var(--bg-surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}

textarea::placeholder, input[type="text"]::placeholder {
    color: var(--text-tertiary) !important;
}

/* ── Primary button ─────────────────────────────────────────── */
button.primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

button.primary:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

/* ── Secondary button ───────────────────────────────────────── */
button.secondary {
    background: var(--bg-surface) !important;
    color: var(--text-secondary) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}

button.secondary:hover {
    background: var(--bg-surface-hover) !important;
    border-color: var(--text-tertiary) !important;
}

/* ── Example chips ──────────────────────────────────────────── */
.gr-examples button {
    background: var(--bg-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-pill) !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
}

.gr-examples button:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-tertiary); }

/* ── Code blocks inside chat ────────────────────────────────── */
.message code {
    background: var(--bg-muted) !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 2px 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    color: var(--code-inline) !important;
}

.message pre {
    background: #2b2b2b !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px 16px !important;
    box-shadow: var(--shadow-sm) !important;
}

.message pre code {
    background: transparent !important;
    color: #e0e0e0 !important;
    padding: 0 !important;
}

/* ── Tables inside chat ─────────────────────────────────────── */
.message table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    font-size: 13px !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden !important;
}

.message th {
    background: var(--bg-muted) !important;
    color: var(--text-secondary) !important;
    padding: 10px 14px !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
}

.message td {
    padding: 8px 14px !important;
    border-bottom: 1px solid var(--border) !important;
}

.message tr:last-child td {
    border-bottom: none !important;
}

/* ── Footer ─────────────────────────────────────────────────── */
footer { display: none !important; }
"""