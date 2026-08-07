"""Visual styling for the Streamlit app.

Kept in one place so the layout code stays readable.

The palette is dark-first and matches ``.streamlit/config.toml`` - change one
and you must change the other. Surfaces are translucent white rather than fixed
hex values, so a card sits on whatever background it lands on and picks up its
depth from the layer beneath instead of fighting it.
"""

from __future__ import annotations

#: Accent. Bright enough to read on a dark surface without glowing.
ACCENT = "#58a6ff"
#: Citation pills carry white text, so their fill is a deeper blue than the
#: accent - #58a6ff behind white is only about 2:1 and hard to read small.
CITATION_FILL = "#1f6feb"

CSS = """
<style>
/* Streamlit's own chrome adds noise to what is a single-purpose internal tool. */
[data-testid="stAppDeployButton"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container { padding-top: 2.2rem; max-width: 60rem; }

/* --- header ------------------------------------------------------------
   Everything here is a <div>, never a <p>. Streamlit's own markdown
   stylesheet sets a font-size on `p` with higher specificity than a bare
   class, so a <p class="amd-title"> silently renders at body size. */
.amd-header { margin-bottom: 1.5rem; }
.amd-title {
  font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em;
  line-height: 1.2; margin: 0;
}
.amd-title span { color: #58a6ff; }
.amd-subtitle {
  font-size: 0.9rem; color: #8b949e; margin-top: 0.3rem;
}

/* --- cards ------------------------------------------------------------- */
.amd-card {
  background: rgba(255, 255, 255, 0.035);
  border: 1px solid rgba(255, 255, 255, 0.09);
  border-radius: 14px;
  padding: 1.15rem 1.35rem;
  margin-bottom: 0.75rem;
}
.amd-question {
  font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: #8b949e; margin-bottom: 0.6rem;
}
.amd-answer { font-size: 1.02rem; line-height: 1.75; color: #e6edf3; }

/* A refusal is a real outcome, not a failure - amber, not red, and clearly
   not styled like an answer. */
.amd-card.refusal {
  background: rgba(210, 153, 34, 0.08);
  border-color: rgba(210, 153, 34, 0.32);
}

/* --- badges ------------------------------------------------------------ */
.amd-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 1rem; }
.amd-badge {
  font-size: 0.74rem; font-weight: 500; padding: 0.24rem 0.7rem;
  border-radius: 999px; white-space: nowrap;
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: #8b949e;
}
.amd-badge.good {
  background: rgba(63, 185, 80, 0.14); border-color: rgba(63, 185, 80, 0.38);
  color: #56d364;
}
.amd-badge.warn {
  background: rgba(210, 153, 34, 0.14); border-color: rgba(210, 153, 34, 0.38);
  color: #e3b341;
}
.amd-badge.bad {
  background: rgba(248, 81, 73, 0.14); border-color: rgba(248, 81, 73, 0.38);
  color: #ff7b72;
}
.amd-badge.plain { background: rgba(255, 255, 255, 0.04); }

/* --- meta line --------------------------------------------------------- */
.amd-meta {
  font-size: 0.76rem; color: #6e7681; margin: 0.5rem 0 1.8rem 0.2rem;
  display: flex; flex-wrap: wrap; gap: 1rem;
}

/* --- section label ----------------------------------------------------- */
.amd-label {
  font-size: 0.76rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.07em; color: #6e7681; margin: 1.2rem 0 0.55rem;
}

/* --- sidebar ----------------------------------------------------------- */
.amd-status {
  display: flex; align-items: center; gap: 0.55rem;
  padding: 0.65rem 0.85rem; border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  background: rgba(255, 255, 255, 0.04);
  font-size: 0.86rem; font-weight: 500; color: #e6edf3;
}
.amd-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.amd-dot.ready { background: #3fb950; box-shadow: 0 0 0 3px rgba(63, 185, 80, 0.2); }
.amd-dot.degraded { background: #d29922; box-shadow: 0 0 0 3px rgba(210, 153, 34, 0.2); }
.amd-dot.offline { background: #f85149; box-shadow: 0 0 0 3px rgba(248, 81, 73, 0.2); }
.amd-stat { font-size: 0.78rem; color: #6e7681; margin-top: 0.5rem; line-height: 1.6; }

/* --- empty state ------------------------------------------------------- */
.amd-empty { text-align: center; padding: 3rem 1rem; color: #6e7681; }
.amd-empty-icon { font-size: 2.2rem; line-height: 1; opacity: 0.85; }
.amd-empty-title {
  font-size: 1.05rem; font-weight: 600; color: #8b949e; margin: 0.75rem 0 0.35rem;
}
.amd-empty-hint { font-size: 0.86rem; }

/* Expanders carry the sources; tighten them so a list of six is scannable. */
[data-testid="stExpander"] details {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.02);
}
[data-testid="stExpander"] details:hover { border-color: rgba(255, 255, 255, 0.16); }
[data-testid="stExpander"] summary { font-size: 0.88rem; }

/* The chat box is the primary control; give it a little more presence. */
[data-testid="stChatInput"] {
  border-radius: 12px;
  border-color: rgba(255, 255, 255, 0.12);
}
</style>
"""


def citation_badge_style() -> str:
    """Inline style for a citation pill inside answer text.

    Inline rather than a class: the answer is injected as raw HTML, and a
    stylesheet class would be at the mercy of Streamlit's own resets.
    """
    return (
        f"display:inline-block;margin:0 3px;background:{CITATION_FILL};color:#fff;"
        "border-radius:5px;padding:1px 7px;font-size:0.78em;font-weight:600;"
        "white-space:nowrap;vertical-align:baseline;"
    )
