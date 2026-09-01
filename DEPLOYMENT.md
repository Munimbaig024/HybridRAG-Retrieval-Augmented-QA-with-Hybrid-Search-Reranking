# HybridRAG — Deployment Guide (branch: `deployment`)

This doc covers how the demo is packaged and hosted, the showcase design decisions, and the exact setup steps. Keep this file at the root of the `deployment` branch.

---

## 1. Hosting decision

**Platform**: Streamlit Community Cloud (free tier).

Why: connects directly to GitHub, auto-detects `app.py` and `requirements.txt`, deploys in one click with no SDK picker, no hardware tier picker, no Dockerfile. HuggingFace Spaces was tried first but their free tier changed mid-2026 — Docker moved behind a paywall and CPU Basic hardware became unselectable for new accounts (ZeroGPU-only), adding setup friction with no benefit for a CPU-only pipeline. Streamlit Community Cloud avoids all of that.

**Trade-off to know**: the free tier gives ~1GB RAM (vs. HF's old 16GB), apps sleep after ~12 hours of inactivity and cold-start on the next visit, and only one private app is allowed (unlimited public apps). None of this blocks the project — see section 3 for how the pipeline stays lightweight enough to fit.

---

## 2. Showcase design (what the demo actually shows)

Decided against plain "upload any doc and chat" — too generic, doesn't demonstrate what makes this project different. Demo instead does three things:

1. **Default corpus pre-loaded.** The Wikipedia subset used for eval ships with the app so it works instantly with zero setup from the visitor.
2. **Eval stats panel.** A sidebar section showing the project's own precision@k, recall@k, MRR, nDCG, and faithfulness numbers, pulled from `eval/results.json`. This is the proof that the "not just vibes-based testing" claim is real.
3. **Dense-only vs hybrid+rerank toggle.** Same query run through both paths side by side, so a visitor can see hybrid search winning on a lexical query (specific term, number, name) in about 10 seconds, without reading code.
4. **Optional "upload your own doc" mode**, kept secondary/collapsed under the default corpus — not the first thing visitors see.

---

## 3. Staying within the 1GB free-tier RAM budget

- Embedding model: `all-MiniLM-L6-v2` (~80MB) — small, fast, good enough for this corpus size.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB) — same reasoning.
- Default corpus: a few hundred to ~1,000 Wikipedia chunks, not tens of thousands. Keeps the BM25 index and vector store small in memory.
- Avoid loading multiple large models at once; load once at startup and cache with `@st.cache_resource`, not per-query.
- If memory becomes an issue in practice, the first thing to shrink is corpus size, not model choice — the models above are already close to the smallest usable options.

---

## 4. API key & rate limiting

- A **dedicated** Groq API key is generated specifically for this deployment — not the local dev key.
- Stored only in Streamlit Community Cloud's **Secrets** manager (Settings → Secrets in the app dashboard), never committed to the repo, never present in commit history.
- Accessed in code via `st.secrets`, not `os.environ` directly (Streamlit Cloud injects secrets this way):

```python
import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
```

- App-level rate limiting: cap requests per session using `st.session_state` as a counter, reset per browser session.
- Optional: `@st.cache_data` on the query function so identical repeated demo queries don't re-hit the API.

---

## 5. Repo structure

```
deployment/
├── app.py                 # Streamlit entrypoint
├── requirements.txt       # pinned dependencies
├── src/                   # pipeline code (ingest.py, query.py, etc.)
├── data/
│   └── wiki_subset/       # pre-baked default corpus
├── eval/
│   └── results.json       # eval metrics the stats panel reads
└── .streamlit/
    └── config.toml        # optional: theme, layout settings
```

No YAML frontmatter or platform-specific config file needed — Streamlit Community Cloud just needs `app.py` and `requirements.txt` to exist and be importable.

---

## 6. Pre-deploy checklist

- [ ] `requirements.txt` has every dependency pinned.
- [ ] No API keys anywhere in the repo or git history (`git log -p | grep -i groq` as a sanity check).
- [ ] Default corpus is small enough per section 3.
- [ ] `app.py` reads the key via `st.secrets`, not a hardcoded string.
- [ ] Rate limiting logic is in place and tested locally.
- [ ] Local smoke test: `streamlit run app.py` works end to end before pushing.

---

## 7. Post-deploy checklist

- [ ] App builds successfully (check the deploy logs in the Streamlit Cloud dashboard for errors).
- [ ] Secret is set in the app's Secrets manager, not in code.
- [ ] Test the live public URL in an incognito window (catches anything that only worked because of local state/cache).
- [ ] Confirm the app "sleeps" as expected after ~12 hours idle and wakes on the next visit (first load after sleep will be slow — this is normal, not a bug).
- [ ] Link the live app in your portfolio/resume/GitHub README.
