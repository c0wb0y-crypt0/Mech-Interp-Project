Grok Polarity Probing

Black-Box Mechanistic Interpretability on Grok Models via Valence Steering

Exploring latent stance manifolds in Grok-4.1 through polarity forcing (defensive/cautious vs aggressive/bold) + temperature variance

Behavioral evidence for "semiotic relativity" in truth-seeking models—meaning distributions bend with prompt frame, temp, and upgrades.
Inspired by 2025 generalizability theory and open problems in mech interp. Data from frontier Grok-4.1 fast API—unique truth-seeking probe.

Thread Discussion (INSERT_THREAD_LINK_HERE) | By @C0wb0y_Crypt0

Overview

This repo contains:
Scripts for polarity probing Grok models via xAI API.
Variance batches (temp 0.0-1.0, repeats).
Analyzer for uniqueness % + stance clustering.
Datasets revealing multi-modality (e.g., aggressive stance shifts on controversial topics).

Core Idea: 
Valence steering acts as "observer frame" in latent semiotic space—revealing relative meaning clusters (bearish/bullish/neutral) anchored by truth prior.

Key findings:
Defensive: Consistent caution/hedging across temps/models.
Aggressive: Multi-modal on crypto ("Bitcoin future")—bearish dominant, temp cracks neutral/optimistic.
Mech interp ("unlock AGI safety"): Grok-4.1 aggressive inverts skeptical vs Grok-4 optimistic—upgrade tightens priors.
Temp modulates access: Low = locked dominant (skeptical), high = rarer paths open.

Black-box behavioral relativity—hypotheses for white-box universality/generalizability.









Installation & Setup

git clone https://github.com/yourusername/grok-polarity-probing.git
cd grok-polarity-probing
pip install openai pandas python-dotenv

Create .env:

XAI_API_KEY=your_api_key_here

Usage
Edit statements.csv (one column: statement).
Run probing:
python polarity_probes.py
Configures temps/repeats auto—saves separate CSVs.
Analyze
python advanced_analyzer.py  # Auto-all CSVs or pass filename

Outputs uniqueness %, stance variants, thread-ready stats.

Data & Results
data/ folder contains variance CSVs:
results_fast_temp*.csv: 20 repeats × temps 0.0-1.0 on Bitcoin + Mech Interp statements.

Highlights (Grok-4.1 fast):
Bitcoin aggressive: Bearish ~65-80%, neutral bleed with temp, rare bullish.
Mech interp aggressive: Skeptical lock 95-100% (temp barely cracks).

Full analyzer summaries in repo.

Contributing
Contributions welcome! Ideas:
Add other models/APIs (Claude, GPT).
Stance classifier improvements.
Visualization notebooks.
Universality task probes.

Open issue or PR.

LicenseMIT License—free to use/fork.
Thanks for checking it out! Feedback/questions: Reply on thread or open issue.#MechInterp #AISafety #GrokProbing
