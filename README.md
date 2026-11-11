# ✈️ PayPilot AI

**Financial intelligence for synthetic business-transaction data.**

PayPilot AI is a local, end-to-end Streamlit dashboard that turns a
transaction ledger into clear answers: revenue trends, merchant and
category performance, statistically unusual activity, and a plain-English
AI assistant that shows how every figure was computed.

> ⚠️ **All data is SYNTHETIC.**
> Every transaction is generated locally by `modules/data_generator.py`
> (default: 500 rows, seed 42). No real payments, accounts, merchants or
> financial records are involved. This prototype is not financial advice.

---

## Features

| Page | What it does |
|---|---|
| 🏠 **app** (landing) | Quick KPIs, monthly revenue trend, and navigation cards to every page. |
| 📊 **Dashboard** | KPI cards with month-over-month deltas, revenue trend (daily / weekly / monthly), category mix, status overview, top merchants, recent transactions, and dataset regeneration controls. |
| 🔎 **Transactions** | Full-text search across merchant / category / ID, category / status / payment-method filters, amount-range slider, CSV export. |
| 📈 **Analytics** | Category and merchant rankings (adjustable top-N) and time-series views with current-vs-previous period comparisons. |
| 🚩 **Anomalies** | Statistically unusual transactions with Z-scores, pattern scores, neutral explanations and tunable thresholds. |
| 💬 **AI Assistant** | Natural-language business questions answered from the data, each with a calculation trace. |
| ℹ️ **About** | Project overview, architecture, tech stack, data policy and known limitations. |

Design principles applied throughout:

- **Shared date filter** — one sidebar picker scopes every page.
- **Safe handling of missing data** — friendly empty states instead of errors.
- **Neutral language** — the app talks about *unusual activity* and *flagged
  transactions*; a statistical flag is never presented as an accusation.
- **No fake integrations** — anything not implemented (e.g. the PayPal
  Sandbox connector) is clearly labelled as a stub.
- **Everything is reproducible** — same seed + same size = identical dataset.

## Screenshots

> *Screenshots to be added here* — run the app and capture:
>
> - [ ] Landing page (`app.py`)
> - [ ] Dashboard with KPI cards
> - [ ] Transactions with filters applied
> - [ ] Analytics merchant ranking
> - [ ] Anomalies scatter + flagged table
> - [ ] AI Assistant conversation

## Architecture

```
paypilot-ai/
├── app.py                      # Streamlit entry point / landing page
├── pages/
│   ├── __init__.py             # Shared UI plumbing (CSS, data loading, date filter)
│   ├── 01_Dashboard.py         # KPIs, trends, dataset controls
│   ├── 02_Transactions.py      # Search / filter / export
│   ├── 03_Analytics.py         # Category, merchant, time-based analysis
│   ├── 04_Anomalies.py         # Explainable anomaly detection UI
│   ├── 05_AI_Assistant.py      # Chat-style Q&A
│   └── 06_About.py             # Project information
├── modules/                    # Business logic (pure, testable)
│   ├── data_generator.py       # Synthetic transaction generation + validation
│   ├── database.py             # SQLite persistence + data-source routing
│   ├── analytics.py            # KPIs, breakdowns, time series, period deltas
│   ├── anomaly_detector.py     # Z-score + IsolationForest detection
│   └── ai_assistant.py         # Rule-based intent parsing + answers
├── integrations/
│   └── paypal_sandbox.py       # PayPal Sandbox connector stub (by design)
├── utils/
│   ├── validators.py           # Input validation helpers
│   ├── formatters.py           # Currency / date / colour formatting
│   └── logger.py               # Centralised logging configuration
├── tests/                      # pytest suite + standalone smoke scripts
├── requirements.txt
└── README.md
```

**Layering rules**

- `modules/` contains all calculations — pure functions, no Streamlit code.
- `utils/` contains presentation-independent helpers (validation, formatting,
  logging).
- `pages/` and `app.py` only orchestrate UI; every number on screen comes
  from `modules/`.

### Data model

Each transaction row (stored in `data/paypilot.db`, auto-created):

| Column | Type | Description |
|---|---|---|
| `transaction_id` | str (UUID) | Unique identifier |
| `date` | datetime | Transaction date |
| `merchant` | str | Merchant name (fictional) |
| `category` | str | Business category |
| `amount` | float | Transaction value in USD |
| `payment_method` | str | e.g. card, bank transfer |
| `status` | str | completed / pending / refunded / failed |
| `data_source` | str | Always `SYNTHETIC` |

### Anomaly detection

Two complementary methods combine into one explainable result:

1. **Z-score** — flags amounts whose standard-deviation distance from the
   mean exceeds a threshold (always runs, works on small samples).
2. **IsolationForest** (scikit-learn) — flags unusual combinations of
   amount, day-of-week and month (requires ≥ 20 rows).

Every flagged row carries a plain-English `flag_reason`, a `z_score` and a
`detection_method`. Thresholds are adjustable in the Anomalies page sidebar.

### AI Assistant

Rule-based intent parsing (regex) over a fixed set of question types —
e.g. *“What is my total revenue?”*, *“Which merchant earns the most?”*,
*“Show me unusual transactions”*. Every answer includes the calculation
trace and the intent it matched; unknown questions get an honest
“I don’t know” rather than a guess. The design is LLM-ready: swapping in a
language model later only replaces the answer-generation step.

## Setup (Windows PowerShell)

```powershell
# 1. Navigate to the project directory
cd C:\Users\h2703\OneDrive\Documents\paypilot-ai

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Run the test suite
python -m pytest tests -v          # unit tests (all modules, utils, UI structure)
python tests\smoke_database.py     # end-to-end database round-trip
python tests\smoke_pages.py        # headless render + interaction test of all pages

# 5. Start the Streamlit application
streamlit run app.py
```

Notes:

- Python 3.11+ (developed and tested on 3.13).
- The SQLite database (`data/paypilot.db`) is created automatically on first
  run and can be regenerated from the Dashboard sidebar.
- If PowerShell blocks `Activate.ps1`, run
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
  first, or use `venv\Scripts\activate.bat` from cmd.
- The test suite and smoke scripts work with the system Python too — just
  make sure the dependencies are installed in whichever interpreter you use.

## Testing

| Command | What it checks |
|---|---|
| `python -m pytest tests -v` | Unit tests for every module and utility, plus UI-structure checks (files present, everything compiles, language policy). |
| `python tests\smoke_database.py` | 16-check end-to-end round trip: generate → save → load → clear → restore. |
| `python tests\smoke_pages.py` | Renders all 7 scripts headlessly (Streamlit AppTest), exercises search, filters, sliders, chat and dataset regeneration, then restores the default dataset. |

## Known limitations

- The dataset is **synthetic** — trends and flagged transactions are
  simulated, not real market behaviour.
- The AI assistant is rule-based; it answers a fixed set of question types
  and says so when it cannot.
- The PayPal Sandbox connector is an intentional stub — the interface is
  defined, but every method raises `NotImplementedError`. No credentials are
  required and no network calls are made.
- Anomaly detection is statistical screening for review, not an audit tool.
- Data volumes are prototype-scale (hundreds to low thousands of rows).
