"""
modules/
--------
Core business logic for PayPilot AI.

Packages included:
  - data_generator  : Generates realistic synthetic transaction data.
  - database        : SQLite read/write + DataSource abstraction.
  - analytics       : KPIs, trends, period comparisons, breakdowns.
  - anomaly_detector: IsolationForest + Z-score flagging with explanations.
  - ai_assistant    : Intent-based Q&A over transaction data (LLM-ready).
"""
