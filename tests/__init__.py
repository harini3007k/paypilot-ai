"""
tests/
-------
Unit tests for PayPilot AI using pytest.

Test files:
  test_data_generator.py    : Data shape, column types, value ranges.
  test_analytics.py          : KPI calculations, period deltas, breakdowns.
  test_anomaly.py            : Anomaly output structure and explanation text.
  test_ai_assistant.py       : Intent parsing and answer structure.
  test_utils_validators.py   : Input validation edge cases (utils/validators.py).
  test_utils_formatters.py   : Display formatting helpers (utils/formatters.py).
  test_utils_logger.py       : Logging configuration (utils/logger.py).
  test_app_structure.py      : UI layer layout, compile + language policy.

Standalone smoke scripts (not collected by pytest — run directly):
  smoke_database.py         : End-to-end database round-trip (16 checks).
  smoke_pages.py            : Headless render + interaction test of all pages.

Run all tests from the project root with:
    pytest tests/ -v
"""
