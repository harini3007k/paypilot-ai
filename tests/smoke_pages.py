"""
Smoke test script for the Streamlit UI layer (app.py + pages/).

Run from the project root:  python tests/smoke_pages.py

Renders every script headlessly with streamlit.testing.v1.AppTest, checks
that none of them raises an exception, exercises the main interactions
(search, filters, sliders, chat, dataset regeneration) and finally restores
the default synthetic dataset (500 rows, seed 42).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Every script that must render cleanly (app.py + the six feature pages).
SCRIPTS = [
    "app.py",
    "pages/01_Dashboard.py",
    "pages/02_Transactions.py",
    "pages/03_Analytics.py",
    "pages/04_Anomalies.py",
    "pages/05_AI_Assistant.py",
    "pages/06_About.py",
]

passes = 0
failures = []


def check(n, condition, msg=""):
    global passes
    if condition:
        print(f"  PASS {n:>2}: {msg}")
        passes += 1
    else:
        print(f"  FAIL {n:>2}: {msg}")
        failures.append(n)


def render(script, timeout=180):
    """Run one script headlessly and return (AppTest, exception_list)."""
    at = AppTest.from_file(str(ROOT / script), default_timeout=timeout)
    at.run()
    return at, list(at.exception)


print(f"Project root: {ROOT}")
print(f"Rendering {len(SCRIPTS)} scripts headlessly…\n")

# ---------------------------------------------------------------- Section 1
# Every page renders without raising an exception.
# -------------------------------------------------------------------------
at, errs = render("app.py")
check(1, not errs, f"app.py renders without exceptions {errs[:1]}")
check(2, len(at.metric) == 3, f"app.py shows 3 KPI metrics (got {len(at.metric)})")
check(
    3,
    any("SYNTHETIC" in m.value for m in at.markdown),
    "app.py shows a visible SYNTHETIC data badge",
)
check(4, len(at.button) == 6, f"app.py shows 6 navigation buttons (got {len(at.button)})")

at, errs = render("pages/01_Dashboard.py")
check(5, not errs, f"Dashboard renders without exceptions {errs[:1]}")
check(6, len(at.metric) == 4, f"Dashboard shows 4 KPI metrics (got {len(at.metric)})")
check(7, len(at.date_input) == 1, "Dashboard renders the shared date filter")
check(8, len(at.segmented_control) == 1, "Dashboard renders the granularity toggle")

at, errs = render("pages/02_Transactions.py")
check(9, not errs, f"Transactions renders without exceptions {errs[:1]}")
check(10, len(at.slider) == 1, "Transactions renders the amount slider")

at, errs = render("pages/03_Analytics.py")
check(11, not errs, f"Analytics renders without exceptions {errs[:1]}")

at, errs = render("pages/04_Anomalies.py")
check(12, not errs, f"Anomalies renders without exceptions {errs[:1]}")
check(13, len(at.metric) == 4, f"Anomalies shows 4 summary metrics (got {len(at.metric)})")

at, errs = render("pages/05_AI_Assistant.py")
check(14, not errs, f"AI Assistant renders without exceptions {errs[:1]}")

at, errs = render("pages/06_About.py")
check(15, not errs, f"About renders without exceptions {errs[:1]}")

# ---------------------------------------------------------------- Section 2
# Interactions — every main widget is used at least once.
# -------------------------------------------------------------------------

# --- Dashboard: granularity toggle + dataset regeneration ------------------
at = AppTest.from_file(str(ROOT / "pages/01_Dashboard.py"), default_timeout=180)
at.run()
at.segmented_control[0].set_value("Daily")
at.run()
check(16, not list(at.exception), "Dashboard: switching granularity to Daily works")

at.number_input[0].set_value(300)
at.button[0].click()  # 🎲 Regenerate dataset
at.run()
check(17, not list(at.exception), "Dashboard: regenerating the dataset (300 rows) works")
check(
    18,
    len(at.metric) == 4,
    "Dashboard: KPIs re-render after regeneration",
)

# --- Transactions: search narrows, then empty state on no match ------------
at = AppTest.from_file(str(ROOT / "pages/02_Transactions.py"), default_timeout=180)
at.run()
rows_before = int(at.metric[0].value.replace(",", ""))

at.text_input[0].set_value("Slack")
at.run()
check(19, not list(at.exception), "Transactions: search 'Slack' runs cleanly")
rows_search = int(at.metric[0].value.replace(",", ""))
check(20, rows_search <= rows_before, f"Transactions: search narrows results ({rows_before} -> {rows_search})")

at.text_input[0].set_value("zzz-no-such-merchant")
at.run()
check(21, not list(at.exception), "Transactions: no-match search runs cleanly")
check(
    22,
    any("No transactions match" in i.value for i in at.info),
    "Transactions: no-match search shows a friendly empty state",
)

# --- Analytics: merchant slider --------------------------------------------
at = AppTest.from_file(str(ROOT / "pages/03_Analytics.py"), default_timeout=180)
at.run()
at.slider[0].set_value(15)
at.run()
check(23, not list(at.exception), "Analytics: merchant-ranking slider (15) works")

# --- Anomalies: both threshold sliders -------------------------------------
at = AppTest.from_file(str(ROOT / "pages/04_Anomalies.py"), default_timeout=180)
at.run()
flagged_before = int(at.metric[1].value.replace(",", ""))
at.slider[0].set_value(0.10)  # sensitivity (contamination)
at.slider[1].set_value(2.0)  # z-score threshold
at.run()
check(24, not list(at.exception), "Anomalies: changing both thresholds works")
flagged_after = int(at.metric[1].value.replace(",", ""))
check(
    25,
    flagged_after > flagged_before,
    f"Anomalies: loosening thresholds flags more transactions "
    f"({flagged_before} -> {flagged_after})",
)

# --- AI Assistant: two real questions, one per major intent -----------------
at = AppTest.from_file(str(ROOT / "pages/05_AI_Assistant.py"), default_timeout=180)
at.run()
at.chat_input[0].set_value("What is my total revenue?")
at.run()
check(26, not list(at.exception), "AI Assistant: 'What is my total revenue?' works")
chat = at.session_state["pp_chat"]
check(27, len(chat) == 2 and chat[0]["role"] == "user", "AI Assistant: history records the question")
check(28, "$" in chat[1]["response"]["answer"], "AI Assistant: revenue answer contains a currency figure")
check(29, chat[1]["response"]["intent"] == "total_revenue", "AI Assistant: parsed as total_revenue intent")

at.chat_input[0].set_value("Which merchant earns the most?")
at.run()
check(30, not list(at.exception), "AI Assistant: 'Which merchant earns the most?' works")
chat = at.session_state["pp_chat"]
check(
    31,
    chat[-1]["response"]["intent"] == "top_merchants",
    "AI Assistant: parsed as top_merchants intent (regression guard)",
)
check(
    32,
    bool(chat[-1]["response"]["answer"].strip()),
    "AI Assistant: merchant answer is non-empty",
)

# --- Landing page navigation button ----------------------------------------
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.run()
at.button[0].click()  # "Open Dashboard"
at.run()
check(33, not list(at.exception), "app.py: navigation button switches to the Dashboard")

# ---------------------------------------------------------------- Section 3
# Restore the default dataset (500 rows, seed 42) and verify.
# -------------------------------------------------------------------------
from modules.database import ensure_synthetic_data, transaction_count

ensure_synthetic_data(n=500, seed=42, force_regenerate=True)
check(34, transaction_count() == 500, f"Dataset restored to 500 rows (got {transaction_count()})")

# ---------------------------------------------------------------- Summary
print(f"\n{'=' * 60}")
print(f"SMOKE PAGES RESULT: {passes} passed, {len(failures)} failed")
print(f"{'=' * 60}")

if failures:
    print(f"Failed checks: {failures}")
    sys.exit(1)
print("All UI smoke checks passed. ✅")
