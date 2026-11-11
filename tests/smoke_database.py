"""
Smoke test script for modules/database.py and integrations/paypal_sandbox.py
Run from the project root:  python tests/smoke_database.py
"""
import sys, tempfile, pathlib, shutil
sys.path.insert(0, r"c:\Users\h2703\OneDrive\Documents\paypilot-ai")

import pandas as pd
from modules.database import (
    DataSource, init_db, save_transactions, load_transactions,
    ensure_synthetic_data, clear_transactions, transaction_count,
    get_data_source_label,
)
from modules.data_generator import generate_transactions
from integrations.paypal_sandbox import (
    PayPalSandboxConnector, fetch_transactions, check_configuration,
)

tmp_dir = pathlib.Path(tempfile.mkdtemp())
tmp = tmp_dir / "test_paypilot.db"
print(f"Using temp DB: {tmp}\n")

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

# 1. transaction_count on missing DB
check(1, transaction_count(tmp) == 0, "transaction_count on missing DB -> 0")

# 2. init_db creates the file
init_db(tmp)
check(2, tmp.exists() and transaction_count(tmp) == 0, "init_db creates file, 0 rows")

# 3. ensure_synthetic_data seeds DB
df = ensure_synthetic_data(n=100, seed=42, db_path=tmp)
check(3, len(df) == 100 and transaction_count(tmp) == 100, f"ensure_synthetic_data -> {len(df)} rows")

# 4. ensure_synthetic_data skips regeneration when data exists
df2 = ensure_synthetic_data(n=200, seed=99, db_path=tmp)
check(4, len(df2) == 100, "skips regeneration when data already present")

# 5. force_regenerate replaces data
df3 = ensure_synthetic_data(n=200, seed=99, db_path=tmp, force_regenerate=True)
check(5, len(df3) == 200 and transaction_count(tmp) == 200, f"force_regenerate -> {len(df3)} rows")

# 6. load_transactions returns correct tuple
loaded_df, source = load_transactions(DataSource.SYNTHETIC, tmp)
check(6,
    source == DataSource.SYNTHETIC
    and len(loaded_df) == 200
    and list(loaded_df.columns) == ["transaction_id","date","merchant","category","amount","payment_method","status","data_source"]
    and pd.api.types.is_datetime64_any_dtype(loaded_df["date"])
    and (loaded_df["data_source"] == "SYNTHETIC").all(),
    f"load_transactions -> {len(loaded_df)} rows, datetime parsed, source={source}"
)

# 7. DataSource str serialisation — .value gives the clean string
check(7,
    DataSource.SYNTHETIC.value == "SYNTHETIC"
    and DataSource("PAYPAL_SANDBOX") == DataSource.PAYPAL_SANDBOX,
    "DataSource.value serialisation works"
)

# 8. get_data_source_label
label = get_data_source_label(DataSource.SYNTHETIC)
check(8,
    "label" in label and "description" in label and "colour" in label
    and "SIMULATED" in label["label"],
    "get_data_source_label returns correct keys and label"
)

# 9. clear_transactions
clear_transactions(tmp)
check(9, transaction_count(tmp) == 0, "clear_transactions empties table")

# 10. load_transactions raises RuntimeError on empty DB
try:
    load_transactions(DataSource.SYNTHETIC, tmp)
    check(10, False, "should have raised RuntimeError on empty DB")
except RuntimeError:
    check(10, True, "RuntimeError raised on empty DB")

# 11. load_transactions raises FileNotFoundError on missing DB
missing = tmp_dir / "does_not_exist.db"
try:
    load_transactions(DataSource.SYNTHETIC, missing)
    check(11, False, "should have raised FileNotFoundError")
except FileNotFoundError:
    check(11, True, "FileNotFoundError raised on missing DB")

# 12. save_transactions rejects invalid data
bad_df = generate_transactions(n=10, seed=0)
bad_df.loc[0, "amount"] = -99.0
try:
    save_transactions(bad_df, db_path=tmp)
    check(12, False, "should have raised ValueError on invalid data")
except ValueError:
    check(12, True, "ValueError raised on invalid DataFrame")

# 13. PayPalSandboxConnector raises NotImplementedError
try:
    PayPalSandboxConnector()
    check(13, False, "should have raised NotImplementedError")
except NotImplementedError:
    check(13, True, "PayPalSandboxConnector raises NotImplementedError")

# 14. module-level fetch_transactions raises NotImplementedError
try:
    fetch_transactions()
    check(14, False, "should have raised NotImplementedError")
except NotImplementedError:
    check(14, True, "fetch_transactions() raises NotImplementedError")

# 15. check_configuration does not raise, returns correct keys
cfg = check_configuration()
check(15,
    "is_configured" in cfg
    and "missing_vars" in cfg
    and "present_vars" in cfg
    and "setup_url" in cfg
    and cfg["is_configured"] is False,
    "check_configuration returns all required keys, is_configured=False"
)

# 16. load_transactions(PAYPAL_SANDBOX) routes to stub -> NotImplementedError
init_db(tmp)
ensure_synthetic_data(n=10, seed=0, db_path=tmp)
try:
    load_transactions(DataSource.PAYPAL_SANDBOX, tmp)
    check(16, False, "should have raised NotImplementedError for PAYPAL_SANDBOX")
except NotImplementedError:
    check(16, True, "load_transactions(PAYPAL_SANDBOX) raises NotImplementedError")

# Cleanup — use ignore_errors because Windows holds SQLite file locks briefly
shutil.rmtree(tmp_dir, ignore_errors=True)

# Summary
print(f"\n{'='*50}")
print(f"Results: {passes}/16 passed, {len(failures)} failed")
if failures:
    print(f"Failed tests: {failures}")
    sys.exit(1)
else:
    print("All smoke tests passed.")
    sys.exit(0)
