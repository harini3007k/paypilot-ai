"""
integrations/
--------------
Connectors to external data sources and payment platforms.

Packages included:
  - paypal_sandbox  : STUB for future PayPal Sandbox API connection.
                      Not connected to any real PayPal account.
                      Synthetic data is NEVER mixed with PayPal data here.

Design rule: Every connector must return a pandas DataFrame with the same
column schema as the synthetic data, so the rest of the app stays unchanged
regardless of which data source is active.
"""
