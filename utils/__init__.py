"""
utils/
-------
Shared helper functions used across all modules and pages.

Packages included:
  - validators : Input validation (date ranges, strings, amounts).
  - formatters : Currency, percentage and date display helpers.
  - logger     : Centralised logging configuration.

Design rule: Nothing in utils/ should import from modules/ or integrations/.
These are pure utility functions with no business logic dependencies.
"""
