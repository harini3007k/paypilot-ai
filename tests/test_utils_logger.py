"""
tests/test_utils_logger.py
--------------------------
Tests for utils/logger.py — centralised logging configuration.

Run with:
    pytest tests/test_utils_logger.py -v

Because the "paypilot" logger is global state shared across the whole
process, every test runs inside the ``isolated_paypilot_logger`` fixture,
which snapshots the logger's handlers / level / propagate flag beforehand
and restores them afterwards.  Without this, ordering against the other
test modules (which import pages/ and configure logging at import time)
would be unpredictable.

Covers
------
  TestConfigureLogging – idempotency, handler wiring, propagate=False
  TestGetLogger        – child naming and pass-through behaviour
  TestSetLogLevel      – runtime level changes (int and string forms)
  TestCoerceLevel      – private string/int normalisation
  TestRemoveHandlers   – handler detachment and safety on empty loggers
  TestLogEmission      – records actually flow through the namespace
"""

import logging

import pytest

from utils.logger import (
    DATE_FORMAT,
    LOG_FORMAT,
    LOGGER_NAME,
    _coerce_level,
    configure_logging,
    get_logger,
    remove_handlers,
    set_log_level,
)


@pytest.fixture
def isolated_paypilot_logger():
    """
    Isolate the global "paypilot" logger for one test, then restore it.

    Saves the current handlers, level and propagate flag, strips the
    handlers so each test starts from a pristine logger, and puts
    everything back afterwards.
    """
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    remove_handlers()
    yield logger
    remove_handlers()
    for handler in saved_handlers:
        logger.addHandler(handler)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_returns_the_paypilot_logger(self, isolated_paypilot_logger):
        result = configure_logging()
        assert isinstance(result, logging.Logger)
        assert result.name == LOGGER_NAME

    def test_installs_exactly_one_handler(self, isolated_paypilot_logger):
        logger = configure_logging()
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_idempotent_no_duplicate_handlers(self, isolated_paypilot_logger):
        first = configure_logging()
        handler_before = logger_handler_ids = [
            id(h) for h in isolated_paypilot_logger.handlers
        ]
        second = configure_logging()
        configure_logging()  # and a third time for good measure

        assert first is second
        assert len(isolated_paypilot_logger.handlers) == 1
        assert [id(h) for h in isolated_paypilot_logger.handlers] == handler_before

    def test_handler_uses_the_module_format(self, isolated_paypilot_logger):
        configure_logging()
        handler = isolated_paypilot_logger.handlers[0]
        formatter = handler.formatter
        assert formatter._fmt == LOG_FORMAT
        assert formatter.datefmt == DATE_FORMAT

    def test_propagate_is_disabled(self, isolated_paypilot_logger):
        configure_logging()
        assert isolated_paypilot_logger.propagate is False

    def test_sets_int_level(self, isolated_paypilot_logger):
        configure_logging(logging.DEBUG)
        assert isolated_paypilot_logger.level == logging.DEBUG

    def test_sets_string_level(self, isolated_paypilot_logger):
        configure_logging("WARNING")
        assert isolated_paypilot_logger.level == logging.WARNING

    def test_default_level_is_info(self, isolated_paypilot_logger):
        configure_logging()
        assert isolated_paypilot_logger.level == logging.INFO


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_child_name_is_namespaced(self, isolated_paypilot_logger):
        assert get_logger("pages").name == "paypilot.pages"

    def test_namespaced_name_passes_through(self, isolated_paypilot_logger):
        assert get_logger("paypilot.pages").name == "paypilot.pages"

    def test_nested_names_pass_through(self, isolated_paypilot_logger):
        assert get_logger("paypilot.modules.database").name == (
            "paypilot.modules.database"
        )

    def test_child_is_effective_child_of_paypilot(self, isolated_paypilot_logger):
        child = get_logger("anomaly")
        assert child.parent is logging.getLogger(LOGGER_NAME)

    def test_child_uses_parent_effective_level(self, isolated_paypilot_logger):
        configure_logging(logging.ERROR)
        child = get_logger("smoke")
        assert child.getEffectiveLevel() == logging.ERROR

    def test_returns_logger_instances(self, isolated_paypilot_logger):
        assert isinstance(get_logger("anything"), logging.Logger)


# ---------------------------------------------------------------------------
# set_log_level
# ---------------------------------------------------------------------------


class TestSetLogLevel:
    def test_accepts_logging_constant(self, isolated_paypilot_logger):
        configure_logging()
        set_log_level(logging.DEBUG)
        assert isolated_paypilot_logger.level == logging.DEBUG

    def test_accepts_string(self, isolated_paypilot_logger):
        configure_logging()
        set_log_level("CRITICAL")
        assert isolated_paypilot_logger.level == logging.CRITICAL

    def test_back_to_info(self, isolated_paypilot_logger):
        configure_logging()
        set_log_level("ERROR")
        set_log_level("INFO")
        assert isolated_paypilot_logger.level == logging.INFO


# ---------------------------------------------------------------------------
# _coerce_level (private helper)
# ---------------------------------------------------------------------------


class TestCoerceLevel:
    def test_int_passes_through(self):
        assert _coerce_level(logging.INFO) == logging.INFO
        assert _coerce_level(10) == 10

    def test_uppercase_string(self):
        assert _coerce_level("INFO") == logging.INFO
        assert _coerce_level("ERROR") == logging.ERROR

    def test_lowercase_string(self):
        assert _coerce_level("info") == logging.INFO
        assert _coerce_level("warning") == logging.WARNING


# ---------------------------------------------------------------------------
# remove_handlers
# ---------------------------------------------------------------------------


class TestRemoveHandlers:
    def test_removes_configured_handlers(self, isolated_paypilot_logger):
        configure_logging()
        assert isolated_paypilot_logger.handlers
        remove_handlers()
        assert isolated_paypilot_logger.handlers == []

    def test_safe_on_handlerless_logger(self, isolated_paypilot_logger):
        remove_handlers()  # must not raise
        remove_handlers()
        assert isolated_paypilot_logger.handlers == []


# ---------------------------------------------------------------------------
# Emission — verify records really flow through the namespace
# ---------------------------------------------------------------------------


class TestLogEmission:
    def test_child_record_reaches_paypilot_handlers(
        self, isolated_paypilot_logger
    ):
        configure_logging(logging.INFO)

        captured: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        capture = CaptureHandler()
        isolated_paypilot_logger.addHandler(capture)
        try:
            get_logger("emission").info("hello %s", "world")
        finally:
            isolated_paypilot_logger.removeHandler(capture)

        assert len(captured) == 1
        record = captured[0]
        assert record.name == "paypilot.emission"
        assert record.levelname == "INFO"
        assert record.getMessage() == "hello world"

    def test_level_filters_child_records(self, isolated_paypilot_logger):
        configure_logging(logging.WARNING)

        captured: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record)

        capture = CaptureHandler()
        isolated_paypilot_logger.addHandler(capture)
        try:
            child = get_logger("filtered")
            child.debug("too chatty")
            child.warning("important")
        finally:
            isolated_paypilot_logger.removeHandler(capture)

        assert [r.getMessage() for r in captured] == ["important"]
