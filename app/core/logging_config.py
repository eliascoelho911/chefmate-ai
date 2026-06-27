import logging
import sys

# Loggers de bibliotecas que ficam muito barulhentos em DEBUG
_NOISY_LOGGERS = [
    "httpcore",
    "httpcore.http11",
    "httpcore.connection",
    "httpx",
    "urllib3",
    "urllib3.connectionpool",
    "openai._base_client",
    "sentence_transformers",
    "transformers",
    "transformers.modeling_utils",
    "transformers.configuration_utils",
]


def _silence_noisy_loggers() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger for the application.
    Level can be overridden via LOG_LEVEL env var.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Avoid duplicate handlers on reload
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers[0] = handler

    # Silence noisy HTTP library loggers while keeping app debug logs
    _silence_noisy_loggers()
