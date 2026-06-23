import logging
import sys


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
