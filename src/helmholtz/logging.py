import logging


def set_simple_logging(level=logging.WARN):
    """Sets a simple logging format with logging level 'level'."""
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=level, format="%(levelname)-8s %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
