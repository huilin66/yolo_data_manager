"""Standalone YDM web application support.

The web layer is intentionally optional.  Importing :mod:`yolo_data_manager`
does not require FastAPI or a JavaScript runtime; those dependencies are only
needed when ``ydm web`` is used.
"""

__all__ = ["create_app"]


def create_app():
    """Create the FastAPI application lazily."""

    from yolo_data_manager.web.server import create_app as _create_app

    return _create_app()
