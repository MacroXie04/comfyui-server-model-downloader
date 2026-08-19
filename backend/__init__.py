"""Server Model Downloader backend for ComfyUI."""

from .api import REGISTERED_API, ServerModelDownloaderAPI, register_routes

# This package contributes HTTP routes rather than graph nodes. The custom-node
# package root is responsible for exporting the sibling frontend directory.
NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REGISTERED_API",
    "ServerModelDownloaderAPI",
    "register_routes",
]
