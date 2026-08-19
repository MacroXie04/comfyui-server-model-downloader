"""Cloudflare Access-protected server model downloader for ComfyUI."""

from .backend import REGISTERED_API


NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
WEB_DIRECTORY = "./frontend"


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REGISTERED_API",
    "WEB_DIRECTORY",
]
