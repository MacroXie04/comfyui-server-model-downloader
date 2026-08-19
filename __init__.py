"""Server Model Downloader extension entry point for ComfyUI."""

# ComfyUI imports custom-node directories as packages, while pytest can collect
# this file as a top-level ``__init__`` module when the checkout directory has a
# non-importable name (for example a hyphenated Git repository name).  Supporting
# both import shapes keeps packaging smoke tests faithful without weakening the
# normal package-relative path.
if __package__:
    from .backend import REGISTERED_API
else:  # pragma: no cover - exercised only by top-level test collection.
    from backend import REGISTERED_API


NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}
WEB_DIRECTORY = "./frontend"


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REGISTERED_API",
    "WEB_DIRECTORY",
]
