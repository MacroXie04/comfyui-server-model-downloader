from __future__ import annotations

import sys
from pathlib import Path


# ``backend`` is deliberately importable as a namespace package while the
# custom-node package is still being assembled.  This keeps the tests runnable
# both from the repository root and directly from this tests directory.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
