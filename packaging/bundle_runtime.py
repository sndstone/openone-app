"""Bundle an inference runtime into the app folder (offline installers).

python packaging/bundle_runtime.py VARIANT DEST [PLATFORM]     e.g.  cpu dist/OpenOne/runtime   or   mps dist/OpenOne.app/Contents/Resources/runtime
Same pinned python-build-standalone + wheels as the in-app installer (app/openone_app/runtime.py).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "app"), str(ROOT / "core")]
from openone_app import runtime  # noqa: E402

variant, dest = sys.argv[1], sys.argv[2]
last = [None]


def progress(**kw):
    key = (kw.get("stage"), kw.get("message"))
    if key != last[0]:
        last[0] = key
        print(f"  {kw.get('stage')}: {kw.get('message', '')}", flush=True)


res = runtime.install_runtime(variant, progress, lambda: False, lambda line: None, base=dest,
                              pk=sys.argv[3] if len(sys.argv) > 3 else None)
print("bundled", res)
