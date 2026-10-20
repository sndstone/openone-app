"""OpenOne desktop app.

    OpenOne                 native window (pywebview: WebView2 on Windows, WKWebView on macOS)
    OpenOne --browser       open the UI in the default browser instead
    OpenOne --serve [PORT]  headless: print the UI URL (for testing)
"""
import argparse
import sys
import threading
import time
import webbrowser

from .controller import Controller
from .server import serve_in_thread

FILE_TYPES = {
    "zip": ("Model package (*.zip)", "All files (*.*)"),
    "jsonl": ("JSON lines (*.jsonl;*.json;*.txt)", "All files (*.*)"),
    "python": ("Python (python.exe;python3;python)", "All files (*.*)"),
}


def autostart(c):
    s = c.store.settings
    if s.get("onboarded") and s["backend"].get("autostart", True) and (s.get("active_model") or s["backend"]["kind"] == "remote"):
        try:
            c.start_backend()
        except Exception as e:                                    # noqa: BLE001 - shown in the UI status
            c.store.log(f"autostart failed: {e}", "error")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="OpenOne")
    ap.add_argument("--browser", action="store_true")
    ap.add_argument("--serve", nargs="?", const=0, type=int, metavar="PORT")
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    c = Controller()
    ui = {}
    srv, token = serve_in_thread(c, a.serve if a.serve is not None else a.port, ui=ui)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={token}"
    threading.Thread(target=autostart, args=(c,), daemon=True).start()
    if a.serve is not None or a.browser:
        if sys.stdout is not None:                       # a windowed (frozen) build has no console
            print(f"OpenOne UI: {url}", flush=True)
        (c.store.dir / "serve-url.txt").write_text(url + "\n", encoding="utf-8")
        if a.browser:
            webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    else:
        import webview
        webview.settings["ALLOW_DOWNLOADS"] = True
        win = webview.create_window("OpenOne", url, width=1320, height=900, min_size=(980, 660), background_color="#0f1115")

        def dialog(kind, filt, filename):
            types = FILE_TYPES.get(filt or "", ("All files (*.*)",))
            if kind == "folder":
                r = win.create_file_dialog(webview.FileDialog.FOLDER)
            elif kind == "save":
                r = win.create_file_dialog(webview.FileDialog.SAVE, save_filename=filename or "", file_types=types)
            else:
                r = win.create_file_dialog(webview.FileDialog.OPEN, file_types=types)
            if not r:
                return []
            return [r] if isinstance(r, str) else list(r)
        ui["dialog"] = dialog
        webview.start(private_mode=False, storage_path=str(c.store.dir / "webview"))
    c.stop_backend()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
