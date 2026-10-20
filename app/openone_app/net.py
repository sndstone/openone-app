"""HTTPS for the app's own downloads (inference runtime, model packages).

A frozen Python on macOS has no CA bundle of its own, so certificates are checked against the operating system's trust
store (truststore: Keychain on macOS, the certificate store on Windows; this also honours corporate TLS-inspection
roots). certifi's bundle is the fallback. Only https is allowed, also after redirects."""
import ssl
import urllib.error
import urllib.request

USER_AGENT = "OpenOne-app"


def ssl_context():
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:                                     # noqa: BLE001 - not installed, or no OS store support
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                     # noqa: BLE001
        return ssl.create_default_context()


class _HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError(f"refusing a redirect to a non-https URL: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = None


def opener():
    global _opener
    if _opener is None:
        _opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context()), _HttpsOnly())
    return _opener


def open_url(url, headers=None, timeout=60):
    if not url.lower().startswith("https://"):
        raise ValueError(f"only https downloads are allowed: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return opener().open(req, timeout=timeout)
