"""OpenOne core: package handling, prompt building, scoring backends and the decision pipeline.

Pure Python except `scorers` (torch / vLLM, imported lazily). Shared by the desktop app, its inference worker and the
remote gate service, so every backend builds the same prompt and every response is built by the same code.
"""
__version__ = "0.1.0"
SCHEMA_VERSION = "1"
