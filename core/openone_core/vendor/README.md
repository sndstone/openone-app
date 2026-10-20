# Vendored Decider prompt code

`decider/prompt.py` and `decider/systemone.py` are verbatim copies from the OpenOne model package (`decider/`, identical
in the bf16 and FP8 v9 r16 bundles), which ships them from `decider-ai` (Apache-2.0, see `decider/LICENSE`).

| File | sha256 |
| --- | --- |
| `prompt.py` | `46f8158cd00c708408f6340340517c5a0c50fb0a339d0e46cd0a30f8801624cd` |
| `systemone.py` | `884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34` |

The app builds every prompt with these copies and never imports Python code from an imported model package. At import
it compares the package's own copies with these hashes and shows a warning when they differ (a future package with a
different prompt format needs an app update).
