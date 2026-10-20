"""Record files for bucket batches: parse CSV/TSV, JSON, JSON lines, XML or plain text into a list of records, and write
classified records back out as CSV, JSON, JSON lines, XML or a zip with one CSV per bucket.

A record is a dict. CSV values are strings; JSON keeps its types; XML elements become dicts (attributes as "@name",
repeated children as lists, a leaf element as its text, mixed content's full text under "text")."""
import csv
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

from openone_core.pipeline import parse_json_text

FORMATS = ("csv", "tsv", "json", "jsonl", "xml", "txt")
EXT = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".json": "json", ".jsonl": "jsonl", ".ndjson": "jsonl",
       ".xml": "xml", ".txt": "txt"}
WHOLE = "*"                        # text_fields value: classify the whole record
RESULT_KEY = "_openone"            # JSON/JSONL exports put the classification under this key
DELIMS = ",;\t|"
BOM = "﻿"


class RecordError(ValueError):
    pass


class _Limit:
    def __init__(self, limit):
        self.limit, self.n = limit, 0

    def add(self):
        self.n += 1
        if self.n > self.limit:
            raise RecordError(f"more than {self.limit:,} records; split the file")


# ------------------------------------------------------------------------------------------------ parsing
def detect(text, filename=None):
    if filename:
        m = re.search(r"\.[A-Za-z]+$", filename)
        if m and m.group(0).lower() in EXT:
            return EXT[m.group(0).lower()]
    s = text.lstrip()
    if s.startswith("<"):
        return "xml"
    if s.startswith("["):
        return "json"
    if s.startswith("{"):
        first = s.split("\n", 1)[0].strip()
        val, err = parse_json_text(first)
        return "jsonl" if not err and isinstance(val, dict) and "\n" in s.strip() else "json"
    first = s[:20000].split("\n", 1)[0]
    if "\t" in first:
        return "tsv"
    if re.search(r"[,;|]", first):
        return "csv"
    return "txt"


def parse(text, fmt=None, filename=None, record_tag=None, limit=50000):
    """-> {"format", "records", "fields", "record_tag", "warnings"}; raises RecordError with a line/position on bad input.
    The record limit is enforced while parsing, so an oversized file fails before it is fully expanded."""
    text = text.lstrip(BOM)
    if not text.strip():
        raise RecordError("the file is empty")
    fmt = fmt or detect(text, filename)
    if fmt not in FORMATS:
        raise RecordError(f"unknown format {fmt!r} (use one of {', '.join(FORMATS)})")
    warnings, tag, lim, cands = [], None, _Limit(limit), []
    if fmt in ("csv", "tsv"):
        records = _csv(text, fmt, warnings, lim)
    elif fmt == "json":
        records = _json(text, lim, warnings)
    elif fmt == "jsonl":
        records = _jsonl(text, lim)
    elif fmt == "xml":
        records, tag, cands = _xml(text, record_tag, lim, warnings)
    else:
        records = []
        for ln in text.splitlines():
            if ln.strip():
                lim.add()
                records.append({"text": ln.strip()})
    if not records:
        raise RecordError("no records found")
    flds = fields(records)
    prev = result_fields(records, flds)
    if prev:
        warnings.append("earlier OpenOne results found (" + ", ".join(prev[:6]) + (", …" if len(prev) > 6 else "") +
                        "); they are kept in exports but never shown to the model")
    return {"format": fmt, "records": records, "fields": flds, "record_tag": tag, "warnings": warnings,
            "result_fields": prev, "record_candidates": [{k: c[k] for k in ("path", "tag", "count")} for c in cands]}


_STATUSES = {"ok", "empty", "invalid_request", "invalid_response", "backend_error"}


def _is_p(v):
    if v in (None, ""):
        return True
    try:
        return 0.0 <= float(v) <= 1.0
    except (TypeError, ValueError):
        return False


def _is_result_obj(v):
    return (isinstance(v, dict) and {"bucket", "p_bucket", "accepted", "status", "probabilities"} <= set(v)
            and v["status"] in _STATUSES and isinstance(v["accepted"], (bool, type(None))) and _is_p(v["p_bucket"])
            and isinstance(v["probabilities"], dict) and all(_is_p(x) for x in v["probabilities"].values()))


def result_fields(records, flds):
    """Fields that hold an earlier OpenOne classification, so whole-record classification can leave them out. They are
    recognized by their full signature, never by name alone:
    - JSON/JSONL: an object value with bucket/p_bucket/accepted/status/probabilities and valid values;
    - CSV: every block a CSV export appended: bucket, p_bucket, accepted, status (each possibly suffixed "_openone" or
      "_N"), followed by at least two p_<bucket> columns, with probabilities in [0, 1], accepted true/false and a
      known status in every sampled row."""
    sample, out = records[:500], []
    for k in flds:
        vals = [r[k] for r in sample if k in r]
        if vals and all(_is_result_obj(v) for v in vals):
            out.append(k)
    base = [re.sub(r"(_openone)*(_\d+)?$", "", f) for f in flds]
    i = 0
    while i < len(flds) - 5:
        if base[i:i + 4] != ["bucket", "p_bucket", "accepted", "status"]:
            i += 1
            continue
        j = i + 4
        while j < len(flds) and base[j].startswith("p_"):
            j += 1
        probs = flds[i + 4:j]
        while j < len(flds) and base[j] in ("expected", "correct", "error"):
            j += 1
        block = flds[i:j]
        ok = len(probs) >= 2 and all(
            _is_p(r.get(flds[i + 1])) and str(r.get(flds[i + 2], "")).lower() in ("true", "false", "")
            and r.get(flds[i + 3]) in _STATUSES and all(_is_p(r.get(c)) for c in probs) for r in sample)
        if ok:
            out += [f for f in block if f not in out]
            i = j
        else:
            i += 1
    return out


def fields(records):
    """Every top-level key, in order of first appearance across all records."""
    out = {}
    for r in records:
        for k in r:
            out.setdefault(k, None)
    return list(out)


def _unique(name, taken):
    n, out = 1, name
    while out in taken:
        n += 1
        out = f"{name}_{n}"
    taken.add(out)
    return out


def _sniff_delimiter(text):
    try:
        return csv.Sniffer().sniff(text[:20000], delimiters=DELIMS).delimiter
    except csv.Error:
        # the sniffer gives up on ragged files: take the delimiter the header line uses most (outside quotes)
        head = re.sub(r'"[^"]*"', "", text.split("\n", 1)[0])
        counts = {d: head.count(d) for d in DELIMS}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","


def _csv(text, fmt, warnings, lim):
    delim = "\t" if fmt == "tsv" else _sniff_delimiter(text)
    rows = csv.reader(io.StringIO(text, newline=""), delimiter=delim)
    try:
        header = next(rows)
    except StopIteration:
        raise RecordError("the file is empty") from None
    except csv.Error as e:
        raise RecordError(f"CSV line 1: {e}") from None
    names, taken = [], set()
    for i, h in enumerate(header):
        names.append(_unique(h.strip() or f"column_{i + 1}", taken))
    out, ragged, extras = [], 0, []
    try:
        for row in rows:
            if not any(c.strip() for c in row):
                continue
            lim.add()
            if len(row) != len(names):
                ragged += 1
            rec = {n: (row[i] if i < len(row) else "") for i, n in enumerate(names)}
            for j, extra in enumerate(row[len(names):]):
                while len(extras) <= j:                  # extra columns get names that collide with nothing
                    extras.append(_unique(f"extra_{len(extras) + 1}", taken))
                rec[extras[j]] = extra
            out.append(rec)
    except csv.Error as e:
        raise RecordError(f"CSV line {rows.line_num}: {e}") from None
    if ragged:
        warnings.append(f"{ragged} row(s) have a different number of columns than the header "
                        f"(delimiter {'tab' if delim == chr(9) else repr(delim)})")
    return out


def _as_record(v):
    return v if isinstance(v, dict) else {"text": v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}


def _json(text, lim, warnings):
    val, err = parse_json_text(text)
    if err:
        raise RecordError(f"JSON line {err.get('line', '?')}, column {err.get('column', '?')}: {err['message']}")
    if isinstance(val, dict):
        if isinstance(val.get("records"), list) and isinstance(val.get("buckets"), list) and "meta" in val:
            val = val["records"]                         # this app's own JSON export: re-import its records
            warnings.append("OpenOne export: re-importing its records (the previous results are in the "
                            f"{RESULT_KEY!r} field)")
        elif len(val) == 1 and isinstance(next(iter(val.values())), list):
            val = next(iter(val.values()))               # a wrapper such as {"items": [...]}
        else:
            lim.add()
            return [val]                                 # one object is one record
    if not isinstance(val, list):
        raise RecordError("JSON must be an array of records (objects or strings), an object holding one array, "
                          "or one object")
    out = []
    for v in val:
        lim.add()
        out.append(_as_record(v))
    return out


def _jsonl(text, lim):
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        lim.add()
        val, err = parse_json_text(line)
        if err:
            raise RecordError(f"JSON lines: line {n}, column {err.get('column', '?')}: {err['message']}")
        out.append(_as_record(val))
    return out


_DECL = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.I)


def _xml(text, record_tag, lim, warnings):
    # no DTDs at all (checked over the whole text, not a prefix): rules out entity expansion and external entities
    if _DECL.search(text):
        raise RecordError("XML with a DOCTYPE or ENTITY declaration is not accepted")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        line, col = e.position
        raise RecordError(f"XML line {line}, column {col + 1}: {e}") from None
    cands = _xml_candidates(root)
    if record_tag:
        elems = _xml_select(root, record_tag)
        if not elems:
            raise RecordError(f"no <{record_tag}> elements in the file")
    elif _local(root.tag) == "openone_buckets":
        record_tag = "openone_buckets/record/data"
        elems = _xml_select(root, record_tag)
        warnings.append("OpenOne XML export: re-importing the original <data> of each record")
    else:
        # best guess; the UI lists every candidate so the user can switch
        complex_ = [c for c in cands if c["complex"]]
        if complex_:
            record_tag = complex_[0]["path"]             # repeated elements that have fields of their own
        elif len(root) == 1:
            record_tag = f"{_local(root.tag)}/{_local(root[0].tag)}"   # <items><item id="7">…</item></items>
        elif len(root) == 0 or len({_local(k.tag) for k in root}) == len(root) or not cands:
            record_tag = _local(root.tag)                # all different fields: the document is one record
        else:
            record_tag = cands[0]["path"]                # a plain list of text elements
        elems = _xml_select(root, record_tag)
        if len(cands) > 1:
            warnings.append(f"{len(cands)} possible record elements; using {record_tag} (pick another under "
                            "\"XML record element\")")
    recs = []
    for e in elems:
        lim.add()
        v = _xml_value(e)
        recs.append(v if isinstance(v, dict) else {"text": v})
    return recs, record_tag, cands


def _xml_candidates(root):
    """Repeated sibling elements that could be records: [{"path", "tag", "count", "complex"}], best first (elements
    with fields of their own before plain text elements, then more of them, then shallower)."""
    out, stack = {}, [(root, _local(root.tag), 0)]
    while stack:
        node, path, depth = stack.pop()
        counts = Counter(_local(k.tag) for k in node)
        for k in node:
            kp = f"{path}/{_local(k.tag)}"
            if counts[_local(k.tag)] >= 2:
                c = out.setdefault(kp, {"path": kp, "tag": _local(k.tag), "count": 0, "complex": False, "depth": depth + 1})
                c["count"] += 1
                c["complex"] = c["complex"] or bool(len(k) or k.attrib)
            if len(k):
                stack.append((k, kp, depth + 1))
    return sorted(out.values(), key=lambda c: (not c["complex"], -c["count"], c["depth"], c["path"]))


def _xml_select(root, spec):
    """spec: a path from the root ("rss/channel/item") or a bare element name (any depth, root included)."""
    if "/" not in spec:
        return [e for e in root.iter() if _local(e.tag) == spec]
    parts = spec.split("/")
    if parts[0] != _local(root.tag):
        return []
    level = [root]
    for name in parts[1:]:
        level = [k for e in level for k in e if _local(k.tag) == name]
    return level


def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else str(tag)


def _xml_value(e):
    kids = list(e)
    if not kids and not e.attrib:
        return (e.text or "").strip()
    out = {f"@{_local(k)}": v for k, v in e.attrib.items()}
    if not kids:
        if (e.text or "").strip():
            out["text"] = e.text.strip()
        return out
    mixed = bool((e.text or "").strip()) or any((k.tail or "").strip() for k in kids)
    for k in kids:
        name, v = _local(k.tag), _xml_value(k)
        if name in out:
            out[name] = out[name] if isinstance(out[name], list) else [out[name]]
            out[name].append(v)
        else:
            out[name] = v
    if mixed:                                            # <body>Please <b>refund</b> me</body>: keep every word
        key = "text" if "text" not in out else "#text"
        out[key] = re.sub(r"\s+", " ", "".join(e.itertext())).strip()
    return out


# ------------------------------------------------------------------------------------------------ state for the model
def state_for(rec, text_fields, exclude=()):
    """The state the model reads: one text field -> {"text": value}; several fields or the whole record -> an object.
    exclude: fields holding earlier OpenOne results (result_fields()); the whole record never includes them."""
    if not text_fields or WHOLE in text_fields:
        st = {k: v for k, v in rec.items() if k not in exclude}
    elif len(text_fields) == 1:
        v = rec.get(text_fields[0])
        st = {"text": v} if isinstance(v, str) else {text_fields[0]: v}
    else:
        st = {k: rec.get(k) for k in text_fields}
    if not any(_filled(v) for v in st.values()):
        return None
    return st


def _filled(v):
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict)):
        return bool(v)
    return True


def cell(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


# ------------------------------------------------------------------------------------------------ output
def sort_rows(rows, buckets, how="bucket"):
    """rows: [{"index", "record", "bucket", "p", "accepted", ...}]. how: bucket (bucket order, accepted first, then
    confidence), confidence (highest first), original."""
    if how == "original":
        return sorted(rows, key=lambda r: r["index"])
    if how == "confidence":
        return sorted(rows, key=lambda r: (-(r.get("p") or 0), r["index"]))
    order = {b: i for i, b in enumerate(buckets)}
    return sorted(rows, key=lambda r: (order.get(r.get("bucket"), len(order)), not r.get("accepted"),
                                       -(r.get("p") or 0), r["index"]))


def out_columns(rows, buckets, rec_fields):
    """[(column name, result key)]: the classification columns, renamed with an "_openone" suffix where an original
    column already has the name (the original data is never overwritten)."""
    keys = ["bucket", "p", "accepted", "status"] + [("p", b) for b in buckets]
    if any("expected" in r for r in rows):
        keys += ["expected", "correct"]
    if any(r.get("error") for r in rows):
        keys.append("error")
    taken, out = set(rec_fields), []
    for k in keys:
        base = f"p_{k[1]}" if isinstance(k, tuple) else ("p_bucket" if k == "p" else k)
        name = base if base not in taken else _unique(f"{base}_openone", taken)
        taken.add(name)
        out.append((name, k))
    return out


def _result(r, k):
    if isinstance(k, tuple):
        return (r.get("probabilities") or {}).get(k[1])
    return r.get(k)


def _result_obj(r, buckets):
    out = {"bucket": r.get("bucket"), "p_bucket": r.get("p"), "accepted": r.get("accepted"), "status": r["status"],
           "probabilities": {b: (r.get("probabilities") or {}).get(b) for b in buckets}}
    if "expected" in r:
        out |= {"expected": r["expected"], "correct": r.get("correct")}
    if r.get("error"):
        out["error"] = r["error"]
    return out


_FORMULA = re.compile(r"^[=@\t\r]|^[+-](?![0-9.]+$)")


def csv_cell(v):
    """cell() plus a leading apostrophe on text a spreadsheet would run as a formula (=, @, +/- not followed by a
    number): classified text is often untrusted."""
    s = cell(v)
    return "'" + s if isinstance(v, str) and _FORMULA.match(s) else s


def to_csv(rows, buckets, rec_fields):
    """UTF-8 with a BOM so spreadsheet apps pick the encoding; original columns first, then the classification."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    extra = out_columns(rows, buckets, rec_fields)
    w.writerow([csv_cell(c) for c in rec_fields] + [csv_cell(n) for n, _ in extra])
    for r in rows:
        w.writerow([csv_cell(r["record"].get(c)) for c in rec_fields] + [csv_cell(_result(r, k)) for _, k in extra])
    return BOM + buf.getvalue()


def _result_key(rec):
    return RESULT_KEY if RESULT_KEY not in rec else _unique(RESULT_KEY, set(rec))


def to_json(rows, buckets, meta):
    """{"meta": {...}, "buckets": [...], "records": [{...original, "_openone": {...}}]}"""
    return json.dumps({"meta": meta, "buckets": buckets,
                       "records": [dict(r["record"]) | {_result_key(r["record"]): _result_obj(r, buckets)} for r in rows]},
                      ensure_ascii=False, indent=2) + "\n"


def to_jsonl(rows, buckets):
    return "".join(json.dumps(dict(r["record"]) | {_result_key(r["record"]): _result_obj(r, buckets)},
                              ensure_ascii=False) + "\n" for r in rows)


_XML_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_XML_BAD = re.compile("[^\t\n\r\x20-퟿-�\U00010000-\U0010ffff]")


def _xs(v):
    """Text XML 1.0 can carry: characters it forbids (NUL, most control characters) become U+FFFD."""
    return _XML_BAD.sub("�", cell(v))


def _xml_put(parent, key, v):
    """Original data back to XML: "@name" keys become attributes, dicts nest, lists repeat, other names that are not
    valid XML element names go into <field name="...">."""
    if key.startswith("@") and _XML_NAME.match(key[1:]) and not isinstance(v, (dict, list)):
        parent.set(key[1:], _xs(v))
        return
    if isinstance(v, list):
        for x in v:
            _xml_put(parent, key, x)
        return
    ok = _XML_NAME.match(key) and not key.lower().startswith("xml")
    e = ET.SubElement(parent, key if ok else "field")
    if not ok:
        e.set("name", _xs(key))
    if isinstance(v, dict):
        for k, x in v.items():
            _xml_put(e, str(k), x)
    else:
        e.text = _xs(v)


def to_xml(rows, buckets, meta):
    """<openone_buckets ...><record bucket= p_bucket= accepted= status=><data>original</data><probabilities/></record>"""
    root = ET.Element("openone_buckets", {k: _xs(v) for k, v in meta.items() if not isinstance(v, (dict, list))})
    for r in rows:
        o = _result_obj(r, buckets)
        attrs = {k: _xs(o[k]) for k in ("bucket", "p_bucket", "accepted", "status", "expected", "correct", "error") if k in o}
        e = ET.SubElement(root, "record", attrs)
        data = ET.SubElement(e, "data")
        for k, v in r["record"].items():
            _xml_put(data, str(k), v)
        p = ET.SubElement(e, "probabilities")
        for b in buckets:
            ET.SubElement(p, "p", {"bucket": _xs(b)}).text = _xs(o["probabilities"][b])
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def to_zip(rows, buckets, rec_fields):
    """One CSV per bucket (accepted rows), plus below_threshold.csv and errors.csv. Groups are keyed apart from their
    file names, so a bucket called "errors" stays its own file."""
    groups = {}
    for r in rows:
        key = ("bucket", r["bucket"]) if r.get("accepted") else ("below",) if r["status"] == "ok" else ("errors",)
        groups.setdefault(key, []).append(r)
    order = [("bucket", b) for b in buckets] + [("below",), ("errors",)]
    buf, used = io.BytesIO(), set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key in order:
            if key not in groups:
                continue
            base = _safe_name(key[1]) if key[0] == "bucket" else {"below": "below_threshold", "errors": "errors"}[key[0]]
            fn, n = base, 1
            while fn.casefold() in used:
                n += 1
                fn = f"{base}_{n}"
            used.add(fn.casefold())
            z.writestr(f"{fn}.csv", to_csv(groups[key], buckets, rec_fields).encode("utf-8"))
    return buf.getvalue()


def _safe_name(s):
    s = re.sub(r"[^\w.-]+", "_", str(s)).strip("._") or "bucket"
    return s[:80]
