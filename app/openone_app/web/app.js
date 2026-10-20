"use strict";
/* OpenOne desktop UI. Vanilla JS, no build step. All model/user text is inserted as text nodes, never as HTML. */

// ------------------------------------------------------------------------------------------------ basics
const qs = new URLSearchParams(location.search);
const TOKEN = qs.get("t") || sessionStorage.getItem("openone-token") || "";
sessionStorage.setItem("openone-token", TOKEN);
if (qs.get("t")) history.replaceState(null, "", "/" + (location.hash || ""));

const $ = (id) => document.getElementById(id);
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "style") e.setAttribute("style", v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}
const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
const fmtP = (p) => (p * 100).toFixed(p >= 0.9995 || p < 0.0005 ? 2 : 1) + "%";
const fmtBytes = (b) => b >= 1e9 ? (b / 1e9).toFixed(2) + " GB" : b >= 1e6 ? (b / 1e6).toFixed(1) + " MB" : (b / 1e3).toFixed(0) + " KB";
const short = (s, n = 12) => (s || "").slice(0, n);

async function api(path, body, opts = {}) {
  const r = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-OpenOne-Token": TOKEN },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data;
  try { data = await r.json(); } catch { data = { error: `HTTP ${r.status}` }; }
  if (!r.ok) { const e = new Error(data.error || `HTTP ${r.status}`); e.status = r.status; e.data = data; throw e; }
  return data;
}

function toast(msg, isErr) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (isErr ? " err" : "");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), isErr ? 7000 : 3500);
}

const S = { status: null, presets: null, adv: localStorage.getItem("openone-adv") === "1", jobsSeen: {} };

// ------------------------------------------------------------------------------------------------ status / chrome
async function refreshStatus() {
  try { S.status = await api("/api/status"); } catch (e) { setBanner("Cannot reach the app backend: " + e.message, true); return; }
  document.body.classList.toggle("native", !!S.status.native);
  renderChrome();
  if (!S.status.onboarded && !S.onboardShown) { S.onboardShown = true; openOnboarding(); }
}
function pollLoop() {
  const busy = S.status && (Object.keys(S.status.jobs || {}).length || (S.status.backend || {}).state === "starting");
  setTimeout(async () => { await refreshStatus(); pollLoop(); }, busy ? 1200 : 6000);
}
function setBanner(text, isErr) {
  const b = $("banner");
  if (!text) { b.classList.add("hidden"); return; }
  b.textContent = text; b.className = "banner" + (isErr ? " error" : "");
}
function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  localStorage.setItem("openone-theme", t || "system");
}
applyTheme(localStorage.getItem("openone-theme"));
function renderChrome() {
  const st = S.status, b = st.backend || {};
  if (st.theme !== undefined && st.theme !== (document.documentElement.dataset.theme || "system")) applyTheme(st.theme);
  const pill = $("backendPill");
  pill.className = "pill " + (b.state || "stopped");
  pill.textContent = b.state === "ready" ? `ready · ${b.kind === "remote" ? "remote" : b.device}` :
    b.state === "starting" ? "starting…" : b.state === "error" ? "error" : "stopped";
  pill.title = b.error || (b.runtime || "");
  const m = b.model || (st.models || []).find((x) => x.name === st.active_model);
  const badge = $("modelBadge");
  if (m) {
    const fb = (b.mode || m.local_mode) === "fp8_dequantized_bf16";
    badge.textContent = m.name + (fb ? " · FP8→bf16 fallback" : ` · ${m.format}`);
    badge.className = "badge" + (fb ? " warn" : "");
  } else { badge.textContent = "no model"; badge.className = "badge muted"; }
  const warns = [];
  if (b.state === "error") warns.push("Model backend error: " + b.error);
  else if (b.state === "ready" && b.mode === "fp8_dequantized_bf16") warns.push(st.fallback_warning);
  if (st.settings && st.settings.provisional) warns.push("Thresholds are provisional for this model variant (fitted values pending).");
  if (st.onboarded && !b.state || (st.onboarded && b.state === "stopped")) warns.push("The model is not running. Start it in Settings.");
  setBanner(warns.join("  ·  "), b.state === "error");
  if (document.querySelector("#tab-settings.active")) renderSettings();
  if (document.querySelector("#tab-tuning.active") && !S.tuningDirty) renderTuning();
}

function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === "tab-" + name));
  location.hash = name;
  if (name === "settings") renderSettings();
  if (name === "tuning") renderTuning();
  if (name === "logs") { loadLog(); loadFeedback(); }
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
function setAdv(on) {
  S.adv = on; localStorage.setItem("openone-adv", on ? "1" : "0");
  document.body.classList.toggle("advanced", on); $("advToggle").checked = on;
  const cur = document.querySelector("#tabs button.active");
  if (!on && cur && cur.classList.contains("adv")) showTab("gate");
}
$("advToggle").addEventListener("change", (e) => setAdv(e.target.checked));

// ------------------------------------------------------------------------------------------------ JSON helpers
function tree(v, key, depth = 0) {
  const keyNode = key !== undefined ? [el("span", { class: "k", text: JSON.stringify(key) }), ": "] : [];
  if (v !== null && typeof v === "object") {
    const isArr = Array.isArray(v), entries = isArr ? v.map((x, i) => [i, x]) : Object.entries(v);
    const d = el("details", depth < 3 ? { open: true } : {});
    d.append(el("summary", {}, ...keyNode, el("span", { class: "z", text: isArr ? `[${entries.length}]` : `{${entries.length}}` })));
    for (const [k, x] of entries) d.append(tree(x, isArr ? undefined : k, depth + 1));
    return d;
  }
  const cls = typeof v === "string" ? "s" : typeof v === "number" ? "n" : typeof v === "boolean" ? "b" : "z";
  return el("div", { class: "leaf" }, ...keyNode, el("span", { class: cls, text: JSON.stringify(v) }));
}

// Position-tracking JSON scan: JSON pointer -> {line, col} of each value (input must be valid JSON).
function jsonLocate(text) {
  const map = {}; let i = 0;
  const pos = (k) => { let line = 1, col = 1; for (let j = 0; j < k; j++) { if (text[j] === "\n") { line++; col = 1; } else col++; } return { line, col }; };
  const ws = () => { while (i < text.length && " \t\r\n".includes(text[i])) i++; };
  const str = () => { let j = i + 1; while (j < text.length && text[j] !== '"') j += text[j] === "\\" ? 2 : 1; const s = JSON.parse(text.slice(i, j + 1)); i = j + 1; return s; };
  const esc = (k) => String(k).replace(/~/g, "~0").replace(/\//g, "~1");
  function val(ptr) {
    ws(); map[ptr] = i;
    const c = text[i];
    if (c === "{") { i++; ws(); if (text[i] === "}") { i++; return; }
      for (;;) { ws(); const kpos = i; const k = str(); map[ptr + "/" + esc(k) + "#key"] = kpos; ws(); i++; val(ptr + "/" + esc(k)); ws(); if (text[i] === ",") { i++; continue; } i++; return; } }
    if (c === "[") { i++; ws(); if (text[i] === "]") { i++; return; } let n = 0;
      for (;;) { val(ptr + "/" + n++); ws(); if (text[i] === ",") { i++; continue; } i++; return; } }
    if (c === '"') { str(); return; }
    while (i < text.length && !",}] \t\r\n".includes(text[i])) i++;
  }
  try { val(""); } catch { return {}; }
  const out = {}; for (const [k, v] of Object.entries(map)) out[k] = pos(v); return out;
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch { const t = el("textarea", {}); t.value = text; document.body.append(t); t.select(); document.execCommand("copy"); t.remove(); toast("Copied"); }
}
function download(name, content, type = "application/json") {
  const a = el("a", { href: URL.createObjectURL(new Blob([content], { type })), download: name }); document.body.append(a); a.click(); a.remove();
}
async function pickPath(kind, filter, filename) {
  if (!S.status || !S.status.native) return null;
  try { const r = await api("/api/dialog", { kind, filter, filename }); return (r.paths || [])[0] || null; }
  catch (e) { toast(e.message, true); return null; }
}

// ------------------------------------------------------------------------------------------------ result rendering
const ACTION_LABEL = { auto_run: "AUTO-RUN", ask: "ASK", alarm: "ALARM" };
function vbadge(label, v) {
  if (!v) return el("span", { class: "vbadge", text: `${label} –` });
  return el("span", { class: "vbadge " + (v.valid ? "pass" : "fail"), title: v.valid ? "schema and consistency checks pass" : v.errors.map((e) => `${e.path || "/"}: ${e.message}`).join("\n"),
    text: `${label} ${v.valid ? "✓ valid" : "✗ invalid"}` });
}

function bars(answer) {
  const p = answer.probabilities, names = Object.keys(p);
  const order = answer.use_case === "toolcall" ? ["allow", "ask", "block"] : names.slice().sort((a, b) => p[b] - p[a]);
  return el("div", { class: "bars" }, order.map((n) => {
    const track = el("div", { class: "track" }, el("div", { class: "fill " + (answer.use_case === "toolcall" ? n : ""), style: `width:${(p[n] * 100).toFixed(2)}%` }));
    let tick = null;
    if (answer.use_case === "toolcall" && n === "allow") tick = answer.verdict.auto_run_at;
    if (answer.use_case === "toolcall" && n === "block") tick = answer.verdict.alarm_at;
    if (answer.use_case === "bucket" && n === answer.choice) tick = answer.accept_at;
    if (tick !== null) track.append(el("div", { class: "tick", style: `left:${(tick * 100).toFixed(2)}%`, title: `threshold ${tick}` }));
    return el("div", { class: "bar" + (n === answer.choice ? " top" : "") }, el("div", { class: "name", title: n, text: n }), track, el("div", { class: "val", text: fmtP(p[n]) }));
  }));
}

function answerCard(qid, a, multi) {
  let cls, big, why;
  if (a.use_case === "toolcall") { cls = "act-" + a.verdict.action; big = ACTION_LABEL[a.verdict.action]; why = a.verdict.rule; }
  else { cls = a.accepted ? "act-accepted" : "act-below"; big = a.choice; why = (a.accepted ? "p ≥ " : "below the acceptance threshold ") + a.accept_at + ` (p = ${fmtP(a.x_p_max)})`; }
  return el("div", { class: "list" },
    el("div", { class: "verdict " + cls }, el("div", { class: "big", text: big }),
      el("div", {}, el("div", { text: (multi ? qid + " · " : "") + (a.use_case === "toolcall" ? `model choice: ${a.choice}` : (a.accepted ? "accepted" : "not accepted: route to a human or a fallback")) }),
        el("div", { class: "why", text: why }))),
    bars(a));
}

function rawView(raw, out) {
  const box = el("div", { class: "list" });
  for (const [qid, v] of Object.entries(raw.questions)) {
    const names = Object.keys(v.letter_logprobs);
    box.append(el("h3", { text: `${qid} · ${v.use_case}` }),
      el("table", { class: "kv" }, el("tr", {}, ["option", "letter", "log-prob (T=1)", "p (T=1)", `p (T=${v.temperature})`].map((h) => el("th", { text: h }))),
        names.map((n) => el("tr", {}, el("td", { text: n }), el("td", { text: v.option_letters[n] }), el("td", { class: "num", text: v.letter_logprobs[n].toFixed(4) }),
          el("td", { class: "num", text: v.probabilities_T1[n].toFixed(4) }), el("td", { class: "num", text: v.probabilities_T[n].toFixed(4) })))),
      el("div", { class: "meta" }, el("span", {}, "temperature ", el("b", { text: v.temperature })),
        ...Object.entries(v.thresholds).map(([k, x]) => el("span", {}, k + " ", el("b", { text: x }))),
        el("span", {}, "prompt ", el("b", { text: v.prompt_tokens + " tokens" })), el("span", {}, "state ", el("b", { text: `${v.state_tokens} / ${v.max_state_tokens}` })),
        v.elided ? el("span", { class: "warn", text: "elided" }) : null, v.truncated ? el("span", { class: "bad", text: "truncated" }) : null,
        v.forward_ms !== null && v.forward_ms !== undefined ? el("span", {}, "forward ", el("b", { text: v.forward_ms + " ms" })) : null),
      el("div", { class: "why", text: "rule: " + v.rule }));
    if (v.prompt_text) {
      const t = v.prompt_text, cut = t.lastIndexOf("(");
      box.append(el("details", {}, el("summary", { text: `Exact prompt text (${v.prompt_tokens} tokens; the answer is read at the highlighted final token)` }),
        el("pre", { class: "prompt mono" }, t.slice(0, cut), el("span", { class: "slot", text: t.slice(cut) }))));
    }
  }
  box.append(el("details", {}, el("summary", { text: "Full raw output (validation + raw)" }), el("pre", { class: "mono", text: JSON.stringify(out, null, 2) })));
  return box;
}

function jsonPanel(out, req) {
  const resp = out.response;
  const panes = { parsed: () => tree(resp), text: () => el("pre", { text: JSON.stringify(resp, null, 2) }) };
  if (out.raw) panes.raw = () => rawView(out.raw, out);
  const body = el("div", { class: "jbody" });
  const btns = {};
  const show = (k) => { clear(body).append(panes[k]()); for (const [n, b] of Object.entries(btns)) b.classList.toggle("active", n === k); };
  const labels = { parsed: "Parsed", text: "Raw JSON", raw: "Raw mode" };
  const tabs = el("div", { class: "jtabs" }, Object.keys(panes).map((k) => (btns[k] = el("button", { text: labels[k], onclick: () => show(k) }))),
    el("span", { class: "sp" }), vbadge("Request", out.request_validation), vbadge("Response", out.response_validation),
    el("button", { text: "Copy", title: "Copy the response JSON", onclick: () => copyText(JSON.stringify(resp, null, 2)) }),
    el("button", { text: "Report", title: "Send this case to the feedback log", onclick: () => reportCase(req, out) }));
  show(S.adv && out.raw ? "raw" : "parsed");
  return el("div", { class: "jsonpanel" }, tabs, body);
}

async function reportCase(req, out) {
  const note = prompt("What looks wrong? (stored locally in the feedback log)");
  if (note === null) return;
  await api("/api/feedback/report", { request: req, response: out.response, validation: out.response_validation, note });
  toast("Added to the feedback log"); refreshStatus();
}

function renderOutcome(box, out, req) {
  clear(box);
  const rv = out.request_validation;
  if (!rv.valid) {
    box.append(el("div", { class: "verdict act-alarm" }, el("div", { class: "big", text: "INVALID" }), el("div", { class: "why", text: "The request does not match the API schema; nothing was scored." })),
      el("ul", { class: "errors" }, rv.errors.map((e) => el("li", {}, el("code", { text: (e.path || "/") + (e.line ? ` (line ${e.line}:${e.column})` : "") }), " ", e.message))));
    return;
  }
  const resp = out.response, ids = Object.keys(resp.answers);
  for (const qid of ids) box.append(answerCard(qid, resp.answers[qid], ids.length > 1));
  const a0 = resp.answers[ids[0]];
  box.append(el("div", { class: "meta" },
    el("span", {}, "latency ", el("b", { text: resp.latency_ms + " ms" })), el("span", {}, "tokens ", el("b", { text: resp.usage.input_tokens })),
    el("span", {}, "T ", el("b", { text: a0.temperature })), el("span", {}, "backend ", el("b", { text: `${resp.backend.kind} · ${resp.backend.device}` })),
    el("span", {}, "model ", el("b", { text: resp.model.name + (resp.model.mode === "fp8_dequantized_bf16" ? " (FP8→bf16 fallback)" : "") })),
    resp.settings.provisional ? el("span", { class: "warn", text: "provisional thresholds" }) : null));
  const warns = resp.warnings.concat(rv.warnings.map((w) => `${w.path}: ${w.message}`));
  if (warns.length) box.append(el("ul", { class: "warnlist" }, warns.map((w) => el("li", { text: w }))));
  if (out.response_validation && !out.response_validation.valid)
    box.append(el("div", { class: "callout err" }, "This response failed validation and was written to the feedback log. ",
      el("button", { class: "small", text: "Open feedback", onclick: () => showTab("logs") })));
  box.append(jsonPanel(out, req));
}

async function runDecision(req, box, msg, raw) {
  msg.textContent = "scoring…"; msg.className = "msg";
  try {
    const out = await api("/api/decide", { request: req, raw: raw ?? S.adv });
    renderOutcome(box, out, req);
    msg.textContent = out.response ? `${out.response.latency_ms} ms` : "invalid request";
    msg.className = "msg" + (out.response ? "" : " err");
    if (out.feedback_logged) refreshStatus();
    return out;
  } catch (e) { msg.textContent = e.message; msg.className = "msg err"; return null; }
}

// ------------------------------------------------------------------------------------------------ tool-call gate
const ROOTS = { windows: "C:\\work\\demo-app", linux: "/home/dev/demo-app", macos: "/Users/dev/demo-app" };
function defaultTool(agent, shell) {
  if (agent === "codex") return "shell";
  if (agent === "opencode") return "bash";
  return shell === "powershell" ? "PowerShell" : "Bash";
}
let toolTouched = false, rootTouched = false;
function gateSync() {
  if (!toolTouched) $("gTool").value = defaultTool($("gAgent").value, $("gShell").value);
  if (!rootTouched) $("gRoot").value = ROOTS[$("gOs").value];
}
["gAgent", "gShell", "gOs"].forEach((id) => $(id).addEventListener("change", () => {
  if (id === "gOs") { const sh = $("gShell"); const os = $("gOs").value;
    if (os === "windows" && ["bash", "sh", "zsh"].includes(sh.value)) sh.value = "powershell";
    if (os !== "windows" && ["powershell", "cmd", "git-bash"].includes(sh.value)) sh.value = os === "macos" ? "zsh" : "bash"; }
  gateSync();
}));
$("gTool").addEventListener("input", () => (toolTouched = true));
$("gRoot").addEventListener("input", () => (rootTouched = true));

function gateRequest() {
  const raw = $("gInput").value.trim();
  if (!raw) throw new Error("enter a command or tool input");
  const agent = $("gAgent").value, shell = $("gShell").value, tool = $("gTool").value.trim() || defaultTool(agent, shell);
  let input;
  if (raw.startsWith("{")) { try { input = JSON.parse(raw); } catch (e) { throw new Error("tool input looks like JSON but does not parse: " + e.message); } }
  else if (agent === "codex" && tool === "shell") input = { command: shell === "powershell" ? ["powershell", "-Command", raw] : shell === "cmd" ? ["cmd", "/c", raw] : [shell === "git-bash" ? "bash" : shell, "-lc", raw], workdir: $("gRoot").value };
  else input = { command: raw };
  const state = { agent, os: $("gOs").value, shell, workspace_root: $("gRoot").value, cwd: $("gRoot").value, tool, input };
  const task = $("gTask").value.trim(); if (task) state.user_task = task;
  const ctx = $("gContext").value.trim();
  if (ctx) { try { state.context = JSON.parse(ctx); } catch (e) { throw new Error("context is not valid JSON: " + e.message); } }
  return { state, questions: { verdict: { type: "choice", instructions: S.status.gate_instructions, criteria: S.status.policy } } };
}
async function gateRun() {
  let req; try { req = gateRequest(); } catch (e) { $("gMsg").textContent = e.message; $("gMsg").className = "msg err"; return; }
  await runDecision(req, $("gateResult"), $("gMsg"));
}
$("gRun").addEventListener("click", gateRun);
$("gatePreset").addEventListener("change", (e) => {
  const p = S.presets.gate.find((x) => x.id === e.target.value); if (!p) return;
  $("gAgent").value = p.agent; $("gOs").value = p.os; $("gShell").value = p.shell; $("gTool").value = p.tool; toolTouched = true; rootTouched = false; gateSync();
  $("gInput").value = p.input ? JSON.stringify(p.input, null, 2) : p.command; $("gTask").value = p.user_task || "";
  $("gContext").value = p.context ? JSON.stringify(p.context) : "";
  if (p.context && !S.adv) toast("This example includes a context object (visible in Advanced mode).");
  $("gMsg").textContent = "expected: " + p.expect; $("gMsg").className = "msg";
});

// ------------------------------------------------------------------------------------------------ buckets
function bucketRow(b = {}) {
  const row = el("div", { class: "bucket" },
    el("input", { placeholder: "name", value: b.name || "", class: "bname" }),
    el("textarea", { rows: 1, placeholder: "description (optional)", class: "bdesc" }),
    el("textarea", { rows: 1, placeholder: "examples, one per line (optional)", class: "bex" }),
    el("button", { class: "small danger", text: "×", title: "remove", onclick: () => row.remove() }));
  row.querySelector(".bdesc").value = b.description || ""; row.querySelector(".bex").value = (b.examples || []).join("\n");
  return row;
}
$("bAdd").addEventListener("click", () => $("bList").append(bucketRow()));
function bucketRequest(bucketsOnly) {             // bucketsOnly: the file batch supplies its own states
  const text = $("bText").value.trim(); if (!text && !bucketsOnly) throw new Error("enter the text to classify");
  let state = { text };
  if (/^[\[{]/.test(text)) { try { const v = JSON.parse(text); if (v && typeof v === "object") state = v; } catch { /* plain text */ } }
  const crit = {};
  for (const r of $("bList").children) {
    const name = r.querySelector(".bname").value.trim(); if (!name) continue;
    if (name in crit) throw new Error(`bucket "${name}" appears twice`);
    const desc = r.querySelector(".bdesc").value.trim(), ex = r.querySelector(".bex").value.split("\n").map((s) => s.trim()).filter(Boolean);
    crit[name] = ex.length ? { what: desc || name, examples: ex } : (desc || null);
  }
  if (Object.keys(crit).length < 2) throw new Error("add at least two buckets");
  return { state, questions: { bucket: { type: "choice", instructions: $("bInstr").value.trim() || S.status.bucket_instructions, criteria: crit } } };
}
$("bRun").addEventListener("click", async () => {
  let req; try { req = bucketRequest(); } catch (e) { $("bMsg").textContent = e.message; $("bMsg").className = "msg err"; return; }
  await runDecision(req, $("bucketResult"), $("bMsg"));
});
function loadBucketPreset(p) {
  $("bText").value = p.text; $("bInstr").value = p.instructions; clear($("bList")); p.buckets.forEach((b) => $("bList").append(bucketRow(b)));
}
$("bucketPreset").addEventListener("change", (e) => { const p = S.presets.bucket.find((x) => x.id === e.target.value); if (p) loadBucketPreset(p); });

// ------------------------------------------------------------------------------------------------ bucket file batch
const BB = { input: null, parsed: null, job: null, result: null, sort: { key: "c:bucket", dir: 1 }, filter: null, limit: 300 };
// sort keys: "c:<column>" or "b:<bucket>" (that bucket's probability); filters: "bucket:<name>", "below", "errors"
const bbMsg = (t, err) => { $("bbMsg").textContent = t; $("bbMsg").className = "msg" + (err ? " err" : ""); };
const bbCell = (v) => v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
$("bbBrowse").addEventListener("click", async () => { const p = await pickPath("open", "records"); if (p) { $("bbPath").value = p; $("bbFile").value = ""; $("bbTag").value = ""; bbLoad(); } });
$("bbFile").addEventListener("change", () => { if ($("bbFile").files[0]) { $("bbPath").value = ""; $("bbTag").value = ""; bbLoad(); } });
$("bbPath").addEventListener("change", () => { $("bbTag").value = ""; });
$("bbLoad").addEventListener("click", () => bbLoad());
$("bbFormat").addEventListener("change", () => BB.input && bbLoad());
$("bbTag").addEventListener("change", () => BB.input && bbLoad());
$("bbTagSel").addEventListener("change", () => { $("bbTag").value = $("bbTagSel").value; if (BB.input) bbLoad(); });
async function bbLoad() {
  const f = $("bbFile").files[0], path = $("bbPath").value.trim();
  if (f) BB.input = { text: await f.text(), filename: f.name }; else if (path) BB.input = { path }; else return toast("Choose a file or enter a path", true);
  const body = { ...BB.input, format: $("bbFormat").value, record_tag: $("bbTag").value };
  let r; try { r = await api("/api/records/parse", body); } catch (e) { $("bbSetup").classList.add("hidden"); return toast(e.message, true); }
  BB.parsed = r; $("bbSetup").classList.remove("hidden");
  $("bbTagRow").classList.toggle("hidden", r.format !== "xml");
  if (r.format === "xml") {
    const sel = clear($("bbTagSel")), cands = r.record_candidates || [];
    const opts = cands.some((c) => c.path === r.record_tag) ? [] : [{ path: r.record_tag, count: r.count }];
    for (const c of [...opts, ...cands]) sel.append(el("option", { value: c.path, text: `${c.path}  (${c.count.toLocaleString()})` }));
    sel.value = r.record_tag;
  }
  $("bbExcluded").classList.toggle("hidden", !(r.result_fields || []).length);
  $("bbExcluded").textContent = (r.result_fields || []).length ? "Earlier OpenOne results (" + r.result_fields.join(", ") +
    ") are kept in exports but left out when classifying the whole record." : "";
  $("bbInfo").textContent = `${r.format.toUpperCase()} · ${r.count.toLocaleString()} records · ${r.fields.length} field(s)` +
    (r.record_tag ? ` · record element <${r.record_tag}>` : "") + (r.warnings.length ? " · " + r.warnings.join("; ") : "");
  // text fields: guess the most text-like column (longest average string)
  const avg = (k) => r.sample.reduce((n, x) => n + (typeof x[k] === "string" ? x[k].length : 0), 0) / Math.max(1, r.sample.length);
  const cand = r.fields.filter((k) => !(r.result_fields || []).includes(k));
  const guess = cand.length === 1 ? cand[0] : [...cand].sort((a, b) => avg(b) - avg(a))[0];
  const chips = clear($("bbFields"));
  const chip = (name, label, on) => {
    const c = el("label", { class: "chip" + (on ? " on" : "") }, el("input", { type: "checkbox", value: name, ...(on ? { checked: true } : {}) }), label);
    c.querySelector("input").addEventListener("change", (e) => {
      if (name === "*" && e.target.checked) chips.querySelectorAll("input").forEach((i) => { if (i !== e.target) { i.checked = false; i.parentNode.classList.remove("on"); } });
      if (name !== "*" && e.target.checked) { const w = chips.querySelector("input[value='*']"); w.checked = false; w.parentNode.classList.remove("on"); }
      c.classList.toggle("on", e.target.checked);
    });
    return c;
  };
  chips.append(chip("*", "whole record", false), ...r.fields.map((k) => chip(k, k, k === guess)));
  for (const id of ["bbId", "bbExpected"]) {
    const s = clear($(id)); s.append(el("option", { value: "", text: "—" }), ...r.fields.map((k) => el("option", { value: k, text: k })));
  }
  const idGuess = r.fields.find((k) => /^(id|key|uuid|ticket_?id|@id)$/i.test(k)); if (idGuess) $("bbId").value = idGuess;
  const sample = clear($("bbSample")), cols = r.fields.slice(0, 12);
  sample.append(el("tr", {}, cols.map((k) => el("th", { text: k }))), ...r.sample.map((x) => el("tr", {}, cols.map((k) => el("td", { class: "clip", text: bbCell(x[k]), title: bbCell(x[k]) })))));
  bbMsg("");
}
function bbFieldsChosen() { return [...$("bbFields").querySelectorAll("input:checked")].map((i) => i.value); }
$("bbRun").addEventListener("click", async () => {
  if (!BB.parsed) return;
  let req; try { req = bucketRequest(true); } catch (e) { return bbMsg(e.message, true); }
  const text_fields = bbFieldsChosen(); if (!text_fields.length) return bbMsg("pick at least one field to classify (or the whole record)", true);
  const body = { ...BB.input, format: BB.parsed.format, record_tag: BB.parsed.record_tag || "", question: req.questions.bucket, text_fields,
    id_field: $("bbId").value, expected_field: $("bbExpected").value };
  let j; try { j = await api("/api/bucket-batch", body); } catch (e) { return bbMsg(e.message, true); }
  BB.job = j.id; bbMsg(""); $("bbCancel").classList.remove("hidden"); $("bbRun").disabled = true; clear($("bbResult"));
  const done = await watchJob(j.id, (x) => clear($("bbProgress")).append(progressBar(x)));
  $("bbCancel").classList.add("hidden"); $("bbRun").disabled = false;
  if (done && done.state === "done") { clear($("bbProgress")); BB.result = { id: j.id, ...done.result }; BB.filter = null; BB.limit = 300; bbRender(); }
});
$("bbCancel").addEventListener("click", () => BB.job && api("/api/job/cancel", { id: BB.job }));
function bbSorted() {
  const r = BB.result, order = Object.fromEntries(r.buckets.map((b, i) => [b, i])), { key, dir } = BB.sort;
  const f = BB.filter;
  const rows = r.rows.filter((x) => f === null ? true : f === "below" ? x.status === "ok" && !x.accepted :
    f === "errors" ? x.status !== "ok" : x.accepted && "bucket:" + x.bucket === f);
  const cols = {
    "c:bucket": (x) => [order[x.bucket] ?? 1e9, x.accepted ? 0 : 1, -(x.p || 0)], "c:p": (x) => [x.p || 0], "c:index": (x) => [x.index],
    "c:id": (x) => [x.id || ""], "c:text": (x) => [x.text || ""], "c:expected": (x) => [x.expected || ""],
  };
  const val = key.startsWith("b:") ? ((x) => [x.probabilities ? x.probabilities[key.slice(2)] || 0 : 0]) : cols[key] || cols["c:index"];
  const cmp = (a, b) => { const va = val(a), vb = val(b); for (let i = 0; i < va.length; i++) { if (va[i] < vb[i]) return -dir; if (va[i] > vb[i]) return dir; } return a.index - b.index; };
  return rows.sort(cmp);
}
function bbRender() {
  const r = BB.result, box = clear($("bbResult")), m = r.meta;
  const stat = (v, l, cls) => el("div", { class: "stat" }, el("div", { class: "v " + (cls || ""), text: v }), el("div", { class: "l", text: l }));
  const below = Object.values(r.counts).reduce((n, c) => n + c.below, 0), errors = m.classified - (r.status.ok || 0);
  box.append(el("div", { class: "stats" },
    stat(m.classified.toLocaleString(), m.cancelled ? `classified (stopped; ${m.records.toLocaleString()} in file)` : "records classified"),
    stat(Object.values(r.counts).reduce((n, c) => n + c.accepted, 0).toLocaleString(), `accepted (p ≥ ${m.accept_at})`, "ok"),
    stat(below.toLocaleString(), "below threshold", below ? "warn" : ""), stat(errors.toLocaleString(), "errors / empty", errors ? "bad" : ""),
    r.accuracy !== null ? stat((r.accuracy * 100).toFixed(1) + "%", `accuracy (${r.graded.toLocaleString()} with an expected bucket)`) : null,
    r.latency_ms ? stat(r.latency_ms.p50 + " ms", `p50 per record (total ${m.seconds} s)`) : null));
  // per-bucket bars; click one to filter the table
  const max = Math.max(1, ...Object.values(r.counts).map((c) => c.accepted + c.below));
  const bars = el("div", { class: "bbars" });
  const fl = (key, label) => el("span", { class: "bfilter" + (BB.filter === key ? " sel" : ""), text: label, title: "show only these rows",
    onclick: () => { BB.filter = BB.filter === key ? null : key; BB.limit = 300; bbRender(); } });
  for (const b of r.buckets) {
    const c = r.counts[b];
    bars.append(fl("bucket:" + b, b), el("div", { class: "bbar" }, el("div", { class: "a", style: `width:${(100 * c.accepted / max).toFixed(2)}%` }), el("div", { class: "b", style: `width:${(100 * c.below / max).toFixed(2)}%` })),
      el("span", { text: `${c.accepted.toLocaleString()}${c.below ? ` + ${c.below.toLocaleString()} below` : ""}` }));
  }
  if (below) bars.append(fl("below", "below threshold"), el("span"), el("span", { text: below.toLocaleString() }));
  if (errors) bars.append(fl("errors", "errors / empty"), el("span"), el("span", { text: errors.toLocaleString() }));
  box.append(el("div", { class: "hint", text: "Solid: accepted. Faded: the bucket was the top choice but below the threshold. Click a name to filter the table." }), bars);
  // export
  const sortSel = el("select", { id: "bbExportSort" }, el("option", { value: "bucket", text: "sorted by bucket, then confidence" }),
    el("option", { value: "confidence", text: "sorted by confidence" }), el("option", { value: "original", text: "in file order" }));
  box.append(el("div", { class: "toolbar" }, el("b", { text: "Export all rows:" }),
    ...[["csv", "CSV"], ["json", "JSON"], ["jsonl", "JSON lines"], ["xml", "XML"], ["zip", "ZIP · one CSV per bucket"]].map(([f, l]) =>
      el("button", { class: "small", text: l, onclick: () => bbExport(f, sortSel.value) })), sortSel));
  // table
  const rows = bbSorted(), shown = rows.slice(0, BB.limit);
  const cols = [["c:index", "#"], ["c:id", "id"], ["c:text", "text"], ["c:bucket", "bucket"], ["c:p", "p"], ...(r.graded ? [["c:expected", "expected"]] : []),
    ...r.buckets.map((b) => ["b:" + b, "p " + b])];
  const th = ([k, l]) => el("th", { class: "sortable", text: l + (BB.sort.key === k ? (BB.sort.dir > 0 ? " ▲" : " ▼") : ""), title: "sort",
    onclick: () => { BB.sort = { key: k, dir: BB.sort.key === k ? -BB.sort.dir : (k === "c:p" || k.startsWith("b:") ? -1 : 1) }; bbRender(); } });
  const tr = (x) => el("tr", {},
    el("td", { text: x.index + 1 }), el("td", { text: x.id ?? "" }), el("td", { class: "clip", text: x.text || x.error || "", title: x.text || x.error || "" }),
    el("td", { class: x.status !== "ok" ? "bad" : x.accepted ? "ok" : "warn", text: x.status !== "ok" ? x.status : x.bucket + (x.accepted ? "" : " (below)") }),
    el("td", { text: x.p !== null && x.p !== undefined ? fmtP(x.p) : "" }),
    ...(r.graded ? [el("td", { class: x.correct === false ? "bad" : "", text: x.expected ? x.expected + (x.correct ? " ✓" : x.correct === false ? " ✗" : "") : "" })] : []),
    ...r.buckets.map((b) => el("td", { text: x.probabilities ? fmtP(x.probabilities[b]) : "" })));
  box.append(el("div", { class: "tablewrap" }, el("table", { class: "grid" }, el("tr", {}, cols.map(th)), shown.map(tr))));
  const notes = [];
  if (rows.length > shown.length) notes.push(el("button", { class: "small", text: `Show ${Math.min(500, rows.length - shown.length)} more`, onclick: () => { BB.limit += 500; bbRender(); } }));
  notes.push(el("span", { class: "hint", text: `${shown.length.toLocaleString()} of ${rows.length.toLocaleString()} row(s) shown` +
    (r.rows.length < m.classified ? ` · the table holds the first ${r.preview_limit.toLocaleString()} records; exports hold all ${m.classified.toLocaleString()}` : "") }));
  box.append(el("div", { class: "row" }, ...notes));
}
async function bbExport(format, sort) {
  const filt = { csv: "csv", json: "json", jsonl: "jsonl", xml: "xml", zip: "archive" }[format];
  const path = await pickPath("save", filt, `openone-buckets.${format}`);
  try {
    const r = await api("/api/bucket-batch/export", path ? { id: BB.result.id, format, sort, path } : { id: BB.result.id, format, sort });
    if (r.base64 !== undefined) {
      const bin = Uint8Array.from(atob(r.base64), (c) => c.charCodeAt(0));
      const type = { csv: "text/csv", json: "application/json", jsonl: "application/json", xml: "application/xml", zip: "application/zip" }[format];
      const a = el("a", { href: URL.createObjectURL(new Blob([bin], { type })), download: r.filename }); document.body.append(a); a.click(); a.remove();
    } else toast("Saved " + r.path);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------------------------------------ JSON editor
const eText = $("eText"), eGutter = $("eGutter");
let eTimer = null, eLast = null;
function eGutterRender(bad = new Set()) {
  const n = eText.value.split("\n").length; clear(eGutter);
  for (let i = 1; i <= n; i++) eGutter.append(el("div", { class: bad.has(i) ? "bad" : "", text: i }));
  eGutter.scrollTop = eText.scrollTop;
}
eText.addEventListener("scroll", () => (eGutter.scrollTop = eText.scrollTop));
eText.addEventListener("input", () => { eGutterRender(); clearTimeout(eTimer); eTimer = setTimeout(eValidate, 250); });
eText.addEventListener("keydown", (e) => {
  if (e.key === "Tab") { e.preventDefault(); const s = eText.selectionStart; eText.setRangeText("  ", s, eText.selectionEnd, "end"); eText.dispatchEvent(new Event("input")); }
});
function gotoLine(line, col = 1) {
  const lines = eText.value.split("\n"); let off = 0; for (let i = 0; i < line - 1 && i < lines.length; i++) off += lines[i].length + 1;
  eText.focus(); eText.setSelectionRange(off + col - 1, off + col - 1);
  eText.scrollTop = Math.max(0, (line - 5) * 18.75);
}
async function eValidate() {
  const text = eText.value; if (text === eLast) return; eLast = text;
  const badge = $("eBadge"), list = clear($("eErrors"));
  if (!text.trim()) { badge.className = "vbadge"; badge.textContent = "–"; eGutterRender(); return; }
  let v; try { v = await api("/api/validate", { text }); } catch (e) { badge.textContent = "?"; return; }
  if (eText.value !== text) return;
  const locs = v.errors.some((e) => e.kind !== "syntax") || v.warnings.length ? jsonLocate(text) : {};
  const bad = new Set();
  const where = (e) => {
    if (e.line) return { line: e.line, col: e.column || 1 };
    let p = e.path || ""; while (p && !locs[p]) p = p.slice(0, p.lastIndexOf("/"));
    return locs[p] || locs[""] || null;
  };
  for (const e of v.errors) {
    const w = where(e); if (w) bad.add(w.line);
    list.append(el("li", { onclick: () => w && gotoLine(w.line, w.col) }, el("code", { text: `${w ? "line " + w.line + " · " : ""}${e.path || "/"}` }), " ", e.message));
  }
  for (const e of v.warnings) { const w = where(e); list.append(el("li", { class: "warn", onclick: () => w && gotoLine(w.line, w.col) }, el("code", { text: `${w ? "line " + w.line + " · " : ""}${e.path}` }), " ", e.message)); }
  badge.className = "vbadge " + (v.valid ? "pass" : "fail");
  badge.textContent = v.valid ? (v.warnings.length ? `✓ valid · ${v.warnings.length} warning(s)` : "✓ valid request") : `✗ ${v.errors.length} error(s)`;
  eGutterRender(bad);
}
function editorTemplate(kind) {
  const pol = S.status.policy;
  const gate = { state: { agent: "claude-code", os: "linux", shell: "bash", workspace_root: "/home/dev/demo-app", cwd: "/home/dev/demo-app", tool: "Bash",
    input: { command: "npm run lint -- --fix" }, user_task: "Fix the lint errors" },
    questions: { verdict: { type: "choice", instructions: S.status.gate_instructions, criteria: pol } } };
  const bucket = { state: { text: "The checkout page shows a 500 error after I enter my card." },
    questions: { bucket: { type: "choice", instructions: "Classify the support message into exactly one bucket.",
      criteria: { billing: "Payments, invoices, refunds.", technical: { what: "Errors, crashes, bugs", examples: ["the app freezes on startup"] }, other: null } } } };
  const multi = { state: { text: "URGENT: prod database is down since 03:00, customers cannot log in" },
    questions: { team: { type: "choice", instructions: "Which team should handle this?", criteria: ["database", "frontend", "billing", "other"] },
      severity: { type: "choice", instructions: "How severe is it?", criteria: { critical: "Outage or data loss", high: null, low: null } } } };
  return { gate, bucket, multi }[kind];
}
$("eTemplate").addEventListener("change", (e) => { if (!e.target.value) return; eText.value = JSON.stringify(editorTemplate(e.target.value), null, 2); e.target.value = ""; eText.dispatchEvent(new Event("input")); });
$("eFormat").addEventListener("click", () => { try { eText.value = JSON.stringify(JSON.parse(eText.value), null, 2); eText.dispatchEvent(new Event("input")); } catch (e) { toast("Cannot format: " + e.message, true); } });
$("eRun").addEventListener("click", async () => {
  const msg = $("eMsg"); msg.textContent = "scoring…"; msg.className = "msg";
  try {
    const out = await api("/api/decide", { text: eText.value, raw: $("eRaw").checked });
    let req = null; try { req = JSON.parse(eText.value); } catch { /* syntax error shown by the validator */ }
    renderOutcome($("editorResult"), out, req);
    msg.textContent = out.response ? `${out.response.latency_ms} ms` : "invalid request"; msg.className = "msg" + (out.response ? "" : " err");
  } catch (e) { msg.textContent = e.message; msg.className = "msg err"; }
});

// ------------------------------------------------------------------------------------------------ jobs
async function watchJob(jobId, onProgress) {
  for (;;) {
    let j; try { j = await api("/api/job?id=" + encodeURIComponent(jobId)); } catch (e) { onProgress({ state: "error", error: e.message, progress: {} }); return null; }
    onProgress(j);
    if (j.state !== "running") return j;
    await new Promise((r) => setTimeout(r, 500));
  }
}
function progressBar(j) {
  const p = j.progress || {}, frac = p.total ? Math.min(1, p.done / p.total) : 0;
  const bytes = p.total > 1e6 ? ` · ${fmtBytes(p.done)} / ${fmtBytes(p.total)}` : p.total ? ` · ${p.done} / ${p.total}` : "";
  return el("div", {}, el("div", { class: "progress" + (p.total ? "" : " indet") }, el("div", { style: `width:${(frac * 100).toFixed(1)}%` })),
    el("div", { class: "ptext", text: j.state === "error" ? "Error: " + j.error : j.state === "cancelled" ? "Cancelled" : `${p.stage || ""}${bytes}${p.message ? " · " + p.message : ""}` }));
}

// ------------------------------------------------------------------------------------------------ batch
let batchJob = null;
$("batchBrowse").addEventListener("click", async () => { const p = await pickPath("open", "jsonl"); if (p) $("batchPath").value = p; });
$("batchRun").addEventListener("click", async () => {
  const f = $("batchFile").files[0], path = $("batchPath").value.trim();
  const body = {};
  if (f) body.text = await f.text(); else if (path) body.path = path; else return toast("Choose a JSONL file or enter a path", true);
  let j; try { j = await api("/api/batch", body); } catch (e) { return toast(e.message, true); }
  batchJob = j.id; $("batchCancel").classList.remove("hidden"); clear($("batchSummary"));
  const done = await watchJob(j.id, (x) => clear($("batchProgress")).append(progressBar(x)));
  $("batchCancel").classList.add("hidden");
  if (done && done.state === "done") renderBatch(done);
});
$("batchCancel").addEventListener("click", () => batchJob && api("/api/job/cancel", { id: batchJob }));
function renderBatch(j) {
  const s = j.result, box = clear($("batchSummary"));
  const stat = (v, l, cls) => el("div", { class: "stat" }, el("div", { class: "v " + (cls || ""), text: v }), el("div", { class: "l", text: l }));
  box.append(el("div", { class: "stats" },
    stat(s.rows, "rows"), stat(s.status.ok || 0, "valid responses", "ok"),
    stat(s.status.invalid_request || 0, "invalid requests", s.status.invalid_request ? "warn" : ""),
    stat(s.status.invalid_response || 0, "invalid responses (→ feedback)", s.status.invalid_response ? "bad" : ""),
    stat(s.status.bad_json || 0, "bad JSON lines", s.status.bad_json ? "warn" : ""),
    s.accuracy !== null ? stat((s.accuracy * 100).toFixed(1) + "%", `accuracy (${s.graded} graded)`) : null,
    s.latency_ms ? stat(s.latency_ms.p50 + " ms", `p50 latency (p95 ${s.latency_ms.p95}, mean ${s.latency_ms.mean})`) : null,
    ...Object.entries(s.actions).map(([k, v]) => stat(v, k.replace("_", "-")))),
    el("div", { class: "row" }, el("button", { class: "small", text: "Export JSONL", onclick: () => exportBatch(j.id, "jsonl") }),
      el("button", { class: "small", text: "Export CSV", onclick: () => exportBatch(j.id, "csv") })),
    el("div", { class: "tablewrap" }, el("table", { class: "grid" },
      el("tr", {}, ["line", "id", "status", "answers", "correct", "ms"].map((h) => el("th", { text: h }))),
      s.preview.map((r) => el("tr", {}, el("td", { text: r.line }), el("td", { text: r.id ?? "" }),
        el("td", { class: r.status === "ok" ? "ok" : "bad", text: r.status, title: (r.errors || []).join("\n") }),
        el("td", { text: Object.entries(r.answers).map(([q, a]) => `${q}: ${a.choice} ${fmtP(a.p)} → ${a.action}`).join("; ") || (r.errors || [])[0] || "" }),
        el("td", { text: r.correct === undefined || r.correct === null ? "" : r.correct ? "✓" : "✗" }), el("td", { text: r.latency_ms ?? "" }))))));
}
async function exportBatch(id, format) {
  const path = await pickPath("save", "jsonl", `openone-batch.${format}`);
  try {
    const r = await api("/api/batch/export", path ? { id, format, path } : { id, format });
    if (r.content !== undefined) download(r.filename, r.content, format === "csv" ? "text/csv" : "application/json"); else toast("Saved " + r.path);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------------------------------------ tuning
const TFIELDS = [["temperature", "toolcall", "Tool-call temperature"], ["temperature", "bucket", "Bucket temperature"],
  ["gate", "auto_run_at", "Auto-run if p(allow) ≥"], ["gate", "alarm_at", "Alarm if p(block) ≥"], ["bucket", "accept_at", "Accept bucket if p ≥"]];
async function renderTuning() {
  let s; try { s = await api("/api/settings/effective"); } catch (e) { return; }
  const box = clear($("tFields"));
  if (!s.base) { box.append(el("p", { class: "hint", text: "Import a model (or connect a server) first." })); return; }
  for (const [sect, key, label] of TFIELDS) {
    const ov = ((s.overrides || {})[sect] || {})[key];
    box.append(el("label", {}, label + ` (default ${s.base[sect][key]})`,
      el("input", { type: "number", step: "0.0001", min: "0", "data-sect": sect, "data-key": key, value: ov ?? "", placeholder: String(s.base[sect][key]), oninput: () => (S.tuningDirty = true) })));
  }
  $("tSource").textContent = `Defaults come from: temperatures ${s.base.source.temperature}; gate ${s.base.source.gate}; bucket ${s.base.source.bucket}.` +
    (s.base.provisional ? " These numbers are PROVISIONAL for this model variant." : "") + (Object.keys(s.overrides).length ? " Overrides are active." : "");
  const pol = S.status.policy; $("pAllow").value = pol.allow; $("pAsk").value = pol.ask; $("pBlock").value = pol.block;
}
$("tSave").addEventListener("click", async () => {
  const ov = {};
  for (const i of $("tFields").querySelectorAll("input")) if (i.value !== "") (ov[i.dataset.sect] ||= {})[i.dataset.key] = Number(i.value);
  try { await api("/api/settings", { overrides: ov }); S.tuningDirty = false; $("tMsg").textContent = "applied"; $("tMsg").className = "msg"; await refreshStatus(); renderTuning(); }
  catch (e) { $("tMsg").textContent = e.message; $("tMsg").className = "msg err"; }
});
$("tReset").addEventListener("click", async () => { await api("/api/settings", { overrides: {} }); S.tuningDirty = false; await refreshStatus(); renderTuning(); toast("Reset to the package defaults"); });
$("pSave").addEventListener("click", async () => {
  try { await api("/api/settings", { policy: { allow: $("pAllow").value, ask: $("pAsk").value, block: $("pBlock").value } }); await refreshStatus(); $("pMsg").textContent = "saved"; $("pMsg").className = "msg"; }
  catch (e) { $("pMsg").textContent = e.message; $("pMsg").className = "msg err"; }
});
$("pReset").addEventListener("click", async () => { await api("/api/settings", { policy: null }); await refreshStatus(); renderTuning(); toast("Policy reset to the default texts"); });

// ------------------------------------------------------------------------------------------------ settings
// the numbers this app applies when it runs the package locally; for the FP8 fallback these are the fitted fallback
// numbers, not the package's own (native FP8, vLLM) openone.json
function modelNumbers(m, fb) {
  const ls = m.local_settings, o = m.openone;
  const t = ls ? [ls.temperature.toolcall, ls.temperature.bucket] : [o.usecase_temperatures.usecase_toolcall, o.usecase_temperatures.usecase_bucket];
  const g = ls ? [ls.gate.auto_run_at, ls.gate.alarm_at] : [o.gate.auto_run_if_p_allow_at_least, o.gate.alarm_if_p_block_at_least];
  const src = ls ? `${fb ? "applied locally (FP8→bf16 fallback): " : ""}${ls.source.gate}` : "openone.json";
  const out = [el("span", { title: src }, "T tool/bucket ", el("b", { text: `${t[0]} / ${t[1]}` })),
    el("span", { title: src }, "auto-run/alarm ", el("b", { text: `${g[0]} / ${(+g[1]).toFixed(4)}` }))];
  if (fb && ls) out.push(el("span", { class: "hint", text: `local fallback numbers${ls.provisional ? " (provisional)" : ", fitted under vLLM"}; ` +
    `native FP8 (remote vLLM): T ${o.usecase_temperatures.usecase_toolcall} / ${o.usecase_temperatures.usecase_bucket}, alarm ${(+o.gate.alarm_if_p_block_at_least).toFixed(4)}` }));
  return out;
}
function modelItem(m, st) {
  if (m.broken) return el("div", { class: "item" }, el("div", { class: "title" }, m.name, el("span", { class: "vbadge fail", text: "broken" })), el("div", { class: "hint", text: m.broken }),
    el("div", { class: "row" }, el("button", { class: "small danger", text: "Remove", onclick: () => removeModel(m.name) })));
  const active = st.active_model === m.name, fb = m.local_mode === "fp8_dequantized_bf16";
  const keep = (st.dequant_cache || {})[m.name] !== false;
  return el("div", { class: "item" + (active ? " active" : "") },
    el("div", { class: "title" }, m.name, el("span", { class: "vbadge", text: m.format }), active ? el("span", { class: "vbadge pass", text: "active" }) : null,
      fb ? el("span", { class: "vbadge fail", text: "local: FP8→bf16 fallback" }) : null),
    el("div", { class: "meta" }, el("span", {}, "weights ", el("b", { text: short(m.weights_sha256) + "…" })), el("span", {}, "size ", el("b", { text: fmtBytes(m.weights_bytes) })),
      ...modelNumbers(m, fb),
      m.prompt_code.matches_app ? null : el("span", { class: "warn", text: "package prompt code differs from the app's (" + m.prompt_code.differs.join(", ") + ")" })),
    m.status ? el("div", { class: "hint", text: "Status: " + m.status }) : null,
    fb ? el("div", { class: "callout", text: st.fallback_warning }) : null,
    fb ? el("label", { class: "inline" }, el("input", { type: "checkbox", checked: keep, onchange: (e) => setDequant(m.name, e.target.checked) }),
      `Keep the dequantized bf16 copy (~${fmtBytes(m.weights_bytes * 1.38)}) ${m.dequant_cached ? "· cached" : "· not cached yet"}; unticked: redo the dequantization on every load (~10 s)`) : null,
    el("div", { class: "row" }, active ? null : el("button", { class: "small", text: "Use this model", onclick: () => activateModel(m.name) }),
      el("button", { class: "small", text: "Verify files", onclick: () => verifyModel(m.name) }),
      el("button", { class: "small danger", text: "Remove", onclick: () => removeModel(m.name) })));
}
async function setDequant(name, keep) { await api("/api/models/dequant", { name, keep }); toast(keep ? "The dequantized copy will be kept" : "Dequantizing on every load; cached copy removed"); refreshStatus(); }
async function activateModel(name) { await api("/api/models/activate", { name }); await refreshStatus(); toast("Active model changed; start the backend to load it"); }
async function removeModel(name) { if (!confirm(`Remove ${name} from this computer?`)) return; try { await api("/api/models/remove", { name }); refreshStatus(); } catch (e) { toast(e.message, true); } }
async function verifyModel(name) {
  const j = await api("/api/models/verify", { name }); toast("Verifying " + name + "…");
  const r = await watchJob(j.id, () => {});
  if (r && r.state === "done") toast(r.result.ok ? `${name}: all ${r.result.files} files match SHA256SUMS` : `${name}: FAILED (${[...r.result.bad, ...r.result.missing].join(", ")})`, !r.result.ok);
}

function renderSettings() {
  const st = S.status; if (!st) return;
  const cfg = st.backend_config, b = st.backend || {}, box = clear($("sBackend"));
  if (S.settingsEditing) return renderSettingsLists(st);
  const kindSel = el("select", { id: "sKind" }, el("option", { value: "local", text: "Local (this computer)" }), el("option", { value: "remote", text: "Remote gate service (vLLM)" }));
  kindSel.value = cfg.kind;
  const rtSel = el("select", { id: "sRuntime" }, (st.runtimes.length ? st.runtimes : [{ id: "", variant: "none installed" }]).map((r) => el("option", { value: r.id, text: `${r.variant} · ${r.origin || ""}` })));
  rtSel.value = cfg.runtime || (st.runtimes[0] || {}).id || "";
  const dev = el("select", { id: "sDevice" }, ["auto", "cpu", "cuda", "xpu", "mps"].map((d) => el("option", { value: d, text: d }))); dev.value = cfg.device || "auto";
  const dt = el("select", { id: "sDtype" }, ["auto", "bf16", "fp16", "fp32"].map((d) => el("option", { value: d, text: d }))); dt.value = cfg.dtype || "auto";
  const url = el("input", { id: "sUrl", value: cfg.remote_url || "", placeholder: "http://192.0.2.10:8130" });
  const key = el("input", { id: "sKey", type: "password", placeholder: cfg.remote_key_set ? "(key saved; type to replace)" : "API key (optional)", autocomplete: "off" });
  const local = el("div", { class: "list" }, el("div", { class: "grid3" }, el("label", {}, "Runtime", rtSel), el("label", {}, "Device", dev), el("label", {}, "Precision", dt)));
  const remote = el("div", { class: "list" }, el("label", {}, "Server URL", url), el("label", {}, "API key", key),
    el("p", { class: "hint", text: "The server's /health tells the app which model, mode and numbers it applies; the app builds and validates every response itself." }));
  const sync = () => { local.classList.toggle("hidden", kindSel.value !== "local"); remote.classList.toggle("hidden", kindSel.value !== "remote"); };
  kindSel.addEventListener("change", sync); sync();
  [kindSel, rtSel, dev, dt, url, key].forEach((x) => x.addEventListener("focus", () => (S.settingsEditing = true)));
  const stateBox = el("div", { class: "item" },
    el("div", { class: "title" }, "Status: ", el("span", { class: "pill " + (b.state || "stopped"), text: b.state || "stopped" })),
    b.state === "ready" ? el("div", { class: "meta" }, el("span", {}, "model ", el("b", { text: b.model.name })), el("span", {}, "mode ", el("b", { text: b.mode })),
      el("span", {}, "device ", el("b", { text: b.device })), el("span", {}, "runtime ", el("b", { text: b.runtime })),
      b.load_seconds ? el("span", {}, "load ", el("b", { text: b.load_seconds + " s" })) : null,
      b.dequant ? el("span", {}, "dequantized ", el("b", { text: `${b.dequant.seconds} s${b.dequant.cached ? ", cached" : ", not cached"}` })) : null) : null,
    b.error ? el("div", { class: "bad", text: b.error }) : null,
    (b.warnings || []).length ? el("ul", { class: "warnlist" }, b.warnings.map((w) => el("li", { text: w }))) : null);
  const prog = el("div", {});
  box.append(el("label", {}, "Backend", kindSel), local, remote,
    el("div", { class: "row" }, el("button", { class: "primary", text: b.state === "ready" ? "Apply and restart" : "Start", onclick: async () => {
      S.settingsEditing = false;
      const body = { kind: kindSel.value, runtime: rtSel.value || null, device: dev.value, dtype: dt.value, remote_url: url.value.trim() };
      if (key.value) body.api_key = key.value;
      await api("/api/backend/configure", body); await startBackend(prog);
    } }), el("button", { class: "small", text: "Stop", onclick: async () => { await api("/api/backend/stop", {}); refreshStatus(); } }),
    el("button", { class: "small", text: "Cancel edits", onclick: () => { S.settingsEditing = false; renderSettings(); } })), prog, stateBox);
  renderSettingsLists(st);
}
function renderSettingsLists(st) {
  clear($("sModels")).append(st.models.length ? el("div", { class: "list" }, st.models.map((m) => modelItem(m, st))) : el("p", { class: "hint", text: "No model imported yet." }));
  const avail = ((st.hardware || {}).recommend || {}).available_runtimes || [];
  clear($("sRuntimes")).append(el("div", { class: "list" },
    st.runtimes.map((r) => el("div", { class: "item" }, el("div", { class: "title" }, r.variant, el("span", { class: "vbadge", text: r.origin })),
      el("div", { class: "hint mono", text: r.python }),
      r.origin === "installed" ? el("div", { class: "row" }, el("button", { class: "small danger", text: "Remove", onclick: async () => { await api("/api/runtime/remove", { id: r.id }); refreshStatus(); } })) : null)),
    st.hardware ? null : el("button", { class: "small", text: "Detect hardware to list runtimes", onclick: async () => { await api("/api/hardware", {}); refreshStatus(); } }),
    avail.length ? el("div", { class: "row wrap" }, "Install:", avail.map((v) => el("button", { class: "small", text: v, onclick: () => installRuntime(v, $("sRuntimes")) }))) : null));
  $("sHistory").checked = !!st.save_history;
  $("sTheme").value = st.theme || "system";
  $("sAbout").textContent = `OpenOne app ${st.app_version} · data folder ${st.data_dir}`;
}
$("sHistory").addEventListener("change", (e) => api("/api/settings", { save_history: e.target.checked }));
$("sTheme").addEventListener("change", async (e) => { applyTheme(e.target.value); await api("/api/settings", { theme: e.target.value }); S.status.theme = e.target.value; });
$("sOpenData").addEventListener("click", () => api("/api/open", { what: "data" }));
$("sOnboard").addEventListener("click", () => openOnboarding());
$("sImport").addEventListener("click", () => importDialog(clear($("sModels"))));

async function startBackend(progBox, onDone) {
  let j; try { j = await api("/api/backend/start", {}); } catch (e) { toast(e.message, true); return null; }
  refreshStatus();
  const r = await watchJob(j.id, (x) => clear(progBox).append(progressBar(x)));
  await refreshStatus();
  if (r && r.state === "done") { clear(progBox); toast("Model ready"); }
  if (onDone) onDone(r);
  return r;
}
async function installRuntime(variant, progBox, onDone) {
  let j; try { j = await api("/api/runtime/install", { variant }); } catch (e) { toast(e.message, true); return; }
  const box = el("div", {}); progBox.append(box);
  const r = await watchJob(j.id, (x) => clear(box).append(progressBar(x)));
  await refreshStatus();
  if (r && r.state === "done") toast(`Runtime ${variant} installed (torch ${r.result.info.torch})`);
  if (onDone) onDone(r);
}
function importDialog(box, onDone) {
  const path = el("input", { class: "grow", placeholder: "C:\\Downloads\\openone-decider-v9-r16-bf16.zip or an extracted folder" });
  const info = el("div", {}), prog = el("div", {});
  const browse = el("button", { class: "small native-only", text: "Browse zip…", onclick: async () => { const p = await pickPath("open", "zip"); if (p) { path.value = p; inspect(); } } });
  const browseDir = el("button", { class: "small native-only", text: "Folder…", onclick: async () => { const p = await pickPath("folder"); if (p) { path.value = p; inspect(); } } });
  async function inspect() {
    clear(info);
    try {
      const r = await api("/api/models/inspect", { path: path.value.trim() });
      info.append(el("div", { class: "callout info" }, `${r.name}: ${r.files} files, ${fmtBytes(r.bytes)} listed in SHA256SUMS`,
        r.missing.length ? el("div", { class: "bad", text: "missing: " + r.missing.join(", ") }) : null,
        r.unlisted.length ? el("div", { class: "hint", text: "not covered by SHA256SUMS: " + r.unlisted.join(", ") + " (documentation is copied, anything else is skipped)" }) : null,
        /fp8/i.test(r.name) ? el("div", { class: "warn", text: "This looks like the FP8 package. Locally it runs as a bf16 fallback (dequantized); the bf16 package is recommended." }) : null));
    } catch (e) { info.append(el("div", { class: "callout err", text: e.message })); }
  }
  const go = el("button", { class: "primary", text: "Verify and import", onclick: async () => {
    go.disabled = true;
    try {
      const j = await api("/api/models/import", { path: path.value.trim() });
      const r = await watchJob(j.id, (x) => clear(prog).append(progressBar(x)));
      if (r && r.state === "done") {
        clear(prog).append(el("div", { class: "callout ok", text: `Imported ${r.result.name}: ${r.result.verified} files verified against SHA256SUMS in ${r.result.seconds} s.` }));
        await refreshStatus(); if (onDone) onDone(r);
      }
    } catch (e) { clear(prog).append(el("div", { class: "callout err", text: e.message })); }
    go.disabled = false;
  } });
  path.addEventListener("change", inspect);
  box.append(el("div", { class: "list" }, el("div", { class: "row" }, path, browse, browseDir, el("button", { class: "small", text: "Inspect", onclick: inspect })), info, el("div", { class: "row" }, go), prog));
  return path;
}

// ------------------------------------------------------------------------------------------------ logs & feedback
async function loadLog() { try { const r = await api("/api/logs?name=" + $("lName").value); $("lText").textContent = r.text || "(empty)"; $("lText").scrollTop = 1e9; } catch (e) { $("lText").textContent = e.message; } }
$("lName").addEventListener("change", loadLog); $("lRefresh").addEventListener("click", loadLog);
$("lOpen").addEventListener("click", () => api("/api/open", { what: "logs" }));
async function loadFeedback() {
  const r = await api("/api/feedback"), box = clear($("fList"));
  if (!r.count) { box.append(el("p", { class: "hint", text: "No feedback yet. Every response the app produced so far passed validation." })); return; }
  box.append(el("p", { text: `${r.count} item(s)` }), el("div", { class: "list" }, r.items.slice().reverse().map((i) => el("div", { class: "item" },
    el("div", { class: "title" }, i.kind === "invalid_response" ? el("span", { class: "vbadge fail", text: "invalid response" }) : el("span", { class: "vbadge", text: "user report" }), el("span", { class: "hint", text: i.time })),
    i.note ? el("div", { text: i.note }) : null, i.errors.length ? el("ul", { class: "warnlist" }, i.errors.map((e) => el("li", { text: `${e.path}: ${e.message}` }))) : null))));
}
$("fExport").addEventListener("click", async () => {
  const p = await pickPath("save", null, "openone-feedback.zip");
  try { const r = await api("/api/feedback/export", p ? { path: p } : {}); toast("Saved " + r.path); } catch (e) { toast(e.message, true); }
});
$("fClear").addEventListener("click", async () => { if (confirm("Delete all feedback items?")) { await api("/api/feedback/clear", {}); loadFeedback(); refreshStatus(); } });

// ------------------------------------------------------------------------------------------------ onboarding
const OB = { step: 0, choice: { backend: "local" } };
function openOnboarding() { OB.step = 0; $("onboard").classList.remove("hidden"); obRender(); }
function closeOnboarding() { $("onboard").classList.add("hidden"); }
$("obBack").addEventListener("click", () => { if (OB.step > 0) { OB.step--; obRender(); } });
$("obNext").addEventListener("click", async () => {
  if (OB.step === 4) { await api("/api/settings", { onboarded: true }); closeOnboarding(); await refreshStatus(); showTab("gate"); return; }
  OB.step++; obRender();
});
$("obSkip").addEventListener("click", async () => { await api("/api/settings", { onboarded: true }); closeOnboarding(); refreshStatus(); });

function obRender() {
  const body = clear($("obBody"));
  [...$("obSteps").children].forEach((li, i) => (li.className = i < OB.step ? "done" : i === OB.step ? "cur" : ""));
  $("obBack").disabled = OB.step === 0; $("obNext").textContent = OB.step === 4 ? "Finish" : "Next";
  [obWelcome, obHardware, obModel, obBackend, obTest][OB.step](body);
}
function obWelcome(b) {
  b.append(el("h2", { text: "Welcome to OpenOne" }),
    el("p", { text: "OpenOne is a small decision model (2B parameters). It is not a chat model. It reads a state and a question with fixed options and returns one calibrated probability per option, from a single forward pass." }),
    el("ul", {}, el("li", { text: "Tool-call gate: is a coding agent's proposed command safe to auto-run, should a human be asked, or should it raise an alarm?" }),
      el("li", { text: "Bucket classifier: put a text or JSON document into one of your buckets, with a probability per bucket." })),
    el("p", { class: "hint", text: "The answer JSON is built by this app from the model's probabilities, so it is always well-formed. The app still validates every request and response and keeps a feedback log of anything that fails." }),
    el("h3", { text: "Which package?" }),
    el("div", { class: "cards" },
      el("div", { class: "card sel" }, el("h4", { text: "bf16 (recommended for this app)" }), el("ul", {}, el("li", { text: "3.8 GB, runs locally on CPU, NVIDIA (CUDA), Intel Arc (XPU) or Apple Silicon (MPS)" }), el("li", { text: "The reference model; its openone.json numbers were fitted on this exact path" }))),
      el("div", { class: "card" }, el("h4", { text: "FP8" }), el("ul", {}, el("li", { text: "2.7 GB, faster, but only runs natively in vLLM on Linux with an Ada/Hopper/Blackwell NVIDIA GPU (the remote gate service)" }),
        el("li", { text: "Locally it falls back to bf16 (dequantized at load): no speed gain, slightly different numbers" })))));
}
async function obHardware(b) {
  b.append(el("h2", { text: "Hardware and inference runtime" }), el("p", { class: "hint", text: "Detecting hardware…" }));
  let hw; try { hw = await api("/api/hardware", {}); } catch (e) { b.append(el("div", { class: "callout err", text: e.message })); return; }
  await refreshStatus(); clear(b);
  const rec = hw.recommend, st = S.status;
  b.append(el("h2", { text: "Hardware and inference runtime" }),
    el("table", { class: "kv" }, el("tr", {}, el("th", { text: "OS" }), el("td", { text: hw.os + " · " + hw.platform })),
      el("tr", {}, el("th", { text: "CPU" }), el("td", { text: hw.cpu })), el("tr", {}, el("th", { text: "Memory" }), el("td", { text: hw.ram_gb + " GB" })),
      el("tr", {}, el("th", { text: "GPUs" }), el("td", { text: hw.gpus.map((g) => g.name + (g.memory_gb ? ` (${g.memory_gb} GB, driver ${g.driver}, compute ${g.compute_capability})` : "")).join("; ") || "none detected" }))),
    el("div", { class: "callout info" }, el("b", { text: "Recommended runtime: " + (rec.runtime || "none (remote only)") }), " — " + rec.runtime_reason));
  const prog = el("div", { class: "list" });
  if (st.runtimes.length) b.append(el("div", { class: "callout ok", text: "Installed: " + st.runtimes.map((r) => `${r.variant} (${r.origin})`).join(", ") + ". You can continue." }));
  b.append(el("p", { text: "The runtime is a private Python with PyTorch, downloaded once (PyPI and download.pytorch.org) into the app's data folder. Pick one:" }),
    el("div", { class: "row wrap" }, rec.available_runtimes.map((v) => el("button", { class: v === rec.runtime ? "primary" : "small", text: `Install ${v}${v === rec.runtime ? " (recommended)" : ""}`,
      onclick: () => installRuntime(v, prog, () => obRender()) }))), prog,
    el("details", {}, el("summary", { text: "Use an existing Python that already has torch and transformers" }),
      (() => { const i = el("input", { placeholder: "C:\\path\\to\\python.exe" }); return el("div", { class: "row" }, i, el("button", { class: "small", text: "Test and use", onclick: async () => {
        try { const r = await api("/api/runtime/custom", { python: i.value.trim() }); toast(`OK: torch ${r.torch}, devices ${r.devices.map((d) => d.device).join(", ")}`); refreshStatus(); } catch (e) { toast(e.message, true); } } })); })()),
    el("p", { class: "hint", text: "Only using a remote gate service? You can skip the runtime." }));
}
function obModel(b) {
  const st = S.status, rec = ((st.hardware || {}).recommend) || {};
  b.append(el("h2", { text: "Import the model package" }),
    el("p", { text: rec.package_reason || "Import the bf16 package for local use." }),
    el("p", { class: "hint", text: "Pick the .zip you received (or its extracted folder). Every file is checked against SHA256SUMS while it is copied into the app's data folder; the original stays where it is." }));
  if (st.models.length) b.append(el("div", { class: "callout ok", text: "Already imported: " + st.models.map((m) => m.name).join(", ") }));
  const box = el("div", {}); b.append(box);
  const fbBox = el("div", {}); b.append(fbBox);
  const showFallback = () => {
    clear(fbBox);
    const m = (S.status.models || []).find((x) => x.name === S.status.active_model);
    if (m && m.local_mode === "fp8_dequantized_bf16") {
      const keep = (S.status.dequant_cache || {})[m.name] !== false;
      fbBox.append(el("div", { class: "callout", text: S.status.fallback_warning }),
        el("label", { class: "inline" }, el("input", { type: "radio", name: "dq", checked: keep, onchange: () => setDequant(m.name, true) }), "Cache the dequantized weights (about 3.8 GB on disk, fast later starts)"),
        el("label", { class: "inline" }, el("input", { type: "radio", name: "dq", checked: !keep, onchange: () => setDequant(m.name, false) }), "Dequantize again on every load (no extra disk, ~10 s more per start)"),
        el("p", { class: "hint", text: "Thresholds for this fallback are provisional: the bf16 package's numbers are used until values fitted for the dequantized FP8 variant are available." }));
    }
  };
  importDialog(box, () => showFallback()); showFallback();
}
function obBackend(b) {
  const st = S.status;
  b.append(el("h2", { text: "Backend" }));
  const prog = el("div", {});
  const url = el("input", { placeholder: "http://192.0.2.10:8130", value: st.backend_config.remote_url || "" });
  const key = el("input", { type: "password", placeholder: st.backend_config.remote_key_set ? "(key saved)" : "API key", autocomplete: "off" });
  const cards = el("div", { class: "cards" });
  const lc = el("div", { class: "card", onclick: () => pick("local") }, el("h4", { text: "Local" }), el("ul", {}, el("li", { text: "Runs the imported package on this computer with the installed runtime" }),
    el("li", { text: st.runtimes.length ? "runtime: " + st.runtimes.map((r) => r.variant).join(", ") : "no runtime installed yet" }), el("li", { text: st.active_model ? "model: " + st.active_model : "no model imported yet" })));
  const rc = el("div", { class: "card", onclick: () => pick("remote") }, el("h4", { text: "Remote gate service" }), el("ul", {}, el("li", { text: "A Linux NVIDIA server running the OpenOne gate service (vLLM; needed for native FP8)" })), url, key);
  cards.append(lc, rc);
  function pick(k) { OB.choice.backend = k; lc.classList.toggle("sel", k === "local"); rc.classList.toggle("sel", k === "remote"); }
  pick(st.backend_config.kind || "local");
  b.append(cards, el("div", { class: "row" }, el("button", { class: "primary", text: "Start / connect", onclick: async () => {
    const body = { kind: OB.choice.backend }; if (OB.choice.backend === "remote") { body.remote_url = url.value.trim(); if (key.value) body.api_key = key.value; }
    else if (st.runtimes.length && !st.backend_config.runtime) body.runtime = st.runtimes[0].id;
    await api("/api/backend/configure", body);
    await startBackend(prog, (r) => { if (r && r.state === "done") prog.append(el("div", { class: "callout ok", text: `Ready: ${r.result.model.name} on ${r.result.device} (${r.result.mode})` })); });
  } })), prog);
  const bs = st.backend || {};
  if (bs.state === "ready") b.append(el("div", { class: "callout ok", text: `Running: ${bs.model.name} on ${bs.device}` }));
}
async function obTest(b) {
  b.append(el("h2", { text: "Test decision" }), el("p", { class: "hint", text: "Two synthetic examples: a risky command (expected: alarm) and a support message (expected: billing)." }));
  const out1 = el("div", { class: "list" }), out2 = el("div", { class: "list" });
  const run = async () => {
    const g = S.presets.gate.find((x) => x.id === "curl-sudo"), bk = S.presets.bucket[0];
    const req1 = { state: { agent: g.agent, os: g.os, shell: g.shell, workspace_root: "/home/dev/demo-app", cwd: "/home/dev/demo-app", tool: g.tool, input: { command: g.command }, user_task: g.user_task },
      questions: { verdict: { type: "choice", instructions: S.status.gate_instructions, criteria: S.status.policy } } };
    const crit = {}; bk.buckets.forEach((x) => (crit[x.name] = x.examples.length ? { what: x.description, examples: x.examples } : x.description));
    const req2 = { state: { text: bk.text }, questions: { bucket: { type: "choice", instructions: bk.instructions, criteria: crit } } };
    await runDecision(req1, out1, el("span", {}), false); await runDecision(req2, out2, el("span", {}), false);
  };
  b.append(el("div", { class: "row" }, el("button", { class: "primary", text: "Run test decisions", onclick: run })), out1, out2);
  if ((S.status.backend || {}).state === "ready") run(); else b.append(el("div", { class: "callout", text: "The backend is not running yet; go back one step and start it." }));
}

// ------------------------------------------------------------------------------------------------ keyboard + boot
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const t = document.querySelector(".tab.active");
    if (t.id === "tab-gate") gateRun(); else if (t.id === "tab-bucket") $("bRun").click(); else if (t.id === "tab-editor") $("eRun").click();
    e.preventDefault();
  }
});

(async function boot() {
  setAdv(S.adv);
  await refreshStatus();
  if (!S.status) return;
  S.presets = await api("/api/presets");
  for (const p of S.presets.gate) $("gatePreset").append(el("option", { value: p.id, text: `${p.title} (${p.expect})` }));
  for (const p of S.presets.bucket) $("bucketPreset").append(el("option", { value: p.id, text: p.title }));
  gateSync(); loadBucketPreset(S.presets.bucket[0]);
  eText.value = JSON.stringify(editorTemplate("gate"), null, 2); eGutterRender(); eValidate();
  const h = location.hash.slice(1); if (h && $("tab-" + h)) showTab(h);
  pollLoop();
})();
