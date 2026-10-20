"""Prompt items: the trained Decider prompt for one {state, question}, as eval_decider_rows.py builds it.

Plain state-first layout, options in the caller's order (no shuffling), up to 255 options, and at most
MAX_STATE_TOKENS tokens of "Context:\\n" + state. The answer slot is the final " (" token. Tool-call (gate) states over
the limit get the INTEGRATION.md field elision first (head + tail per long string field); a state that still does not
fit is cut at the limit, as in training, and flagged.
"""
from .policy import is_gate
from .vendor.decider import prompt as P
from .vendor.decider import systemone as S1

MAX_STATE_TOKENS = 2048
ELIDE_STEPS = ((2500, 1500, 700), (1200, 700, 400), (600, 350, 200))   # (field limit, head, tail) characters
QUESTION_KEYS = ("type", "instructions", "criteria")


class _Fixed:
    def shuffle(self, x):
        pass

    def sample(self, xs, k):
        return xs[:k]


class _Q:
    def __init__(self, text, options):
        self.text, self.options, self.gold = text, options, -1


class _Ex:
    def __init__(self, context, qs):
        self.context, self.qs = context, qs


def elide(x, limit, head, tail):
    if isinstance(x, str) and len(x) > limit:
        return x[:head] + f"\n...[{len(x) - head - tail} characters omitted]...\n" + x[-tail:]
    if isinstance(x, dict):
        return {k: elide(v, limit, head, tail) for k, v in x.items()}
    if isinstance(x, list):
        return [elide(v, limit, head, tail) for v in x]
    return x


class Builder:
    def __init__(self, tok, max_state_tokens=MAX_STATE_TOKENS):
        self.tok = tok
        self.max_state_tokens = max_state_tokens
        self.label_names, self.label_ids, _ = P.label_table(tok)
        P.letter_ids(tok)                                   # asserts A..J are single tokens

    def state_tokens(self, state):
        return len(self.tok.encode("Context:\n" + S1.render_state(state), add_special_tokens=False))

    def build(self, state, question, want_text=False):
        """-> dict(ids, slot, letter_ids, letters, names, state_tokens, elided, truncated, [prompt_text])."""
        q = {k: question[k] for k in QUESTION_KEYS if k in question}
        rq = S1.render_question(q)
        n_state = self.state_tokens(state)
        elided = False
        if is_gate(question) and n_state > self.max_state_tokens and isinstance(state, (dict, list)):
            for limit, head, tail in ELIDE_STEPS:
                cut = elide(state, limit, head, tail)
                n_cut = self.state_tokens(cut)
                if n_cut < n_state:
                    state, n_state, elided = cut, n_cut, True
                if n_state <= self.max_state_tokens:
                    break
        it = P.build(_Ex(S1.render_state(state), [_Q(rq["question"], rq["options"])]), self.tok, _Fixed(),
                     max_options=P.MAX_OPTIONS, max_ctx_tokens=self.max_state_tokens)
        n = it["nopts"][0]
        slot = it["slots"][0]
        assert slot == len(it["ids"]) - 1 and it["perms"][0] == list(range(n)), "unexpected prompt layout"
        out = {"ids": it["ids"], "slot": slot, "letter_ids": self.label_ids[:n], "letters": self.label_names[:n],
               "names": [str(x) for x in rq["names"]], "state_tokens": n_state, "elided": elided,
               "truncated": n_state > self.max_state_tokens}
        if want_text:
            out["prompt_text"] = self.tok.decode(it["ids"])
        return out
