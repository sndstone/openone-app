"""Builder + scorer -> the raw per-question results the pipeline consumes (shared by the local worker and the server)."""
from .prompting import MAX_STATE_TOKENS, Builder


class Engine:
    def __init__(self, scorer, max_state_tokens=MAX_STATE_TOKENS):
        self.scorer = scorer
        self.builder = Builder(scorer.tok, max_state_tokens)

    def score(self, specs):
        """specs: [{"state", "question", "want_text"}] -> [raw dict] in order."""
        items = [self.builder.build(s["state"], s["question"], bool(s.get("want_text"))) for s in specs]
        scored = self.scorer.score(items)
        out = []
        for it, (lp, ms) in zip(items, scored):
            r = {"names": it["names"], "letters": it["letters"], "letter_logprobs": lp, "input_tokens": len(it["ids"]),
                 "state_tokens": it["state_tokens"], "max_state_tokens": self.builder.max_state_tokens,
                 "elided": it["elided"], "truncated": it["truncated"], "forward_ms": ms}
            if "prompt_text" in it:
                r["prompt_text"] = it["prompt_text"]
            out.append(r)
        return out
