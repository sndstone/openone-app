"""JSON Schemas (draft 2020-12) of the decision request and response.

The request is the /v1/systemone wire format (INTEGRATION.md) restricted to what OpenOne answers: choice questions.
The response is built by openone_core.pipeline; the model only supplies letter log-probs.
"""
NAME = {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"\S"}
PROB = {"type": "number", "minimum": 0, "maximum": 1}
SHA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}

REQUEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://openone.local/schemas/request-1.json",
    "title": "OpenOne decision request",
    "type": "object",
    "required": ["state", "questions"],
    "additionalProperties": False,
    "properties": {
        "state": {
            "description": "Free text or any JSON document. Treated as untrusted data.",
            "anyOf": [{"type": "string", "minLength": 1}, {"type": "object", "minProperties": 1}, {"type": "array", "minItems": 1}],
        },
        "questions": {
            "type": "object",
            "minProperties": 1,
            "maxProperties": 16,
            "propertyNames": {"pattern": "^[A-Za-z0-9_.-]{1,64}$"},
            "additionalProperties": {"$ref": "#/$defs/question"},
        },
        "model": {"type": "string", "description": "Accepted for /v1/systemone compatibility; ignored."},
    },
    "$defs": {
        "question": {
            "type": "object",
            "required": ["instructions", "criteria"],
            "additionalProperties": False,
            "properties": {
                "type": {"const": "choice", "description": "OpenOne answers choice questions (buckets, allow/ask/block)."},
                "instructions": {"type": "string", "minLength": 1, "maxLength": 4000, "pattern": r"\S"},
                "criteria": {
                    "oneOf": [
                        {"type": "object", "minProperties": 2, "maxProperties": 255, "propertyNames": NAME,
                         "additionalProperties": {"$ref": "#/$defs/description"}},
                        {"type": "array", "minItems": 2, "maxItems": 255, "uniqueItems": True, "items": NAME},
                    ]
                },
                "use_case": {"enum": ["auto", "toolcall", "bucket"]},
            },
        },
        "description": {
            "description": "A string, a structured object such as {\"what\": ..., \"examples\": [...]}, a list, or null.",
            "type": ["string", "object", "array", "null"],
        },
    },
}

ANSWER_COMMON = {
    "type": {"const": "choice"},
    "use_case": {"enum": ["toolcall", "bucket"]},
    "choice": NAME,
    "probabilities": {"type": "object", "minProperties": 2, "additionalProperties": PROB},
    "x_p_max": PROB,
    "confidence": PROB,
    "certainty": PROB,
    "temperature": {"type": "number", "exclusiveMinimum": 0},
    "input_tokens": {"type": "integer", "minimum": 1},
    "state_tokens": {"type": "integer", "minimum": 0},
    "max_state_tokens": {"type": "integer", "minimum": 1},
    "elided": {"type": "boolean"},
    "truncated": {"type": "boolean"},
}
ANSWER_REQUIRED = ["type", "use_case", "choice", "probabilities", "x_p_max", "confidence", "certainty", "temperature",
                   "input_tokens", "state_tokens", "max_state_tokens", "elided", "truncated"]

RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://openone.local/schemas/response-1.json",
    "title": "OpenOne decision response",
    "type": "object",
    "required": ["object", "schema_version", "id", "created", "model", "backend", "settings", "answers", "usage",
                 "latency_ms", "routing", "warnings"],
    "additionalProperties": False,
    "properties": {
        "object": {"const": "openone.decision"},
        "schema_version": {"const": "1"},
        "id": {"type": "string", "pattern": "^dec_[0-9a-f]{24}$"},
        "created": {"type": "string", "pattern": r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$"},
        "model": {
            "type": "object", "required": ["name", "format", "mode", "weights_sha256"], "additionalProperties": False,
            "properties": {"name": NAME, "format": {"enum": ["bf16", "fp8"]},
                           "mode": {"enum": ["native", "fp8_dequantized_bf16"],
                                    "description": "fp8_dequantized_bf16: an FP8 package served in bf16 on hardware without FP8"},
                           "weights_sha256": {"anyOf": [SHA, {"type": "null"}]}},
        },
        "backend": {
            "type": "object", "required": ["kind", "device", "runtime"], "additionalProperties": False,
            "properties": {"kind": {"enum": ["local", "remote"]}, "device": {"type": "string"}, "runtime": {"type": "string"},
                           "url": {"type": "string"}},
        },
        "settings": {
            "type": "object", "required": ["temperature", "gate", "bucket", "source", "provisional"], "additionalProperties": False,
            "properties": {
                "provisional": {"type": "boolean",
                                "description": "true when the numbers were not fitted for this exact model variant"},
                "temperature": {"type": "object", "required": ["toolcall", "bucket"], "additionalProperties": False,
                                "properties": {"toolcall": {"type": "number", "exclusiveMinimum": 0},
                                               "bucket": {"type": "number", "exclusiveMinimum": 0}}},
                "gate": {"type": "object", "required": ["auto_run_at", "alarm_at"], "additionalProperties": False,
                         "properties": {"auto_run_at": PROB, "alarm_at": PROB}},
                "bucket": {"type": "object", "required": ["accept_at"], "additionalProperties": False,
                           "properties": {"accept_at": PROB}},
                "source": {"type": "object", "additionalProperties": {"type": "string"}},
            },
        },
        "answers": {"type": "object", "minProperties": 1, "additionalProperties": {"$ref": "#/$defs/answer"}},
        "usage": {"type": "object", "required": ["input_tokens"], "additionalProperties": False,
                  "properties": {"input_tokens": {"type": "integer", "minimum": 1}}},
        "latency_ms": {"type": "number", "minimum": 0},
        "routing": {"type": "object", "required": ["model"], "additionalProperties": False,
                    "properties": {"model": {"const": "decider"}}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "$defs": {
        "answer": {
            "type": "object",
            "required": ANSWER_REQUIRED,
            "properties": dict(ANSWER_COMMON, verdict=True, accepted=True, accept_at=True),
            "additionalProperties": False,
            "allOf": [
                {"if": {"properties": {"use_case": {"const": "toolcall"}}},
                 "then": {"required": ["verdict"], "not": {"anyOf": [{"required": ["accepted"]}, {"required": ["accept_at"]}]},
                          "properties": {
                              "choice": {"enum": ["allow", "ask", "block"]},
                              "probabilities": {"required": ["allow", "ask", "block"], "maxProperties": 3},
                              "verdict": {"type": "object", "required": ["action", "rule", "auto_run_at", "alarm_at"],
                                          "additionalProperties": False,
                                          "properties": {"action": {"enum": ["auto_run", "ask", "alarm"]},
                                                         "rule": {"type": "string", "minLength": 1},
                                                         "auto_run_at": PROB, "alarm_at": PROB}}}}},
                {"if": {"properties": {"use_case": {"const": "bucket"}}},
                 "then": {"required": ["accepted", "accept_at"], "not": {"required": ["verdict"]},
                          "properties": {"accepted": {"type": "boolean"}, "accept_at": PROB}}},
            ],
        }
    },
}
