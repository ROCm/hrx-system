# Finite decisions through SystemOne

`POST /v1/systemone` accepts a catalog `model`, a string/object/array `state`,
and a nonempty map of `questions`. The native adapter implements the
[TypeSafe request/response profile](https://docs.typesafe.ai/api); the selected
source package owns formatting, tokenization, capacity, learned heads and
probabilities. Registering a decision deployment does not make a checkpoint
compatible with this source ABI.

```json
{
  "model": "judge",
  "state": "The customer cannot log in.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose a team.",
      "criteria": {"support": "Account access", "billing": "Payments"}
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate urgency.",
      "criteria": ["Low", "Medium", "High"]
    },
    "needs_reply": {"type": "noul", "instructions": "Is a reply needed?"}
  }
}
```

Choice criteria contain 1–255 named string/object/array/null descriptions.
Scores have 2–10 ordered string/object/array levels. Noul optionally describes
`false` and `true` through string/object/array criteria. Instructions also
accept strings, objects or arrays. Question and choice IDs are nonempty decoded
UTF-8 strings; duplicate decoded IDs reject, including escaped aliases.
The native schema bound is 1,024 questions; source capacity can be smaller.
The transport separately bounds total body bytes and accepted connections.

Responses retain the original question and option identities. Choice returns
the winning ID and the complete probability map. Score returns the expected
zero-based level, its probability map, and a legend preserving the original
typed criteria. Noul returns the probability of `true`, with no confidence
field. Usage contains source-reported `input_tokens` and `output_tokens:0`:
this contract performs no autoregressive generation.

Choice and score confidence follow the documented
[distribution statistics](https://docs.typesafe.ai/confidence). Equal maxima
select the first public ordinal; a single-option choice has confidence one.
The source returns normalized F32 vectors. Nonfinite, negative, over-one,
wrong-width or materially unnormalized results fail; the host does not apply
softmax, truncate or renormalize. Sum validation permits four F32 epsilons per
element for accumulated rounding. This is a serialization tolerance, not a
quality claim about a learned model.

This local profile uses strict JSON so structured legends remain valid JSON
when echoed. Comments, trailing commas, malformed Unicode, unknown fields,
unsupported question types and image-input extensions reject. Schema and
source input failures return 422; the shared local admission policy uses 503
for exhausted request/parameter capacity. Model/endpoint mismatch returns 404.
The loopback listener has no hosted authentication or rate-limit emulation.

## Source boundary

[protocol.h](protocol.h) defines indexed records followed by the original JSON
bytes. Escaped strings remain escaped in source input; structured values retain
their JSON type and exact ranges. The source can use the existing `json`
capability for model-specific canonical formatting. Public option order is
explicit: a model sorting its internal schema must map probabilities back to
those ordinals. Score rows are ascending levels; noul rows are `[false,true]`.

The [finite runner](../README.md) shares JIT, parameter residency,
cohort scheduling, cancellation and compact feedback with embeddings. Only
parsing and response encoding differ. No decision-model name appears in this
native adapter. The [deterministic package](../../../fixture/decision/README.md)
demonstrates actual VM/GPU/HTTP execution without downloading weights; it is not
CLEF or another learned decision model.
