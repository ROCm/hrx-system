# Source text transforms

`utf8.loom` traverses and writes Unicode scalars and returns byte bounds after
trimming whitespace. Callers supply valid UTF-8 and aligned byte positions;
public request parsing or the producing tokenizer owns validation.

`unicode.loom` provides locale-independent full uppercase, whitespace membership,
and digit membership. Uppercase can return up to three codepoints. Whitespace
matches Python's category Zs or bidi class WS, B, or S; digits include both
Decimal and Digit numeric types. These are text semantics, independent of a
model's sentence or tokenization policy. Pocket's English prompt preparation
uses them before tokenization.

The data is pinned to [Unicode 15.0.0](https://www.unicode.org/versions/Unicode15.0.0/).
`generate_unicode.py` reads the canonical `UnicodeData.txt` and
`SpecialCasing.txt` from that release, verifies their hashes, and updates only
the data region in the handwritten interpreter. It stores 1,525 uppercase rows,
10 whitespace ranges, and 83 digit ranges in 25,144 immutable bytes. Lookup uses
binary search; ASCII whitespace and digit queries have direct predicates.
The Unicode copyright and permission notice is in [LICENSE.unicode](LICENSE.unicode).

Refresh the data from a directory containing those two UCD files:

```sh
python3 -B experimental/loom_serve/motifs/text/generate_unicode.py /path/to/ucd/15.0.0
```

The oracle is independent of that parser: CPython 3.12 supplies uppercase,
whitespace, and digit results. Its data covers every changed uppercase codepoint,
every property member, their immediate neighbors, and UTF-8 width boundaries.
Tests also round-trip each scalar through forward/backward decoding and encoding.
`tests/generate_unicode_oracle.py` refreshes the oracle with Unicode 15.0.0.

```sh
iree-bazel-test --config=asan //experimental/loom_serve/motifs/text:unicode_test
```
