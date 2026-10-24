# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import json
from pathlib import Path

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.documentation import main, render_markdown
from loom.model.lock import FileIdentity, ModelLock


def _card():
    service = {
        "programs": ["mixed"],
        "config": {"width": 4, "quantized": True},
        "parameters": {
            "dit": {
                "origin": "@checkpoint/",
                "files": [
                    "part2.safetensors",
                    {
                        "path": "part1.safetensors",
                        "origin": "@checkpoint/upstream/part1.safetensors",
                    },
                ],
            }
        },
        "tokenizers": {
            "text": {
                "path": "vocabulary.data",
                "origin": "@checkpoint/some/other-name.json",
                "format": "huggingface-json",
            }
        },
    }
    other = copy.deepcopy(service)
    other["parameters"]["dit"]["files"] = [
        {
            "path": "swatch.data",
            "origin": {
                "kind": "hf",
                "repository": "org/trained",
                "path": "full.safetensors",
                "revision": "experiment",
            },
            "format": "safetensors",
            "transform": {
                "kind": "safetensors",
                "version": 1,
                "tensors": ["layer.weight"],
                "alignment": 4096,
            },
        }
    ]
    return ModelCard(
        {
            "schema_version": 1,
            "model": {"id": "org/model", "name": "Fixture", "license": "Apache-2.0"},
            "origins": {
                "checkpoint": {
                    "kind": "hf",
                    "repository": "org/trained",
                    "revision": "candidate/v2",
                },
                "code": {"kind": "url", "url": "https://example.invalid/release/"},
            },
            "programs": {
                "mixed": {
                    "directory": "program",
                    "origin": "@code/",
                    "inputs": [
                        {
                            "path": "program/prepare.cxx",
                            "options": ["cxx:std=c++20"],
                            "data": ["program/auxiliary.data"],
                        }
                    ],
                    "libraries": [
                        {
                            "path": "program/library.data",
                            "origin": "bundles/library.loombc",
                            "format": "loombc",
                        }
                    ],
                    "config": {"width": 99, "layers": 2},
                }
            },
            "services": {
                "chat": {"kind": "text", "version": 2, "interface": {"max_tokens": 16}}
            },
            "variants": {
                "original": {
                    "description": "Original weights with explicit subsets.",
                    "partitions": {
                        "node-0": {"services": {"chat": service}},
                        "other side": {
                            "description": "Developer swatch.",
                            "services": {"chat": other},
                        },
                    },
                }
            },
            "presets": {
                "check": {
                    "variant": "original",
                    "partition": "node-0",
                    "service": "chat",
                    "programs": "mixed",
                    "request": {"prompt": "hello"},
                }
            },
        }
    )


def _lock(card):
    identities = []
    for choice in card.choices():
        for file in card.select(**choice).files.values():
            resolved = (
                {"revision": "b" * 40}
                if isinstance(file.origin, dict) and file.origin["kind"] == "hf"
                else None
            )
            identities.append(
                FileIdentity(
                    file.origin,
                    4096 if file.transform else 12,
                    "c" * 64,
                    resolved,
                    file.transform,
                )
            )
    origins = {
        name: {
            "requested": origin,
            **({"resolved": {"revision": "a" * 40}} if origin["kind"] == "hf" else {}),
        }
        for name, origin in card.document["origins"].items()
    }
    return ModelLock(ModelLock.snapshot(origins, identities))


def test_documentation_preserves_order_scopes_configuration_and_source_coordinates():
    rendered = render_markdown(_card())
    assert "## Files: original / node&#45;0 / chat / mixed" in rendered
    assert "## Files: original / other side / chat / mixed" in rendered
    assert "| <code>part2.safetensors</code> | parameters.dit&#91;0&#93;" in rendered
    assert "| <code>part1.safetensors</code> | parameters.dit&#91;1&#93;" in rendered
    assert rendered.index("<code>part2.safetensors</code>") < rendered.index(
        "<code>part1.safetensors</code>"
    )
    assert (
        "| <code>vocabulary.data</code> | tokenizers.text | huggingface&#45;json |"
        in rendered
    )
    assert (
        "https://huggingface.co/org/trained/resolve/candidate%2Fv2/some/other-name.json"
        in rendered
    )
    assert "https://example.invalid/release/program/prepare.cxx" in rendered
    assert "data for program/prepare.cxx" in rendered
    assert "<code>bundles/library.loombc</code> (relative to model card)" in rendered
    assert '"width": 4' in rendered and '"width": 99' not in rendered
    assert '"layers": 2' in rendered and '"quantized": true' in rendered
    assert '"cxx:std=c++20"' in rendered and '"prompt": "hello"' in rendered
    assert '"tensors": [' in rendered and '"layer.weight"' in rendered
    assert "No acquisition lock was supplied" in rendered
    assert "Locked output files" not in rendered


def test_lock_pins_alias_and_inline_repository_origins_and_output_identity():
    card = _card()
    rendered = render_markdown(
        card, lock=_lock(card), card_url="https://example.invalid/model/model.json"
    )
    assert (
        "https://huggingface.co/org/trained/resolve/"
        + "a" * 40
        + "/some/other-name.json"
        in rendered
    )
    assert (
        "https://huggingface.co/org/trained/resolve/" + "b" * 40 + "/full.safetensors"
        in rendered
    )
    assert "https://example.invalid/model/bundles/library.loombc" in rendered
    assert (
        "| <code>swatch.data</code> | 4096 | <code>" + "c" * 64 + "</code> |"
        in rendered
    )
    assert "locks identify the resulting archive" in rendered
    assert "candidate%2Fv2" not in rendered


def test_lock_covers_every_choice_and_matches_roots():
    card = _card()
    lock = _lock(card)
    lock.origins["checkpoint"]["requested"]["revision"] = "different"
    with pytest.raises(ManifestError, match="does not match origin"):
        render_markdown(card, lock=lock)
    lock = _lock(card)
    lock.files = {
        key: identity for key, identity in lock.files.items() if not identity.transform
    }
    with pytest.raises(ManifestError, match=r"does not cover 'swatch\.data'"):
        render_markdown(card, lock=lock)


def test_inline_local_relative_and_http_sources_are_literal_without_payload_reads():
    document = _card().document
    document["origins"]["local"] = {"kind": "local", "path": "C:\\missing"}
    document["programs"]["mixed"]["libraries"] = [
        {"path": "empty.data", "origin": {"kind": "inline", "base64": "Y29kZQ=="}},
        {"path": "local.data", "origin": "@local/sub/file.data"},
        {"path": "relative.data", "origin": "file with % and #.data"},
        {"path": "http.data", "origin": "HTTPS://example.invalid/a(b)?x=1&copy;=2|3"},
    ]
    rendered = render_markdown(
        ModelCard(document), card_url="https://example.invalid/card/model.json"
    )
    assert "Inline data (4 bytes)" in rendered and "Y29kZQ==" not in rendered
    assert "C:&#92;missing/sub/file.data</code> (local source)" in rendered
    assert "https://example.invalid/card/file%20with%20%25%20and%20%23.data" in rendered
    assert "(<HTTPS://example.invalid/a(b)?x=1&amp;copy;=2%7C3>)" in rendered


def test_author_text_cannot_inject_html_markdown_or_close_json_fences():
    document = _card().document
    document["model"].update(
        name="<script> [click](evil) | *title*",
        description="hello\n# injected\r\x1b[2J",
    )
    document["services"]["chat"]["interface"]["````\n<script>"] = "value"
    document["programs"]["mixed"]["inputs"][0]["origin"] = (
        "https://example.invalid/a>)(javascript:evil)<b"
    )
    rendered = render_markdown(ModelCard(document))
    assert rendered.startswith(
        "# &lt;script&gt; &#91;click&#93;(evil) &#124; &#42;title&#42;"
    )
    assert "hello<br>&#35; injected&#13;&#27;&#91;2J" in rendered
    assert "\n# injected" not in rendered
    assert '"````\\n<script>": "value"' in rendered
    assert "\n<script>" not in rendered
    assert "(<https://example.invalid/a%3E)(javascript:evil)%3Cb>)" in rendered


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/card.json",
        "javascript:alert(1)",
        "https://user:pass@example.com/card.json",
    ],
)
def test_invalid_publication_url_rejected(url):
    with pytest.raises(ManifestError, match="HTTP\\(S\\)"):
        render_markdown(_card(), card_url=url)


def test_real_model_lists_checkpoint_tokenizer_and_development_abi():
    card = ModelCard.parse(
        (Path(__file__).parent / "testdata/moonshine.json").read_text(encoding="utf-8")
    )
    rendered = render_markdown(card)
    assert "Streaming English speech recognition from 16 kHz mono audio." in rendered
    revision = "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a"
    prefix = (
        "https://huggingface.co/moonshine-ai/moonshine-streaming-tiny/resolve/"
        + revision
    )
    assert prefix + "/model.safetensors" in rendered
    assert prefix + "/tokenizer.json" in rendered
    assert "parameters.rotary&#91;0&#93;" in rendered
    assert "Inline data (136 bytes)" in rendered
    assert "unversioned development ABI" in rendered
    assert '"sample_rate": 16000' in rendered


def test_preset_guidance_keeps_assumptions_and_unknown_components():
    document = _card().document
    document["presets"]["check"]["memory"] = {
        "weights_bytes": 180000000,
        "workload": "Original F32; history and runtime pools excluded.",
    }
    rendered = render_markdown(ModelCard(document))
    assert "Estimated memory: weights 171.7 MiB; transients unknown." in rendered
    assert "Original F32; history and runtime pools excluded." in rendered
    assert '"weights_bytes": 180000000' in rendered
    assert '"transient_bytes"' not in rendered


def test_cli_is_utf8_metadata_only_and_retains_output_on_invalid_input(
    tmp_path, capsys
):
    card = tmp_path / "model.json"
    document = _card().document
    document["model"]["name"] = "模型"
    card.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "reference.md"
    assert main(["--card", str(card), "--output", str(output)]) == 0
    expected = output.read_bytes()
    assert expected.startswith("# 模型\n".encode())
    assert main(["--card", str(card)]) == 0
    assert capsys.readouterr().out.encode() == expected
    assert main(["--card", str(card), "--output", str(card)]) == 2
    assert "must not overwrite" in capsys.readouterr().err
    hardlink = tmp_path / "alias.json"
    hardlink.hardlink_to(card)
    assert main(["--card", str(card), "--output", str(hardlink)]) == 2
    assert "must not overwrite" in capsys.readouterr().err
    card.write_text("{broken")
    assert main(["--card", str(card), "--output", str(output)]) == 2
    assert output.read_bytes() == expected


def test_standalone_overview_and_upstream_links_precede_generated_contract(tmp_path):
    document = _card().document
    document["model"]["origin"] = {
        "kind": "hf",
        "repository": "org/trained",
        "revision": "model/v1",
    }
    overview = "## Run this model\n\nEnglish transcription. [Upstream](https://example.org/model).\n\n```sh\nloom-fetch model.json --preset=check --output=./installed\n```\n"
    card = tmp_path / "model.json"
    card.write_text(json.dumps(document))
    introduction = tmp_path / "OVERVIEW.md"
    introduction.write_text(overview)
    output = tmp_path / "reference.md"
    assert (
        main(
            [
                "--card",
                str(card),
                "--overview",
                str(introduction),
                "--card-url",
                "https://example.org/release/model.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    introduction.unlink()
    rendered = output.read_text()
    assert overview.strip() in rendered
    assert rendered.index("## Run this model") < rendered.index("## Model contract")
    assert "(<https://example.org/release/model.json>)" in rendered
    assert "(<https://huggingface.co/org/trained>)" in rendered
    assert "(<https://huggingface.co/org/trained/tree/model%2Fv1>)" in rendered
    assert "(<https://example.invalid/release/>)" in rendered
    assert str(tmp_path) not in rendered
    assert "parameters.dit" in rendered and "## Services" in rendered


def test_overview_is_an_explicit_input_and_cannot_be_overwritten(tmp_path, capsys):
    card = tmp_path / "model.json"
    card.write_text(json.dumps(_card().document))
    overview = tmp_path / "overview.md"
    overview.write_text("A publisher-authored overview.\n")
    assert (
        main(
            [
                "--card",
                str(card),
                "--overview",
                str(overview),
                "--output",
                str(overview),
            ]
        )
        == 2
    )
    assert "must not overwrite" in capsys.readouterr().err
    assert overview.read_text() == "A publisher-authored overview.\n"
    output = tmp_path / "reference.md"
    output.write_text("previous publication")
    assert (
        main(
            [
                "--card",
                str(card),
                "--overview",
                str(tmp_path / "missing.md"),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert output.read_text() == "previous publication"
