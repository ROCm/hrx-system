# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build admission rejects incomplete or misleading declared inputs before I/O."""

import json
from pathlib import Path

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.build import build


def _plan():
    root = Path(__file__).parents[4]
    card = root / "loom/src/loom/serve/fixture/text/native.card.json"
    selection = ModelCard.parse(card.read_text()).select("check")
    return {
        "card": str(card),
        "preset": "check",
        "encoding": "bc",
        "files": {
            path: {"input": "unopened", "output": "unwritten"}
            for path, file in selection.files.items()
            if "program" in file.roles
        },
        "data": {
            path: "unopened"
            for path, file in selection.files.items()
            if "auxiliary" in file.roles
        },
        "assets": {
            path: "unopened"
            for path, file in selection.files.items()
            if not file.roles & {"program", "auxiliary"}
        },
    }


@pytest.mark.parametrize("group", ["files", "data"])
def test_missing_declared_inputs_fail_before_linking(group):
    plan = _plan()
    plan[group].pop(next(iter(plan[group])))
    with pytest.raises(ManifestError, match="must match the selected closure"):
        build(plan, "unavailable-linker")


@pytest.mark.parametrize("group", ["files", "data", "assets"])
def test_unused_declared_inputs_fail_before_linking(group):
    plan = _plan()
    plan[group]["extra.data"] = "unopened"
    with pytest.raises(ManifestError, match=r"unused.*extra\.data"):
        build(plan, "unavailable-linker")


def test_auxiliary_code_is_not_silently_indexed():
    plan = _plan()
    plan["files"]["weights.loom"] = plan["data"].pop("weights.loom")
    with pytest.raises(ManifestError, match=r"unused.*weights\.loom"):
        build(plan, "unavailable-linker")


def test_output_documents_have_reserved_paths(tmp_path):
    plan = _plan()
    document = json.loads(Path(plan["card"]).read_text())
    document["programs"]["source"]["inputs"].append("model.json")
    card = tmp_path / "input.json"
    card.write_text(json.dumps(document))
    plan["card"] = str(card)
    plan["files"]["model.json"] = {"input": "unopened", "output": "unwritten"}
    with pytest.raises(ManifestError, match="conflicts with a generated document"):
        build(plan, "unavailable-linker")


def test_preservation_retains_provider_contract_without_a_linker(tmp_path):
    plan = _plan()
    document = json.loads(Path(plan["card"]).read_text())
    provider = document["programs"]["source"]["inputs"][0]
    provider.update(
        origin="https://example.com/prepare.cxx",
        format="cxx",
        options=["cxx:std=c++20"],
    )
    card = tmp_path / "input.json"
    card.write_text(json.dumps(document))
    plan.update(
        card=str(card),
        encoding="preserve",
        name="fixture-text",
        output_card=str(tmp_path / "model.json"),
        output_deployment=str(tmp_path / "serve.json"),
    )
    build(plan)
    result = ModelCard.parse((tmp_path / "model.json").read_text()).select("check")
    assert result.program["inputs"][0] == {
        "path": "prepare.loom",
        "format": "cxx",
        "options": ["cxx:std=c++20"],
        "data": ["weights.loom"],
    }
    assert result.parameters == ModelCard(document).select("check").parameters


def test_runtime_acquisition_survives_build_without_reading_weights(tmp_path):
    plan = _plan()
    document = json.loads(Path(plan["card"]).read_text())
    document["origins"] = {
        "checkpoint": {"kind": "hf", "repository": "org/model", "revision": "pinned"},
        "unused": {"kind": "url", "url": "https://example.invalid/code/"},
    }
    service = document["variants"]["fixture"]["services"]["chat"]
    service["parameters"]["model"]["origin"] = "@checkpoint/"
    service["parameters"]["model"]["files"][1]["transform"] = {
        "kind": "safetensors",
        "version": 1,
        "alignment": 4096,
        "tensors": ["bias"],
    }
    service["tokenizers"]["text"] = {
        "path": "tokenizer.json",
        "origin": "@checkpoint/some/other-name.json",
    }
    card = tmp_path / "input.json"
    card.write_text(json.dumps(document))
    plan.update(
        card=str(card),
        encoding="preserve",
        assets={},
        name="fixture-text",
        output_card=str(tmp_path / "model.json"),
        output_deployment=str(tmp_path / "serve.json"),
    )
    build(plan)
    result = ModelCard.parse((tmp_path / "model.json").read_text())
    assert result.document["origins"] == {
        "checkpoint": document["origins"]["checkpoint"]
    }
    selected = result.select("check")
    assert selected.files["bias.data"].origin == "@checkpoint/bias.data"
    assert selected.files["bias.data"].transform["tensors"] == ["bias"]
    assert selected.files["tokenizer.json"].origin == "@checkpoint/some/other-name.json"
    assert selected.files["weights.loom"].origin == "weights.loom"
    assert selected.files["prepare.loom"].origin == "prepare.loom"
    assert not (tmp_path / "bias.data").exists()
