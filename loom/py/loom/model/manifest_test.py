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


def _document():
    # This is also the native server's executable fixture, not a second Python
    # interpretation of the model's parameter/configuration contract.
    root = Path(__file__).parents[4]
    return json.loads(
        (root / "loom/src/loom/serve/fixture/text/native.card.json").read_text()
    )


def _service(document):
    return document["variants"]["fixture"]["services"]["chat"]


def test_components_share_files_and_retain_independent_contracts():
    document = _document()
    document["services"]["chat"]["components"] = {
        "image": {"version": 3, "options": ["patches"]}
    }
    document["presets"]["check"]["components"] = {
        "image": {
            "options": {"patches": 4},
            "execution": {"specialization_cache": {"capacity": 2}},
        }
    }
    _service(document)["parameters"]["vision"] = copy.deepcopy(
        _service(document)["parameters"]["model"]
    )
    selected = ModelCard(document).select("check")
    assert selected.service["version"] == 1
    assert selected.service["components"]["image"]["version"] == 3
    assert len(selected.files) == len(ModelCard(_document()).select("check").files)
    projected = ModelCard(selected.document).select()
    assert projected.service == selected.service
    assert projected.settings == selected.settings
    assert projected.parameters["vision"] == projected.parameters["model"]


@pytest.mark.parametrize(
    "component", [{"version": 0}, {"version": True}, {"programs": ["source"]}]
)
def test_component_contract_rejects_invalid_version_or_independent_files(component):
    document = _document()
    document["services"]["chat"]["components"] = {"image": component}
    with pytest.raises(ManifestError):
        ModelCard(document)


def test_component_presets_require_declared_roles_and_options():
    document = _document()
    document["presets"]["check"]["components"] = {"image": {"options": {"patches": 4}}}
    with pytest.raises(ManifestError, match="undeclared component"):
        ModelCard(document)
    document["services"]["chat"]["components"] = {"image": {}}
    with pytest.raises(ManifestError, match="undeclared model options"):
        ModelCard(document)
    document["services"]["chat"]["components"]["image"]["options"] = ["patches"]
    assert (
        "version"
        not in ModelCard(document).select("check").service["components"]["image"]
    )


def test_memory_estimate_survives_exact_selection_and_projection():
    card = ModelCard(_document())
    selected = card.select("check")
    assert selected.settings["memory"]["weights_bytes"] == 4
    assert "transient_bytes" not in selected.settings["memory"]
    assert ModelCard(selected.document).select("check").settings == selected.settings
    selected.settings["memory"]["weights_bytes"] = 999
    assert card.select("check").settings["memory"]["weights_bytes"] == 4


@pytest.mark.parametrize(
    "override",
    [
        {"programs": "archive"},
        {"service": "development"},
        {"variant": "different"},
        {"partition": "other side"},
    ],
)
def test_changed_selection_drops_estimate_in_settings_and_projected_card(override):
    document = _document()
    document["variants"]["different"] = copy.deepcopy(document["variants"]["fixture"])
    if "partition" in override:
        scope = document["variants"]["fixture"]
        document["variants"]["fixture"] = {
            "partitions": {"node-0": scope, "other side": copy.deepcopy(scope)}
        }
        for settings in document["presets"].values():
            settings["partition"] = "node-0"
    card = ModelCard(document)
    selected = card.select("check", **override)
    assert "memory" not in selected.settings
    assert "memory" not in selected.document["presets"]["check"]
    assert card.select("check").settings["memory"]["weights_bytes"] == 4


def test_implicit_single_partition_keeps_matching_estimate():
    document = _document()
    document["variants"]["fixture"] = {
        "partitions": {"only": document["variants"]["fixture"]}
    }
    selected = ModelCard(document).select("check", partition="only")
    assert selected.settings["memory"]["weights_bytes"] == 4


@pytest.mark.parametrize("field", ["weights_bytes", "transient_bytes"])
@pytest.mark.parametrize("value", [-1, 2**64, 1.5, True, "8", None])
def test_memory_counts_are_optional_unsigned_integer_bytes(field, value):
    document = _document()
    document["presets"]["check"]["memory"][field] = value
    with pytest.raises(ManifestError, match="integer bytes"):
        ModelCard(document)


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"workload": "one request"},
        {"weights_bytes": 0},
        {"workload": "", "weights_bytes": 1},
        {"workload": "one request", "weights_bytes": 1, "device": "gpu"},
    ],
)
def test_memory_estimates_require_assumptions_and_counts_only(value):
    document = _document()
    document["presets"]["check"]["memory"] = value
    with pytest.raises(ManifestError):
        ModelCard(document)


def test_memory_zero_and_uint64_max_are_explicit_values():
    document = _document()
    memory = {
        "workload": "Count boundaries",
        "weights_bytes": 0,
        "transient_bytes": 2**64 - 1,
    }
    document["presets"]["check"]["memory"] = memory
    assert ModelCard(document).select("check").settings["memory"] == memory


def test_partition_selection_preserves_one_complete_closure():
    document = _document()
    services = document["variants"]["fixture"].pop("services")
    document["variants"]["fixture"]["partitions"] = {
        "node-0": {"services": services},
        "other side": {
            "description": "Second closure",
            "services": copy.deepcopy(services),
        },
    }
    for settings in document["presets"].values():
        settings["partition"] = "node-0"
    other = document["variants"]["fixture"]["partitions"]["other side"]["services"][
        "chat"
    ]
    other["parameters"]["model"]["files"] = [
        {"path": "second.data", "format": "safetensors"}
    ]
    other["config"]["fixture.bias_delta"] = 3
    card = ModelCard(document)
    selected = card.select("check", partition="other side")
    assert selected.partition == "other side"
    assert selected.config["fixture.bias_delta"] == 3
    assert "second.data" in selected.files and "bias.data" not in selected.files
    assert list(selected.document["variants"]["fixture"]["partitions"]) == [
        "other side"
    ]
    assert ModelCard(selected.document).select().files == selected.files
    assert len(card.choices()) == 18
    with pytest.raises(ManifestError, match="select explicitly"):
        card.select(variant="fixture", service="chat", programs="source")
    with pytest.raises(ManifestError, match="unknown reference"):
        card.select("check", partition="missing")
    document["variants"]["fixture"]["services"] = services
    with pytest.raises(ManifestError, match="exactly one"):
        ModelCard(document)


def test_partition_choices_require_names_and_supported_shape():
    with pytest.raises(ManifestError, match="unpartitioned"):
        ModelCard(_document()).select("check", partition="node-0")
    for value in ({}, [], None, {"node-0": {"services": {}}}):
        document = _document()
        document["variants"]["fixture"] = {"partitions": value}
        with pytest.raises(ManifestError):
            ModelCard(document)


def test_native_selected_closure():
    selected = ModelCard(_document()).select("check")
    assert selected.service == {"kind": "text", "version": 1}
    assert selected.config == {"fixture.card": True, "fixture.bias_delta": 1}
    assert selected.settings["execution"]["rows"] == 4
    assert set(selected.files) == {
        "prepare.loom",
        "control.loom",
        "service.loom",
        "storage.loom",
        "advance.loom",
        "weights.loom",
        "metadata.safetensors",
        "bias.data",
        "tokenizer.json",
    }
    assert selected.files["bias.data"].format == "safetensors"
    assert selected.files["weights.loom"].roles == {"auxiliary"}
    assert selected.parameters["model"]["files"][0] == "metadata.safetensors"
    assert ModelCard(selected.document).select().files == selected.files


def test_shared_providers_keep_distinct_service_selections():
    card = ModelCard(_document())
    with pytest.raises(ManifestError, match="select explicitly"):
        card.select()
    text = card.select("check")
    sequence = card.select("encode")
    assert sequence.service == {"kind": "sequence", "version": 1}
    assert sequence.program == text.program
    assert sequence.parameters == text.parameters
    assert sequence.config == {"fixture.card": True, "fixture.bias_delta": 0}
    assert sequence.tokenizers == {}
    assert set(sequence.files) == set(text.files) - {"tokenizer.json"}
    assert ModelCard(sequence.document).select().service == sequence.service


def test_source_and_bytecode_names_are_opaque():
    document = _document()
    document["programs"]["textual"] = document["programs"].pop("archive")
    for service in document["variants"]["fixture"]["services"].values():
        service["programs"] = ["source", "textual", "mixed"]
    selected = ModelCard(document).select("check", programs="textual")
    assert "service_module.loombc" in selected.files
    assert "prepare.loom" not in selected.files
    mixed = ModelCard(document).select("check", programs="mixed")
    assert "prepare.loom" in mixed.files and "control_module.loombc" in mixed.files


def test_moonshine_metadata_is_not_a_qualification_claim():
    document = json.loads(
        (Path(__file__).parent / "testdata/moonshine.json").read_text()
    )
    selected = ModelCard(document).select("streaming")
    assert selected.model_id == "moonshine-ai/moonshine-streaming-tiny"
    assert selected.service["kind"] == "recognition"
    assert len(selected.files) == 19
    assert (
        selected.files["asset/model.safetensors"].origin
        == "@checkpoint/model.safetensors"
    )
    assert selected.files["asset/text-vocabulary.data"].format == "huggingface-json"
    assert "asset/config.json" not in selected.files


def test_origins_separate_acquisition_from_placement():
    document = _document()
    document["origins"] = {"checkpoint": {"kind": "hf", "repository": "owner/model"}}
    parameters = _service(document)["parameters"]["model"]
    parameters["origin"] = "@checkpoint/"
    parameters["files"][1]["origin"] = "@checkpoint/some/stupid/name.tensor"
    selected = ModelCard(document).select("check")
    assert (
        selected.files["metadata.safetensors"].origin
        == "@checkpoint/metadata.safetensors"
    )
    assert selected.files["bias.data"].origin == "@checkpoint/some/stupid/name.tensor"
    assert selected.files["bias.data"].path == "bias.data"


def test_owned_snapshots():
    document = _document()
    card = ModelCard(document)
    selected = card.select("check")
    document.clear()
    selected.settings["execution"]["rows"] = 77
    selected.document["programs"].clear()
    assert card.select("check").settings["execution"]["rows"] == 4
    assert card.select("check").document["programs"]


def test_independent_service_versions():
    document = _document()
    document["services"]["future"] = {"kind": "image", "version": 54}
    _service(document)["programs"] = ["source"]
    assert ModelCard(document).select("check").service["version"] == 1
    document["services"]["chat"].pop("version")
    assert "version" not in ModelCard(document).select("check").service


def test_explicit_selection_without_canonical_preset():
    document = _document()
    document.pop("presets")
    card = ModelCard(document)
    with pytest.raises(ManifestError, match="select explicitly"):
        card.select()
    assert card.select(service="chat", programs="mixed").programs == "mixed"


def test_ambiguous_presets_require_selection_even_with_identical_files():
    document = _document()
    settings = document["presets"]["check"]
    document["presets"] = {
        "small": settings,
        "large": settings | {"execution": {"rows": 8}},
    }
    card = ModelCard(document)
    with pytest.raises(ManifestError, match="preset: select explicitly"):
        card.select()
    assert card.select("large").settings["execution"]["rows"] == 8


def test_programs_must_be_supported_by_variant():
    document = _document()
    _service(document)["programs"] = ["source"]
    with pytest.raises(ManifestError, match="unknown reference"):
        ModelCard(document).select("check", programs="archive")


def test_same_destination_combines_roles_but_not_conflicting_sources():
    document = _document()
    parameter = _service(document)["parameters"]["model"]
    parameter["files"].append("weights.loom")
    card = ModelCard(document)
    assert card.select("check").files["weights.loom"].roles == {
        "auxiliary",
        "parameter",
    }
    parameter["files"][-1] = {"path": "weights.loom", "origin": "other.loom"}
    with pytest.raises(ManifestError, match="conflicting selected"):
        ModelCard(document)


def test_options_preserve_types_but_require_declared_names():
    document = _document()
    document["services"]["chat"]["options"] = ["voice"]
    document["presets"]["check"]["options"] = {"voice": {"tokens": [1, 2]}}
    selected = ModelCard(document).select("check")
    assert selected.settings["options"]["voice"]["tokens"] == [1, 2]
    document["presets"]["check"]["options"]["layers"] = 17
    with pytest.raises(ManifestError, match="undeclared model options"):
        ModelCard(document)


@pytest.mark.parametrize(
    "source",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":1,"schema_versi\\u006fn":1}',
        '{"schema_version":NaN}',
        '{"schema_version":1,}',
        '{/* comment */"schema_version":1}',
        '{"value":1e400}',
    ],
)
def test_rejects_nonportable_json(source):
    with pytest.raises(ManifestError):
        ModelCard.parse(source)


@pytest.mark.parametrize(
    "path", ["/root/a", "../a", "a/../b", "./a", "a//b", "a/", "C:/a", "a\\b", "a\0b"]
)
def test_rejects_unsafe_destinations(path):
    document = _document()
    document["programs"]["source"]["inputs"][0]["path"] = path
    with pytest.raises(ManifestError):
        ModelCard(document)


@pytest.mark.parametrize("origin", ["../shared.loom", "../../../program/shared.loom"])
@pytest.mark.parametrize("root", [None, "local", "url"])
def test_sibling_sources_preserve_contained_destination(origin, root):
    document = _document()
    provider = document["programs"]["source"]["inputs"][0]
    destination = provider["path"]
    if root:
        document.setdefault("origins", {})["document"] = (
            {"kind": "local", "path": "/checkout/model/publisher/example"}
            if root == "local"
            else {
                "kind": "url",
                "url": "https://example.org/snapshot/model/publisher/example/",
            }
        )
        origin = "@document/" + origin
    provider["origin"] = origin
    selected = ModelCard(document).select("check")
    assert selected.files[destination].origin == origin
    assert selected.files[destination].path == destination


def test_sibling_group_origin_appends_destination_once():
    document = _document()
    document["programs"]["source"]["origin"] = "../../../"
    selected = ModelCard(document).select("check")
    for provider in selected.program["inputs"]:
        destination = provider["path"] if isinstance(provider, dict) else provider
        assert selected.files[destination].origin == "../../../" + destination


@pytest.mark.parametrize(
    "origin",
    ["../", "..", ".././code.loom", "../a/../b", "../a//b", "../C:/a", "../a\\b"],
)
def test_sibling_sources_still_require_normalized_file_names(origin):
    document = _document()
    document["programs"]["source"]["inputs"][0]["origin"] = origin
    with pytest.raises(ManifestError):
        ModelCard(document)


def test_hugging_face_member_cannot_ascend_out_of_repository():
    document = _document()
    document.setdefault("origins", {})["checkpoint"] = {
        "kind": "hf",
        "repository": "publisher/model",
    }
    provider = document["programs"]["source"]["inputs"][0]
    for origin in (
        "@checkpoint/../other/code.loom",
        {"kind": "hf", "repository": "publisher/model", "path": "../other/code.loom"},
    ):
        provider["origin"] = origin
        with pytest.raises(ManifestError):
            ModelCard(document)


def test_rejects_file_directory_collision():
    document = _document()
    _service(document)["parameters"]["model"]["files"].append("bias.data/nested")
    with pytest.raises(ManifestError, match="collides"):
        ModelCard(document)


def test_rejects_duplicate_provider_role():
    document = _document()
    document["programs"]["source"]["libraries"].append("prepare.loom")
    with pytest.raises(ManifestError, match="duplicate provider"):
        ModelCard(document)


@pytest.mark.parametrize("value", [None, [], {"nested": 1}])
def test_jit_configuration_is_scalar(value):
    document = _document()
    _service(document)["config"]["invalid"] = value
    with pytest.raises(ManifestError, match="scalar"):
        ModelCard(document)


def test_new_schema_dispatch_precedes_structure():
    document = {"schema_version": 7, "unknown_future_shape": True}
    with pytest.raises(ManifestError, match="unsupported model-card version"):
        ModelCard(document)


def test_inline_and_root_validation():
    document = _document()
    target = document["programs"]["source"]["inputs"][0]
    for origin in (
        {"kind": "inline", "base64": "!"},
        "@unknown/source",
        "https://user:password@example.org/file",
        "https://@example.org/file",
        "https://example.org:invalid/file",
        "https://example.org/line\nbreak",
    ):
        target["origin"] = origin
        with pytest.raises(ManifestError):
            ModelCard(document)
    document = _document()
    document["programs"]["source"]["origin"] = "https://example.org/?path=/"
    with pytest.raises(ManifestError, match="query"):
        ModelCard(document)


def test_projection_excludes_other_selections():
    document = _document()
    document["variants"]["unused"] = copy.deepcopy(document["variants"]["fixture"])
    selected = ModelCard(document).select("check")
    assert set(selected.document["programs"]) == {"source"}
    assert set(selected.document["services"]) == {"chat"}
    assert set(selected.document["variants"]) == {"fixture"}


def test_original_voice_recordings_are_selected_deduplicated_and_projected():
    document = _document()
    recording = {
        "file": {
            "path": "voice/recording.data",
            "origin": "https://example.com/original.wav",
        }
    }
    _service(document)["voices"] = {
        "first": recording,
        "second": copy.deepcopy(recording),
    }
    selected = ModelCard(document).select("check")
    assert selected.files["voice/recording.data"].roles == frozenset({"voice"})
    assert (
        selected.files["voice/recording.data"].origin
        == "https://example.com/original.wav"
    )
    assert ModelCard(selected.document).select().voices == selected.voices
    selected.voices["first"]["file"]["path"] = "changed.data"
    assert (
        ModelCard(document).select("check").voices["first"]["file"]["path"]
        == "voice/recording.data"
    )


@pytest.mark.parametrize(
    "voice",
    [
        "recording.wav",
        {},
        {"file": "../recording.wav"},
        {"file": {"path": "recording.wav", "format": "wav"}},
        {"file": "recording.data", "content_type": ""},
        {"file": "recording.data", "prepared": True},
    ],
)
def test_voice_recordings_require_one_explicit_role_and_format_contract(voice):
    document = _document()
    _service(document)["voices"] = {"narrator": voice}
    with pytest.raises(ManifestError):
        ModelCard(document)
