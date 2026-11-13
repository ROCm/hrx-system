# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json

import pytest

from loom.launch.deployment import context_settings, deployment_document
from loom.launch.errors import LaunchError
from loom.launch.hardware import detect, evaluate
from loom.launch.profile import Profile

CHAT = {
    "system_prefix": "<|im_start|>system\n",
    "user_prefix": "<|im_start|>user\n",
    "assistant_prefix": "<|im_start|>assistant\n",
    "suffix": "<|im_end|>\n",
    "default_system": "",
    "think_open": "<think>",
    "think_close": "</think>",
    "history_think": "never",
    "generation_prompt": "<|im_start|>assistant\n",
    "generation_prompt_plain": "<|im_start|>assistant\n",
    "thinking": "toggle",
    "thinking_default": True,
    "terminal": "<|im_end|>",
    "call_format": "hermes",
    "tools_prefix": "TOOLS:",
    "tools_suffix": ":END",
    "tools_system_separator": "\n",
}


def card(execution=None, presets=None):
    execution = execution or {
        "rows": 4,
        "context_capacity": 262144,
        "pool_capacity": 327680,
        "prefill_capacity": 1024,
        "max_tokens": 4096,
        "draft_depth": 3,
    }
    return {
        "schema_version": 1,
        "model": {"id": "Qwen/Qwen3-4B", "name": "Qwen3-4B (Loom port, GGUF)"},
        "services": {"chat": {"kind": "text", "version": 1}},
        "presets": presets
        or {
            "chat": {
                "service": "chat",
                "variant": "gguf",
                "programs": "source",
                "execution": execution,
            }
        },
    }


def measurements():
    return {
        "schema": "loom-card/2",
        "card": {
            "memory_bytes": 63000000000,
            "features": {
                "context": {"status": "yes", "value": {"native": 262144}},
                "tool_calling": {"status": "yes", "value": None},
                "audio": {"status": "n/a", "value": None},
            },
        },
        "gates": {
            "compare": {"status": "pass", "level": "full", "date": "2026-11-08"},
            "vision": {"status": "n/a"},
        },
        "performance": {
            "baseline": {"name": "llama.cpp", "version": "b1368", "args": "-ngl 99"},
            "sessions": [
                {"date": "2026-11-08", "metrics": {"decode": {"loom": 70.0}}},
                {
                    "date": "2026-11-11",
                    "metrics": {
                        "decode": {"loom": 76.1, "baseline": 51.7},
                        "prefill_1k": {"loom": 2891.8},
                        "spec_acceptance": {"loom": 0.44},
                    },
                },
            ],
            "baselines": {
                "vulkan": {"session": {"metrics": {"decode": {"baseline": 72.6}}}}
            },
        },
        "roofline": {
            "device": {
                "name": 'AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151)',
                "bandwidth_bytes_per_s": 216000000000.0,
                "peak_flops": 59000000000000.0,
            },
            "metrics": {
                "decode": {"ceiling": 123.6},
                "prefill_1k": {"ceiling": 7802.6},
            },
        },
    }


def config():
    return {
        "arch": "qwen3",
        "gguf": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "context_length": 262144,
        "chat": dict(CHAT),
        "drafter": {"kind": "dflash", "depth": 3},
    }


def profile(**overrides):
    arguments = {
        "repo": "loom-fun/qwen3-4b-loom",
        "model_card": card(),
        "info": measurements(),
        "llm_config": config(),
        "readme": "# Qwen3-4B (Q4_K_M) for AMD Loom\n",
        "tags": ("loom", "license:apache-2.0"),
        "revision": "abc123",
    }
    arguments.update(overrides)
    return Profile.from_documents(
        arguments.pop("repo"), arguments.pop("model_card"), **arguments
    )


def test_profile_reads_identity_capacities_and_features():
    parsed = profile()
    assert parsed.model_id == "Qwen/Qwen3-4B"
    assert parsed.quant == "Q4_K_M"
    assert parsed.architecture == "qwen3"
    assert parsed.license == "apache-2.0"
    assert parsed.context_length == 262144
    assert parsed.memory_bytes == 63000000000
    assert parsed.gfx_arch == "gfx1151"
    assert parsed.enabled("context")
    assert parsed.enabled("tool_calling")
    assert not parsed.enabled("audio")
    assert parsed.execution.preset == "chat"
    assert parsed.execution.service_kind == "text"
    assert parsed.execution.max_tokens == 4096
    assert parsed.passed_gates() == ("compare",)
    assert parsed.failed_gates() == ()
    assert parsed.drafter["kind"] == "dflash"


def test_measurements_take_the_last_session_that_measured_each_metric():
    performance = profile().performance
    assert [row.metric for row in performance.measurements] == [
        "decode",
        "prefill_1k",
        "spec_acceptance",
    ]
    decode = performance.measurement("decode")
    assert decode.value == 76.1
    assert decode.baseline == 51.7
    assert decode.date == "2026-11-11"
    assert decode.roofline_fraction is not None
    assert abs(decode.roofline_fraction - 76.1 / 123.6) < 1e-9
    assert performance.measurement("spec_acceptance").unit == "fraction"
    assert performance.extra_baselines == {"vulkan": {"decode": 72.6}}
    assert performance.baseline_name == "llama.cpp"


def test_an_older_card_without_measurements_still_reads():
    parsed = Profile.from_documents("loom-fun/x-Q4_K_M-loom", card())
    assert parsed.quant == "Q4_K_M"
    assert parsed.context_length == 262144
    assert parsed.features == {}
    assert parsed.gates == {}
    assert parsed.performance is None
    assert parsed.chat is None
    assert parsed.memory_bytes is None
    assert parsed.benchmark_rows() == []


def test_chat_policy_renders_system_user_and_generation_prompt():
    parsed = profile()
    rendered = parsed.render(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "2+2?"},
        ],
        thinking=False,
    )
    assert rendered == (
        "<|im_start|>system\nbe terse<|im_end|>\n"
        "<|im_start|>user\n2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_chat_policy_defaults_to_the_package_thinking_mode():
    parsed = profile()
    assert parsed.chat.supports_thinking
    assert parsed.chat.default_thinking
    assert parsed.render([{"role": "user", "content": "hi"}]).endswith(
        "<|im_start|>assistant\n"
    )
    assert parsed.chat.thinking_mode == "toggle"


def test_chat_policy_strips_reasoning_from_history_when_it_says_never():
    parsed = profile()
    rendered = parsed.render(
        [
            {
                "role": "assistant",
                "content": "x<think>hidden</think>y",
            },
            {"role": "user", "content": "hi"},
        ],
        thinking=False,
    )
    assert "<think>" not in rendered
    assert "x y" not in rendered
    assert "xy<|im_end|>" in rendered


def test_chat_policy_renders_tool_declarations_after_the_system_turn():
    parsed = profile()
    rendered = parsed.render(
        [{"role": "user", "content": "hi"}],
        thinking=False,
        tools=[{"type": "function", "function": {"name": "read"}}],
    )
    assert 'TOOLS:[{"name":"read"}]:END' in rendered


def test_chat_policy_renders_tool_results_as_a_tool_turn():
    parsed = profile()
    parsed.chat.policy.update(
        {
            "tool_turn_prefix": "<|im_start|>user",
            "tool_item_prefix": "\n<tool_response>\n",
            "tool_item_suffix": "\n</tool_response>",
            "tool_turn_suffix": "<|im_end|>\n",
        }
    )
    rendered = parsed.render([{"role": "tool", "content": "42"}], thinking=False)
    assert "<tool_response>" in rendered
    assert "42" in rendered
    assert "</tool_response>" in rendered


def test_execution_prefers_a_named_chat_preset():
    presets = {
        "alpha": {"service": "chat", "execution": {"context_capacity": 4096}},
        "chat": {"service": "chat", "execution": {"context_capacity": 8192}},
    }
    parsed = profile(model_card=card(presets=presets))
    assert parsed.execution.preset == "chat"
    assert parsed.execution.context_capacity == 8192


def test_context_settings_default_to_the_card():
    settings = context_settings(profile())
    assert settings.context_length == 262144
    assert settings.max_tokens == 4096
    assert settings.rows == 4
    assert settings.max_sessions is None
    assert settings.draft_depth == 3
    definition = settings.definition()
    assert definition["kind"] == "text"
    assert definition["context_capacity"] == 262144
    assert definition["max_tokens"] == 4096
    assert "session_cache_capacity" not in definition


def test_narrower_context_and_output_are_accepted():
    settings = context_settings(profile(), context_length=8192, max_tokens=512)
    assert (settings.context_length, settings.max_tokens) == (8192, 512)


def test_context_beyond_the_model_is_rejected():
    with pytest.raises(LaunchError) as error:
        context_settings(profile(), context_length=300000)
    assert "exceeds the model's 262144 tokens" in str(error.value)


def test_output_limit_must_leave_room_in_the_context():
    with pytest.raises(LaunchError) as error:
        context_settings(profile(), context_length=100, max_tokens=100)
    assert "leaves no room" in str(error.value)


def test_parallel_sets_rows_and_sessions_together():
    settings = context_settings(profile(), parallel=2)
    assert (settings.rows, settings.max_sessions) == (2, 2)


def test_deployment_document_references_the_card_and_preset():
    parsed = profile()
    settings = context_settings(parsed, context_length=8192)
    document = deployment_document(parsed, "/models/qwen3/model.json", settings)
    assert document["name"] == "loom-fun-qwen3-4b-loom"
    assert document["model"] == {
        "path": "/models/qwen3/model.json",
        "preset": "chat",
    }
    assert document["definition"]["context_capacity"] == 8192


ROCMINFO = "Agent 1\n  Name: gfx1151\n  Marketing Name: AMD Radeon 8060S Graphics\n"


def detected():
    return detect(
        runner=lambda argv: ROCMINFO,
        files={
            "/proc/meminfo": "MemTotal: 131072000 kB\n",
            "/proc/cpuinfo": "model name\t: AMD Ryzen\n",
        },
        system="Linux",
        machine="x86_64",
    )


def test_detection_reads_memory_cpu_and_the_gpu_arch():
    hardware = detected()
    assert hardware.total_memory_bytes == 131072000 * 1024
    assert hardware.cpu == "AMD Ryzen"
    assert hardware.gfx_arch() == "gfx1151"
    assert hardware.gpu().name == "AMD Radeon 8060S Graphics"
    assert hardware.default_device_uri() == "amdgpu://"
    assert hardware.npu is False


def test_a_cpu_only_machine_defaults_to_the_task_device():
    hardware = detect(
        runner=lambda argv: None,
        files={"/proc/meminfo": "MemTotal: 1000 kB\n"},
    )
    assert hardware.devices == ()
    assert hardware.default_device_uri() == "task"


def test_compatibility_rewards_fit_and_a_matching_device():
    hardware = detected()
    good = evaluate(hardware, profile())
    assert good.fits_memory is True
    assert good.device_match is True
    assert good.score >= 0.7
    # The card records a gate that is not applicable, which is not a pass.
    assert "gates not passing: vision" in " ".join(good.notes)
    large = profile(info={**measurements(), "card": {"memory_bytes": 200000000000}})
    bad = evaluate(hardware, large)
    assert bad.fits_memory is False
    assert bad.score < good.score
    assert "does not fit" in " ".join(bad.notes)


def test_local_profile_reads_an_installed_card(tmp_path):
    directory = tmp_path / "qwen3"
    directory.mkdir()
    (directory / "model.json").write_text(json.dumps(card()), encoding="utf-8")
    parsed = Profile.from_local(directory)
    assert parsed.repo == "Qwen/Qwen3-4B"
    assert parsed.path == str(directory)
    assert parsed.context_length == 262144
    assert parsed.chat is None


def hybrid():
    """The Qwen3.8-27B GPU/NPU hybrid: its preset serves 40960 of 262144 tokens."""
    return Profile.from_documents(
        "qwen3.8-27b",
        card(
            {
                "rows": 1,
                "context_capacity": 40960,
                "prefill_capacity": 1024,
                "max_tokens": 4096,
                "draft_depth": 0,
            }
        ),
        info={
            "card": {
                "memory_bytes": 52000000000,
                "features": {
                    "context": {
                        "status": "yes",
                        "value": {"native": 262144, "tokens": 40960},
                    }
                },
            }
        },
        llm_config={"arch": "qwen35", "context_length": 262144},
    )


def test_the_served_context_is_the_preset_and_the_native_one_is_the_ceiling():
    parsed = hybrid()
    assert parsed.context_length == 40960
    assert parsed.native_context_length == 262144
    assert parsed.context_text == "40960 tokens (model native 262144)"
    assert parsed.describe()["native_context_length"] == 262144
    assert "ctx 40960" in parsed.summary_line()
    assert profile().context_text == "262144 tokens"
    settings = context_settings(parsed)
    assert (settings.context_length, settings.max_tokens) == (40960, 4096)
    assert context_settings(parsed, context_length=50000).context_length == 50000
    with pytest.raises(LaunchError, match="exceeds the model's 262144 tokens"):
        context_settings(parsed, context_length=262145)


def test_without_a_preset_the_measured_tokens_then_the_native_context_serve():
    measured = Profile.from_documents(
        "x/y",
        {"model": {"id": "x/y"}},
        info={
            "card": {
                "features": {
                    "context": {
                        "status": "yes",
                        "value": {"native": 262144, "tokens": 32768},
                    }
                }
            }
        },
    )
    assert (measured.context_length, measured.native_context_length) == (32768, 262144)
    bare = Profile.from_documents("x/y", {"model": {"id": "x/y"}})
    assert (bare.context_length, bare.native_context_length) == (None, None)
    assert bare.context_text == "unknown"
