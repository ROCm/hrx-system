# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Published Loom profiles: cards, measurements, chat policy and requirements.

A profile merges the documents a Loom publisher ships: the portable model card
(`model.json`, the runtime contract), the measurement record (`info.json`,
schema `loom-card/2`), the generator's configuration (`llm_config.json`, which
owns the chat policy), and the human README. Each document is read tolerantly: a
publisher may add fields, and an older card may omit a whole section, without
making the profile unusable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loom.launch.errors import ProfileError
from loom.launch.hub import Hub, HubModel
from loom.launch.values import (
    bool_value,
    dict_value,
    int_value,
    list_value,
    number_value,
    str_value,
)

CARD_DOCUMENT = "model.json"
MEASUREMENT_DOCUMENT = "info.json"
CONFIG_DOCUMENT = "llm_config.json"
README_DOCUMENT = "README.md"

#: Metrics a Loom measurement record publishes, in publishing order.
METRIC_ORDER = (
    "decode",
    "decode_2k",
    "decode_32k",
    "prefill_1k",
    "prefill_2k",
    "prefill_8k",
    "prefill_16k",
    "prefill_32k",
    "streams4",
    "spec_acceptance",
)

METRIC_LABELS = {
    "decode": "decode, short prompt",
    "decode_2k": "decode, 2k context",
    "decode_32k": "decode, 32k context",
    "prefill_1k": "prefill, 1k document",
    "prefill_2k": "prefill, 2k document",
    "prefill_8k": "prefill, 8k document",
    "prefill_16k": "prefill, 16k document",
    "prefill_32k": "prefill, 32k document",
    "streams4": "decode, 4 streams",
    "spec_acceptance": "draft acceptance",
}

#: Metrics measured as a fraction rather than a rate.
FRACTION_METRICS = frozenset({"spec_acceptance"})

_QUANTIZATION = re.compile(
    r"\b(IQ[1-4]_[A-Z0-9]{2,4}|Q[2-8]_K_[SML]|Q[2-8]_K|Q[2-8]_[01]|Q[2-8]"
    r"|BF16|F16|F32|MXFP4|MXFP8)\b"
)
_GFX_ARCH = re.compile(r"\bgfx[0-9a-f]{3,6}\b")


@dataclass(frozen=True)
class Feature:
    """One capability the publisher's evidence established."""

    name: str
    status: str = "unknown"
    detail: dict = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.status == "yes"

    def describe(self) -> dict:
        return {"status": self.status, "detail": dict(self.detail)}


@dataclass(frozen=True)
class Gate:
    """One correctness gate the publisher ran against the package."""

    name: str
    status: str = "unknown"
    level: str | None = None
    date: str | None = None
    source: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def failed(self) -> bool:
        return self.status in ("fail", "failed", "error")

    def describe(self) -> dict:
        return {
            "status": self.status,
            "level": self.level,
            "date": self.date,
            "source": self.source,
        }


@dataclass(frozen=True)
class Measurement:
    """One measured metric: this engine, the baseline and the hardware ceiling."""

    metric: str
    value: float | None = None
    baseline: float | None = None
    acceptance: float | None = None
    ceiling: float | None = None
    date: str | None = None

    @property
    def label(self) -> str:
        return METRIC_LABELS.get(self.metric, self.metric)

    @property
    def unit(self) -> str:
        return "fraction" if self.metric in FRACTION_METRICS else "tok/s"

    @property
    def roofline_fraction(self) -> float | None:
        if self.value is None or not self.ceiling:
            return None
        return self.value / self.ceiling

    @property
    def baseline_ratio(self) -> float | None:
        if self.value is None or not self.baseline:
            return None
        return self.value / self.baseline

    def describe(self) -> dict:
        return {
            "metric": self.metric,
            "label": self.label,
            "unit": self.unit,
            "value": self.value,
            "baseline": self.baseline,
            "acceptance": self.acceptance,
            "ceiling": self.ceiling,
            "roofline_fraction": self.roofline_fraction,
            "baseline_ratio": self.baseline_ratio,
            "date": self.date,
        }


@dataclass(frozen=True)
class Performance:
    """The measurements a card publishes, with their baseline and roofline."""

    measurements: tuple[Measurement, ...] = ()
    baseline_name: str | None = None
    baseline_version: str | None = None
    baseline_args: str | None = None
    device: str | None = None
    #: Other engines the publisher compared against: name -> metric -> value.
    extra_baselines: dict[str, dict[str, float]] = field(default_factory=dict)
    session_count: int = 0

    @classmethod
    def from_measurement_document(cls, info: dict) -> Performance | None:
        """Reads `info.json`; the last session that measured a metric wins."""
        performance = dict_value(info.get("performance"))
        sessions = [
            dict_value(item) for item in list_value(performance.get("sessions"))
        ]
        roofline = dict_value(info.get("roofline"))
        roofline_metrics = dict_value(roofline.get("metrics"))
        if not sessions and not roofline_metrics:
            return None
        latest: dict[str, dict] = {}
        dates: dict[str, str | None] = {}
        for session in sessions:
            date = str_value(session.get("date"))
            for name, entry in dict_value(session.get("metrics")).items():
                entry = dict_value(entry)
                if number_value(entry.get("loom")) is None:
                    continue
                latest[name] = entry
                dates[name] = date
        ordered = [name for name in METRIC_ORDER if name in latest]
        ordered += [name for name in sorted(latest) if name not in METRIC_ORDER]
        measurements = tuple(
            Measurement(
                metric=name,
                value=number_value(latest[name].get("loom")),
                baseline=number_value(latest[name].get("baseline")),
                acceptance=number_value(latest[name].get("acceptance")),
                ceiling=number_value(
                    dict_value(roofline_metrics.get(name)).get("ceiling")
                ),
                date=dates.get(name),
            )
            for name in ordered
        )
        extra: dict[str, dict[str, float]] = {}
        for name, entry in dict_value(performance.get("baselines")).items():
            session = dict_value(dict_value(entry).get("session"))
            values: dict[str, float] = {}
            for metric, row in dict_value(session.get("metrics")).items():
                value = number_value(dict_value(row).get("baseline"))
                if value is not None:
                    values[metric] = value
            if values:
                extra[name] = values
        baseline = dict_value(performance.get("baseline"))
        device = dict_value(roofline.get("device"))
        return cls(
            measurements=measurements,
            baseline_name=str_value(baseline.get("name")),
            baseline_version=str_value(baseline.get("version")),
            baseline_args=str_value(baseline.get("args")),
            device=str_value(device.get("name")),
            extra_baselines=extra,
            session_count=len(sessions),
        )

    def by_metric(self) -> dict[str, Measurement]:
        return {measurement.metric: measurement for measurement in self.measurements}

    def measurement(self, metric: str) -> Measurement | None:
        return self.by_metric().get(metric)

    def has_measurements(self) -> bool:
        return any(measurement.value is not None for measurement in self.measurements)


@dataclass(frozen=True)
class Execution:
    """The capacities a card's preset declares for its service."""

    preset: str | None = None
    service: str | None = None
    service_kind: str | None = None
    rows: int | None = None
    context_capacity: int | None = None
    pool_capacity: int | None = None
    prefill_capacity: int | None = None
    max_tokens: int | None = None
    draft_depth: int | None = None

    @classmethod
    def from_card(
        cls, model_card: dict, preferred: str | None = None
    ) -> Execution | None:
        presets = dict_value(model_card.get("presets"))
        declared = preferred or str_value(model_card.get("preset"))
        selected = _choose_preset(presets, declared)
        if selected is None:
            return None
        entry = dict_value(presets.get(selected))
        execution = dict_value(entry.get("execution"))
        service_name = str_value(entry.get("service"))
        services = dict_value(model_card.get("services"))
        service = dict_value(services.get(service_name)) if service_name else {}
        return cls(
            preset=selected,
            service=service_name,
            service_kind=str_value(service.get("kind")),
            rows=int_value(execution.get("rows")),
            context_capacity=int_value(execution.get("context_capacity")),
            pool_capacity=int_value(execution.get("pool_capacity")),
            prefill_capacity=int_value(execution.get("prefill_capacity")),
            max_tokens=int_value(execution.get("max_tokens")),
            draft_depth=int_value(execution.get("draft_depth")),
        )

    def describe(self) -> dict:
        return {
            "preset": self.preset,
            "service": self.service,
            "service_kind": self.service_kind,
            "rows": self.rows,
            "context_capacity": self.context_capacity,
            "pool_capacity": self.pool_capacity,
            "prefill_capacity": self.prefill_capacity,
            "max_tokens": self.max_tokens,
            "draft_depth": self.draft_depth,
        }


@dataclass(frozen=True)
class ChatPolicy:
    """The chat rendering policy the package's generator emitted.

    The served model renders prompts in its own programs; this policy is what
    the generator recorded, which lets the launcher preview a rendered prompt,
    count its tokens and explain which template controls a package consumes.
    """

    policy: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, llm_config: dict) -> ChatPolicy | None:
        chat = dict_value(llm_config.get("chat"))
        return cls(policy=chat) if chat else None

    def _field(self, name: str, default: str = "") -> str:
        return str_value(self.policy.get(name)) or default

    @property
    def thinking_mode(self) -> str:
        return self._field("thinking", "never")

    @property
    def supports_thinking(self) -> bool:
        return self.thinking_mode != "never" and bool(self._field("think_open"))

    @property
    def default_thinking(self) -> bool:
        value = bool_value(self.policy.get("thinking_default"))
        return bool(value)

    @property
    def terminal(self) -> str:
        return self._field("terminal")

    @property
    def tool_call_format(self) -> str | None:
        return str_value(self.policy.get("call_format"))

    @property
    def default_system(self) -> str:
        return self._field("default_system")

    def describe(self) -> dict:
        return {
            "thinking": self.thinking_mode,
            "supports_thinking": self.supports_thinking,
            "default_thinking": self.default_thinking,
            "terminal": self.terminal,
            "tool_calls": self.tool_call_format,
            "default_system": self.default_system,
        }

    def render(
        self,
        messages: list[dict],
        *,
        thinking: bool | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Renders a conversation the way this package frames prompts."""
        if thinking is None:
            thinking = self.default_thinking
        system_text: str | None = None
        turns: list[tuple[str, str]] = []
        for message in messages:
            message = dict_value(message)
            role = str_value(message.get("role")) or "user"
            text = _message_text(message.get("content"))
            if role == "system" and system_text is None:
                system_text = text
                continue
            turns.append((role, text))
        if system_text is None:
            system_text = self.default_system
        parts: list[str] = []
        if system_text:
            parts.append(self._system_text(system_text))
        if tools:
            parts.append(self._tools_section(tools))
        for role, text in turns:
            parts.append(self._turn_text(role, text))
        parts.append(self._generation_prompt(thinking))
        return "".join(parts)

    def _system_text(self, text: str) -> str:
        return self._field("system_prefix") + text + self._field("suffix")

    def _turn_text(self, role: str, text: str) -> str:
        if role == "assistant":
            return (
                self._field("assistant_prefix")
                + self._history_text(text)
                + self._field("suffix")
            )
        if role == "tool":
            return (
                self._field("tool_turn_prefix")
                + self._field("tool_item_prefix")
                + text
                + self._field("tool_item_suffix")
                + self._field("tool_turn_suffix")
            )
        if role == "system":
            return self._system_text(text)
        return self._field("user_prefix") + text + self._field("suffix")

    def _history_text(self, text: str) -> str:
        open_text = self._field("think_open")
        close_text = self._field("think_close")
        if self._field("history_think") != "never" or not open_text or not close_text:
            return text
        pattern = re.escape(open_text) + ".*?" + re.escape(close_text)
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def _tools_section(self, tools: list[dict]) -> str:
        prefix = self._field("tools_prefix")
        if not prefix:
            return ""
        entries = []
        for tool in tools:
            tool = dict_value(tool)
            entries.append(tool.get("function") or tool)
        body = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        return (
            prefix
            + body
            + self._field("tools_suffix")
            + self._field("tools_system_separator")
        )

    def _generation_prompt(self, thinking: bool) -> str:
        if thinking:
            return self._field("generation_prompt", self._field("assistant_prefix"))
        return self._field(
            "generation_prompt_plain",
            self._field("generation_prompt", self._field("assistant_prefix")),
        )


@dataclass(frozen=True)
class Profile:
    """One published Loom package, seen through its own documents."""

    repo: str
    model_id: str
    name: str
    revision: str | None = None
    #: Local installation directory, when the profile came from one.
    path: str | None = None
    tags: tuple[str, ...] = ()
    license: str | None = None
    quant: str | None = None
    architecture: str | None = None
    #: The context the card's preset serves (the headline figure).
    context_length: int | None = None
    #: The model's own context ceiling, which a launch may not exceed.
    native_context_length: int | None = None
    memory_bytes: int | None = None
    device_hint: str | None = None
    gfx_arch: str | None = None
    features: dict[str, Feature] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    performance: Performance | None = None
    execution: Execution | None = None
    chat: ChatPolicy | None = None
    drafter: dict = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_documents(
        cls,
        repo: str,
        model_card: dict,
        *,
        info: dict | None = None,
        llm_config: dict | None = None,
        readme: str | None = None,
        revision: str | None = None,
        tags: tuple[str, ...] = (),
        path: str | None = None,
    ) -> Profile:
        model_card = dict_value(model_card)
        info = dict_value(info)
        llm_config = dict_value(llm_config)
        model = dict_value(model_card.get("model"))
        card = dict_value(info.get("card"))
        features = {
            name: Feature(
                name=name,
                status=str_value(dict_value(entry).get("status")) or "unknown",
                detail=dict_value(dict_value(entry).get("value")),
            )
            for name, entry in dict_value(card.get("features")).items()
        }
        gates = {}
        for name, entry in dict_value(info.get("gates")).items():
            entry = dict_value(entry)
            gates[name] = Gate(
                name=name,
                status=str_value(entry.get("status")) or "unknown",
                level=str_value(entry.get("level")),
                date=str_value(entry.get("date")),
                source=str_value(entry.get("source")),
            )
        execution = Execution.from_card(model_card)
        performance = Performance.from_measurement_document(info)
        memory_bytes = int_value(card.get("memory_bytes"))
        if memory_bytes is None:
            memory_bytes = _preset_memory(model_card, execution)
        context_feature = features.get("context")
        context_length, native_context_length = _served_and_native_context(
            int_value(llm_config.get("context_length")),
            execution.context_capacity if execution is not None else None,
            dict_value(context_feature.detail) if context_feature else {},
        )
        device_hint = performance.device if performance else None
        return cls(
            repo=repo,
            model_id=str_value(model.get("id")) or repo,
            name=str_value(model.get("name")) or model_id_of(repo),
            revision=revision,
            path=path,
            tags=tags,
            license=_license(tags),
            quant=_quantization(
                str_value(llm_config.get("gguf")),
                str_value(dict_value(info.get("roofline")).get("checkpoint")),
                str_value(model.get("name")),
                _readme_title(readme),
                repo,
            ),
            architecture=str_value(llm_config.get("arch")),
            context_length=context_length,
            native_context_length=native_context_length,
            memory_bytes=memory_bytes,
            device_hint=device_hint,
            gfx_arch=_gfx_arch(" ".join(filter(None, [device_hint, repo]))),
            features=features,
            gates=gates,
            performance=performance,
            execution=execution,
            chat=ChatPolicy.from_config(llm_config),
            drafter=dict_value(llm_config.get("drafter")),
        )

    @classmethod
    def from_hub(
        cls,
        hub: Hub,
        repo: str,
        *,
        revision: str | None = None,
        model: HubModel | None = None,
    ) -> Profile:
        """Reads a published profile, pinning every document to one revision."""
        model = model or hub.model(repo)
        revision = revision or model.revision
        model_card = hub.json_optional(repo, CARD_DOCUMENT, revision)
        if not model_card:
            raise ProfileError(
                f"{repo} does not publish a root {CARD_DOCUMENT}; "
                "pass an explicit model.json path"
            )
        info = hub.json_optional(repo, MEASUREMENT_DOCUMENT, revision)
        llm_config = hub.json_optional(repo, CONFIG_DOCUMENT, revision)
        readme = hub.raw_optional(repo, README_DOCUMENT, revision).decode(
            "utf-8", "replace"
        )
        return cls.from_documents(
            repo,
            model_card,
            info=info,
            llm_config=llm_config,
            readme=readme,
            revision=revision,
            tags=model.tags,
        )

    @classmethod
    def from_local(cls, directory: str | Path) -> Profile:
        """Reads an installed package; sidecar documents are optional."""
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        card_path = directory / CARD_DOCUMENT
        if not card_path.is_file():
            raise ProfileError(f"{card_path} is not an installed model card")
        try:
            model_card = json.loads(card_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"{card_path} is unreadable: {exc}") from exc
        repo = str_value(dict_value(model_card.get("model")).get("id")) or (
            directory.name
        )
        return cls.from_documents(
            repo,
            model_card,
            info=_optional_json(directory / MEASUREMENT_DOCUMENT),
            llm_config=_optional_json(directory / CONFIG_DOCUMENT),
            readme=_optional_text(directory / README_DOCUMENT),
            path=str(directory),
        )

    # -- views --------------------------------------------------------------
    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}"

    @property
    def slug(self) -> str:
        return self.repo.replace("/", "-")

    def feature(self, name: str) -> Feature | None:
        return self.features.get(name)

    def enabled(self, name: str) -> bool:
        feature = self.features.get(name)
        return bool(feature and feature.enabled)

    def passed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, gate in self.gates.items() if gate.passed))

    def failed_gates(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, gate in self.gates.items() if gate.failed))

    def render(
        self,
        messages: list[dict],
        *,
        thinking: bool | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        if self.chat is None:
            raise ProfileError(f"{self.repo} publishes no chat policy")
        return self.chat.render(messages, thinking=thinking, tools=tools)

    def benchmark_rows(self) -> list[dict]:
        if self.performance is None:
            return []
        return [
            measurement.describe()
            for measurement in self.performance.measurements
            if measurement.value is not None
        ]

    def describe(self) -> dict:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "model_id": self.model_id,
            "name": self.name,
            "path": self.path,
            "url": self.url,
            "tags": list(self.tags),
            "license": self.license,
            "quant": self.quant,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "native_context_length": self.native_context_length,
            "memory_bytes": self.memory_bytes,
            "device": self.device_hint,
            "gfx_arch": self.gfx_arch,
            "features": {
                name: feature.describe() for name, feature in self.features.items()
            },
            "gates": {name: gate.describe() for name, gate in self.gates.items()},
            "execution": self.execution.describe() if self.execution else None,
            "chat": self.chat.describe() if self.chat else None,
            "drafter": dict(self.drafter),
            "benchmarks": self.benchmark_rows(),
            "performance": (
                {
                    "baseline": self.performance.baseline_name,
                    "baseline_version": self.performance.baseline_version,
                    "device": self.performance.device,
                    "sessions": self.performance.session_count,
                }
                if self.performance
                else None
            ),
        }

    @property
    def context_text(self) -> str:
        """`40960 tokens (model native 262144)`, or `262144 tokens` when the
        preset serves the whole native context."""
        served = self.context_length or None
        native = self.native_context_length or None
        if served and native and served != native:
            return f"{served} tokens (model native {native})"
        if served or native:
            return f"{served or native} tokens"
        return "unknown"

    def summary_line(self) -> str:
        fields = [self.repo]
        if self.quant:
            fields.append(self.quant)
        if self.context_length:
            fields.append(f"ctx {self.context_length}")
        if self.performance is not None:
            decode = self.performance.measurement("decode")
            if decode is not None and decode.value is not None:
                fields.append(f"decode {decode.value:.1f} tok/s")
        return "  ".join(fields)


def _served_and_native_context(
    config_context: int | None, preset_context: int | None, feature: dict
) -> tuple[int | None, int | None]:
    """The context a profile serves and the model's own ceiling.

    The served context is the preset's `context_capacity`, else the measured
    `context.tokens` feature, else the native context. The native context is
    the generator's `context_length`, else the `context.native` feature, else
    the served context: a launch may narrow the served context, and may widen
    it up to the native one.
    """
    native = config_context or int_value(feature.get("native")) or None
    served = preset_context or int_value(feature.get("tokens")) or native
    return served, native or served


def model_id_of(repo: str) -> str:
    """The trailing repository name, used when a card omits a model name."""
    return repo.split("/", 1)[-1]


def _choose_preset(presets: dict, preferred: str | None) -> str | None:
    if not presets:
        return None
    if preferred and preferred in presets:
        return preferred
    if len(presets) == 1:
        return next(iter(presets))
    for name in ("chat", "default", "interactive", "serve"):
        if name in presets:
            return name
    return sorted(presets)[0]


def _preset_memory(model_card: dict, execution: Execution | None) -> int | None:
    presets = dict_value(model_card.get("presets"))
    if execution is None or execution.preset is None:
        return None
    memory = dict_value(dict_value(presets.get(execution.preset)).get("memory"))
    return int_value(memory.get("weights_bytes"))


def _license(tags: tuple[str, ...]) -> str | None:
    for tag in tags:
        if tag.startswith("license:"):
            return tag[len("license:") :]
    return None


def _quantization(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        match = _QUANTIZATION.search(text)
        if match:
            return match.group(0)
    return None


def _gfx_arch(text: str) -> str | None:
    if not text:
        return None
    match = _GFX_ARCH.search(text)
    return match.group(0) if match else None


def _readme_title(readme: str | None) -> str | None:
    if not readme:
        return None
    for line in readme.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def _message_text(content: object) -> str:
    text = str_value(content)
    if text is not None:
        return text
    parts = []
    for part in list_value(content):
        part = dict_value(part)
        if str_value(part.get("type")) == "text":
            parts.append(str_value(part.get("text")) or "")
    return "".join(parts)


def _optional_json(path: Path) -> dict:
    text = _optional_text(path)
    if not text:
        return {}
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return document if isinstance(document, dict) else {}


def _optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
