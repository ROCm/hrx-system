# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The `loom-launch` command line.

Six verbs cover the launcher's work: `search` finds published Loom profiles,
`install` acquires one, `show` explains one, `serve` runs one behind an
instrumented OpenAI endpoint, `chat` talks to one, and `bench` measures one.
`status`, `logs`, `stop`, `hardware` and `doctor` observe and diagnose.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loom.launch import bench as benchmarks
from loom.launch.chat import ChatSession, describe_turn, one_shot, run_repl
from loom.launch.client import Client
from loom.launch.deployment import context_settings
from loom.launch.errors import LauncherError, ProfileError
from loom.launch.hardware import Compatibility, detect, evaluate
from loom.launch.hub import Hub, HubModel
from loom.launch.install import (
    InstallResult,
    default_models_directory,
    fetch_command,
    install_and_record,
)
from loom.launch.profile import (
    CARD_DOCUMENT,
    Profile,
)
from loom.launch.scenario import Scenario
from loom.launch.server import (
    LaunchOptions,
    LaunchPlan,
    Supervisor,
    describe_state,
    plan_launch,
    plan_scenario,
    read_current,
    running_endpoint,
    serve_command,
)
from loom.launch.store import (
    BenchmarkStore,
    InstalledModel,
    InstallRegistry,
    ProfileCache,
    RunStore,
    state_directory,
)
from loom.launch.values import human_bytes, human_rate, str_value

DEFAULT_PORT = 8080
DEFAULT_LIMIT = 20


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 2
    try:
        return args.handler(args)
    except LauncherError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130


# -- argument surface ------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-launch",
        description=(
            "Search, install, serve and benchmark published Loom profiles behind "
            "an instrumented OpenAI-compatible endpoint."
        ),
    )
    parser.add_argument(
        "--state",
        help="launcher state directory (default $LOOM_LAUNCH_STATE or "
        "~/.local/state/loom-launch)",
    )
    parser.add_argument(
        "--hub",
        help="model hub endpoint (default $LOOM_HUB_ENDPOINT or "
        "https://huggingface.co)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="find published Loom profiles")
    search.add_argument("query", nargs="?", help="name fragment, or an owner prefix")
    search.add_argument("--author", help="restrict to one hub owner")
    search.add_argument(
        "--tag",
        action="append",
        default=[],
        help="require an additional hub tag (repeatable)",
    )
    search.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT, help="maximum profiles to show"
    )
    search.add_argument(
        "--no-filter",
        action="store_true",
        help="list every match, not only installable profiles",
    )
    search.add_argument(
        "--no-details",
        action="store_true",
        help="skip reading each profile's documents (quantization, context, benchmarks)",
    )
    search.add_argument(
        "--offline", action="store_true", help="use cached profile documents only"
    )
    search.set_defaults(handler=_cmd_search)

    show = commands.add_parser("show", help="explain a profile")
    show.add_argument("target", help="owner/name, or a path to an installed card")
    show.add_argument(
        "--prompt",
        help="render this user prompt through the package's chat policy",
    )
    show.add_argument("--system", help="system prompt for --prompt")
    show.add_argument(
        "--thinking", choices=("on", "off"), help="thinking control for --prompt"
    )
    show.add_argument("--raw", action="store_true", help="print the profile document")
    show.set_defaults(handler=_cmd_show)

    install = commands.add_parser("install", help="install a published profile")
    install.add_argument("target", help="owner/name, or a hub URL")
    install.add_argument("--output", help="installation directory")
    install.add_argument("--preset", help="named card preset to install")
    install.add_argument(
        "--materialize",
        choices=("auto", "copy", "share"),
        default="copy",
        help="how installs treat programs and weights",
    )
    install.add_argument("--revision", help="pin a branch, tag or commit")
    install.add_argument("--loom-fetch", help="path to loom-fetch")
    install.add_argument(
        "--offline",
        action="store_true",
        help="forbid network requests: loom-fetch runs --offline, and a profile's "
        "deployment.json/devices.json come only from next to the card or the "
        "local hub cache",
    )
    install.set_defaults(handler=_cmd_install)

    serve = commands.add_parser("serve", help="serve a profile over OpenAI endpoints")
    _add_profile_arguments(serve)
    _add_launch_arguments(serve)
    serve.set_defaults(handler=_cmd_serve)

    chat = commands.add_parser("chat", help="talk to a served profile")
    _add_profile_arguments(chat)
    _add_endpoint_arguments(chat)
    chat.add_argument("--prompt", help="send one prompt and exit")
    chat.add_argument("--system", help="system prompt")
    chat.add_argument("--max-tokens", type=int, help="output token limit per turn")
    chat.add_argument("--temperature", type=float, help="sampling temperature")
    chat.add_argument(
        "--think",
        choices=("on", "off", "auto"),
        default="auto",
        help="request or suppress the package's reasoning block",
    )
    chat.add_argument("--session", help="named retained session (X-Loom-Session)")
    chat.add_argument(
        "--template-kwargs",
        help="JSON object passed to the package's chat policy (chat_template_kwargs)",
    )
    chat.add_argument("--save", help="write the transcript here on exit")
    chat.set_defaults(handler=_cmd_chat)

    bench = commands.add_parser("bench", help="benchmark a profile")
    _add_profile_arguments(bench)
    _add_endpoint_arguments(bench)
    bench.add_argument(
        "--suite",
        choices=tuple(benchmarks.SUITES),
        default="standard",
        help="metric set to measure",
    )
    bench.add_argument(
        "--repeats", type=int, default=1, help="samples per metric (median reported)"
    )
    bench.add_argument(
        "--think",
        choices=("on", "off", "auto"),
        default="off",
        help="thinking control during measurement",
    )
    bench.add_argument("--list", action="store_true", help="list stored runs")
    bench.add_argument("--show", help="print one stored run by id")
    bench.add_argument("--no-store", action="store_true", help="do not keep the result")
    bench.set_defaults(handler=_cmd_bench)

    status = commands.add_parser("status", help="report the running launch")
    status.set_defaults(handler=_cmd_status)

    logs = commands.add_parser("logs", help="show a launch's server log")
    logs.add_argument("--lines", type=int, default=40, help="lines to show")
    logs.add_argument("--run", help="run id (default: the latest run)")
    logs.add_argument("--follow", action="store_true", help="follow the log")
    logs.set_defaults(handler=_cmd_logs)

    stop = commands.add_parser("stop", help="stop the running launch")
    stop.set_defaults(handler=_cmd_stop)

    hardware = commands.add_parser("hardware", help="describe this machine")
    hardware.set_defaults(handler=_cmd_hardware)

    doctor = commands.add_parser("doctor", help="check the launcher's environment")
    doctor.set_defaults(handler=_cmd_doctor)
    return parser


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        nargs="?",
        help="owner/name, an installed card path, or nothing for the last install",
    )
    parser.add_argument("--revision", help="pin a branch, tag or commit")
    parser.add_argument("--preset", help="card preset to select")


def _add_launch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context", type=int, help="context length in tokens")
    parser.add_argument("--max-tokens", type=int, help="default output token limit")
    parser.add_argument(
        "--parallel", type=int, help="resident sessions (rows and max_sessions)"
    )
    parser.add_argument(
        "--draft-depth", type=int, help="speculative draft depth, 0 to disable"
    )
    parser.add_argument("--checkpoint-capacity", type=int, help="named checkpoints")
    parser.add_argument("--session-cache", type=int, help="retained idle histories")
    parser.add_argument("--prefix-cache", type=int, help="retained input prefixes")
    parser.add_argument("--prefill-capacity", type=int, help="packed prefill shape")
    parser.add_argument("--device", help="HAL device URI (default: detected)")
    parser.add_argument("--loom-serve", help="path to the native server")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    parser.add_argument("--workers", type=int, help="server worker threads")
    parser.add_argument("--pool-backing", help="fixed or elastic pool backing")
    parser.add_argument("--memory-bytes", type=int, help="pool memory budget")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="serve the native endpoint without the instrumentation proxy",
    )
    parser.add_argument(
        "--proxy-port", type=int, default=0, help="instrumentation proxy port"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOOM_LAUNCH_API_KEY"),
        help="require this bearer token from clients",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1800.0,
        help="seconds to wait for model preparation",
    )
    parser.add_argument(
        "--detach", action="store_true", help="run the server in the background"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="refuse to acquire a profile that is not installed",
    )


def _add_endpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        help="talk to an existing endpoint instead of launching one",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOOM_LAUNCH_API_KEY"),
        help="bearer token for the endpoint",
    )
    parser.add_argument(
        "--request-timeout", type=float, default=900.0, help="per-request timeout"
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="leave a launch started for this command running afterwards",
    )
    _launch_defaults(parser)


def _launch_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--context", type=int, help="context length in tokens")
    parser.add_argument("--parallel", type=int, help="resident sessions")
    parser.add_argument("--device", help="HAL device URI")
    parser.add_argument("--loom-serve", help="path to the native server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    parser.add_argument("--workers", type=int, help="server worker threads")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="serve the native endpoint without the instrumentation proxy",
    )
    parser.add_argument(
        "--wait", type=float, default=1800.0, help="seconds to wait for preparation"
    )


# -- shared resolution -----------------------------------------------------
@dataclass
class Resolved:
    """A profile plus the local card a launch can serve."""

    profile: Profile
    installed: InstalledModel | None = None

    @property
    def card_path(self) -> Path:
        if self.installed is None:
            raise ProfileError(f"{self.profile.repo} is not installed")
        return Path(self.installed.path) / CARD_DOCUMENT


def _state_root(args: argparse.Namespace) -> Path:
    return state_directory(getattr(args, "state", None))


def _hub(args: argparse.Namespace) -> Hub:
    return Hub(endpoint=getattr(args, "hub", None))


def _profile_from_repo(
    args: argparse.Namespace, repo: str, revision: str | None
) -> Profile:
    """Reads a profile from the hub, falling back to cached documents."""
    root = _state_root(args)
    cache = ProfileCache(root)
    offline = bool(getattr(args, "offline", False))
    if not offline:
        try:
            profile = Profile.from_hub(_hub(args), repo, revision=revision)
            cache.put(repo, _profile_documents(_hub(args), repo, profile.revision))
            return profile
        except LauncherError:
            raise
        except Exception as exc:  # Network failures must fall back, not abort.
            cached = cache.get(repo)
            if cached is None:
                raise LauncherError(f"cannot read {repo}: {exc}") from exc
    cached = cache.get(repo)
    if cached is None:
        raise LauncherError(
            f"{repo} is not cached; run without --offline to read it from the hub"
        )
    return Profile.from_documents(
        repo,
        cached.get("model_card") or {},
        info=cached.get("info") or {},
        llm_config=cached.get("llm_config") or {},
        readme=cached.get("readme"),
        revision=revision,
        tags=tuple(cached.get("tags") or ()),
    )


def _profile_documents(hub: Hub, repo: str, revision: str | None) -> dict:
    return {
        "model_card": hub.json_optional(repo, "model.json", revision),
        "info": hub.json_optional(repo, "info.json", revision),
        "llm_config": hub.json_optional(repo, "llm_config.json", revision),
        "readme": hub.raw_optional(repo, "README.md", revision).decode(
            "utf-8", "replace"
        ),
        "tags": [],
    }


def _local_target(target: str) -> Path | None:
    candidate = Path(target).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.is_dir() and (candidate / CARD_DOCUMENT).is_file():
        return candidate
    return None


def _resolve(args: argparse.Namespace, *, install: bool) -> Resolved:
    """Resolves the command's target to a profile and an installed card."""
    root = _state_root(args)
    registry = InstallRegistry(root)
    target = getattr(args, "target", None)
    revision = getattr(args, "revision", None)
    preset = getattr(args, "preset", None)
    if target is None:
        installed = _latest_install(registry)
        if installed is None:
            raise LauncherError(
                "no target given and nothing installed; "
                "run `loom-launch search`, then `loom-launch install OWNER/NAME`"
            )
        return Resolved(_local_profile(args, installed), installed)
    local = _local_target(target)
    if local is not None:
        profile = Profile.from_local(local)
        directory = local if local.is_dir() else local.parent
        installed = registry.find_by_path(directory) or InstalledModel(
            repo=profile.repo, path=str(directory)
        )
        return Resolved(profile, installed)
    if "/" not in target:
        raise LauncherError(
            f"{target} is neither an installed card nor an owner/name profile"
        )
    installed = registry.get(target)
    if installed is not None and installed.card_path.is_file():
        return Resolved(_local_profile(args, installed), installed)
    if not install:
        raise LauncherError(
            f"{target} is not installed; run `loom-launch install {target}`"
        )
    result, model = _install_target(args, target, preset=preset)
    profile = _enrich(args, _local_profile(args, model), revision)
    return Resolved(profile, model)


def _latest_install(registry: InstallRegistry) -> InstalledModel | None:
    models = [model for model in registry.list() if model.card_path.is_file()]
    if not models:
        return None
    models.sort(key=lambda model: model.installed_at or "")
    return models[-1]


def _local_profile(args: argparse.Namespace, installed: InstalledModel) -> Profile:
    return _enrich(args, Profile.from_local(installed.card_path), None)


def _enrich(
    args: argparse.Namespace, profile: Profile, revision: str | None
) -> Profile:
    """Adds the hub's documents (features, measurements, chat policy) locally."""
    remote = _ProfileEnricher(args, profile.repo, revision or profile.revision)
    return remote.apply(profile)


class _ProfileEnricher:
    def __init__(self, args: argparse.Namespace, repo: str, revision: str | None):
        self.args = args
        self.repo = repo
        self.revision = revision

    def apply(self, profile: Profile) -> Profile:
        root = _state_root(self.args)
        cache = ProfileCache(root)
        documents = cache.get(self.repo)
        if documents is None and not getattr(self.args, "offline", False):
            try:
                documents = _profile_documents(
                    _hub(self.args), self.repo, self.revision
                )
                cache.put(self.repo, documents)
            except Exception:
                documents = None
        if not documents:
            return profile
        enriched = Profile.from_documents(
            self.repo,
            documents.get("model_card") or {},
            info=documents.get("info") or {},
            llm_config=documents.get("llm_config") or {},
            readme=documents.get("readme"),
            revision=self.revision,
            tags=profile.tags,
            path=profile.path,
        )
        return enriched


def _install_target(
    args: argparse.Namespace, repo: str, preset: str | None = None
) -> tuple[InstallResult, InstalledModel]:
    root = _state_root(args)
    output = getattr(args, "output", None) or (
        default_models_directory(root) / repo.replace("/", "-")
    )
    materialize = getattr(args, "materialize", "copy")
    progress = None if getattr(args, "json", False) else _stderr_progress

    def on_line(line: str) -> None:
        if progress is not None:
            progress(line)

    result, model = install_and_record(
        repo,
        output=output,
        preset=preset,
        revision=getattr(args, "revision", None),
        materialize=materialize,
        offline=getattr(args, "offline", False),
        explicit_fetch=getattr(args, "loom_fetch", None),
        state_root=root,
        on_line=on_line,
        hub=_hub(args),
    )
    if result.documents.warning:
        sys.stderr.write(f"warning: {result.documents.warning}\n")
    return result, model


def _stderr_progress(line: str) -> None:
    sys.stderr.write(f"  fetch: {line}\n")
    sys.stderr.flush()


# -- search ---------------------------------------------------------------
def _cmd_search(args: argparse.Namespace) -> int:
    hub = _hub(args)
    tags = list(args.tag)
    if args.no_filter:
        found = hub.search(args.query, author=args.author, tags=tags, limit=args.limit)
    else:
        found = hub.profiles(
            args.query,
            author=args.author,
            limit=args.limit,
            tags=("loom-serve", *tags),
        )
    hardware = detect()
    rows = []
    with_details = not args.no_details
    for model in found:
        row = {"model": model, "profile": None, "compat": None}
        if with_details:
            try:
                profile = _profile_from_repo(args, model.repo, model.revision)
                row["profile"] = profile
                row["compat"] = evaluate(hardware, profile)
            except LauncherError as exc:
                row["error"] = str(exc)
        rows.append(row)
    rows.sort(key=_search_order)
    if args.json:
        document = {
            "query": args.query,
            "count": len(rows),
            "hardware": hardware.describe(),
            "profiles": [
                {
                    **row["model"].describe(),
                    "profile": row["profile"].describe() if row["profile"] else None,
                    "compatibility": row["compat"].describe()
                    if row["compat"]
                    else None,
                    "error": row.get("error"),
                }
                for row in rows
            ],
        }
        print(json.dumps(document, indent=1))
        return 0
    if not rows:
        print("no matching profiles")
        return 1
    _print_search_table(rows)
    print()
    print(
        f"{len(rows)} profile(s); `loom-launch install OWNER/NAME` acquires one, "
        "`loom-launch show OWNER/NAME` explains it"
    )
    return 0


def _search_order(row: dict) -> tuple:
    compat = row.get("compat")
    score = compat.score if isinstance(compat, Compatibility) else 0.0
    return (-score, row["model"].repo)


def _print_search_table(rows: list[dict]) -> None:
    header = f"{'profile':<52} {'quant':<9} {'ctx':>8} {'memory':>9} {'decode':>9} {'fit':>5}"
    print(header)
    print("-" * len(header))
    for row in rows:
        model: HubModel = row["model"]
        profile: Profile | None = row["profile"]
        compat: Compatibility | None = row["compat"]
        decode = None
        if profile is not None and profile.performance is not None:
            measurement = profile.performance.measurement("decode")
            decode = measurement.value if measurement else None
        print(
            f"{model.repo:<52} "
            f"{(profile.quant if profile and profile.quant else '-'):<9} "
            f"{(str(profile.context_length) if profile and profile.context_length else '-'):>8} "
            f"{(human_bytes(profile.memory_bytes) if profile else '-'):>9} "
            f"{(f'{decode:,.1f}' if decode else '-'):>9} "
            f"{(f'{compat.score:.2f}' if compat else '-'):>5}"
        )
        if row.get("error"):
            print(f"  ! {row['error']}")


# -- show -----------------------------------------------------------------
def _cmd_show(args: argparse.Namespace) -> int:
    local = _local_target(args.target) if args.target else None
    installed = InstallRegistry(_state_root(args)).get(args.target)
    revision = getattr(args, "revision", None)
    if local is not None:
        profile = Profile.from_local(local)
        if not getattr(args, "offline", False):
            profile = _enrich(args, profile, None)
    elif (
        installed is not None
        and installed.card_path.is_file()
        and revision in (None, installed.revision)
    ):
        # An installed profile is explained from what was installed, its
        # scenario documents included.
        profile = _local_profile(args, installed)
    else:
        profile = _profile_from_repo(args, args.target, revision)
    scenario = _profile_scenario(profile)
    hardware = detect()
    compat = evaluate(hardware, profile)
    if args.raw:
        print(json.dumps(profile.describe(), indent=1))
        return 0
    if args.json:
        document = profile.describe()
        document["compatibility"] = compat.describe()
        document["scenario"] = scenario.describe() if scenario else None
        print(json.dumps(document, indent=1))
        return 0
    _print_profile(profile, compat, scenario)
    if args.prompt:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        thinking = None
        if args.thinking:
            thinking = args.thinking == "on"
        try:
            rendered = profile.render(messages, thinking=thinking)
        except ProfileError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        print("rendered prompt")
        print("-" * 62)
        print(rendered)
    return 0


def _profile_scenario(profile: Profile) -> Scenario | None:
    """The scenario a local profile's directory carries."""
    if not profile.path:
        return None
    try:
        return Scenario.detect(profile.path)
    except LauncherError:
        return None


def _print_profile(
    profile: Profile, compat: Compatibility, scenario: Scenario | None = None
) -> None:
    print(profile.repo + (f" @ {profile.revision}" if profile.revision else ""))
    print(f"  model        {profile.model_id}  ({profile.name})")
    print(f"  quant        {profile.quant or 'unknown'}")
    print(f"  architecture {profile.architecture or 'unknown'}")
    print(f"  context      {profile.context_text}")
    print(f"  memory       {human_bytes(profile.memory_bytes)}")
    if profile.device_hint:
        print(f"  measured on  {profile.device_hint}")
    if profile.execution is not None:
        execution = profile.execution
        print(
            "  preset       "
            f"{execution.preset or '-'}  rows {execution.rows or '-'}  "
            f"prefill {execution.prefill_capacity or '-'}  "
            f"max_tokens {execution.max_tokens or '-'}  "
            f"draft {execution.draft_depth if execution.draft_depth is not None else '-'}"
        )
    if profile.chat is not None:
        chat = profile.chat
        print(
            "  chat         "
            f"thinking {chat.thinking_mode}"
            f"{' (default on)' if chat.default_thinking else ''}  "
            f"tools {chat.tool_call_format or '-'}  terminal {chat.terminal or '-'}"
        )
    if profile.features:
        enabled = sorted(
            name for name, feature in profile.features.items() if feature.enabled
        )
        print(f"  features     {', '.join(enabled) if enabled else 'none recorded'}")
    if profile.gates:
        print(
            f"  gates        {len(profile.passed_gates())} passed, "
            f"{len(profile.failed_gates())} not passing"
        )
    if scenario is not None:
        print(
            f"  scenario     deployment {scenario.name} on {scenario.device_text} "
            "(deployment.json + devices.json)"
        )
    print(f"  fit          {compat.score:.2f}  " + "; ".join(compat.notes))
    if profile.performance is not None:
        print()
        print("published measurements")
        print("-" * 62)
        for line in benchmarks.format_published(profile):
            print(line)
        performance = profile.performance
        if performance.baseline_name:
            print(
                f"baseline     {performance.baseline_name} "
                f"{performance.baseline_version or ''}".rstrip()
            )
        if performance.extra_baselines:
            print(
                "also compared with " + ", ".join(sorted(performance.extra_baselines))
            )
    print()
    print(_next_step(profile))


def _next_step(profile: Profile) -> str:
    """What to run next: a local package is served by its path, a hub profile
    is installed by its repository."""
    if profile.path:
        return f"serve with: loom-launch serve {os.path.abspath(profile.path)}"
    return f"install with: loom-launch install {profile.repo}"


# -- install --------------------------------------------------------------
def _cmd_install(args: argparse.Namespace) -> int:
    result, model = _install_target(args, args.target, preset=args.preset)
    if args.json:
        print(
            json.dumps(
                {"result": result.describe(), "install": model.describe()}, indent=1
            )
        )
        return 0
    print(f"installed {model.repo} into {model.path}")
    print(f"  card      {model.card_path}")
    print(f"  size      {human_bytes(model.size_bytes)}")
    if result.preset:
        print(f"  preset    {result.preset}")
    if result.revision:
        print(f"  revision  {result.revision}")
    summary = result.documents.summary()
    if summary:
        print(f"  scenario  {summary}")
    print(f"  elapsed   {result.seconds:.1f}s")
    print()
    print(f"serve it with: loom-launch serve {model.repo}")
    return 0


# -- serve ----------------------------------------------------------------
def _cmd_serve(args: argparse.Namespace) -> int:
    root = _state_root(args)
    if args.detach:
        return _detach(sys.argv[1:], root, args.wait)
    resolved = _resolve(args, install=not args.no_install)
    prepared = _prepare(args, resolved, SERVE_CAPACITY_OPTIONS)
    options = LaunchOptions(
        device=args.device,
        host=args.host,
        port=args.port,
        workers=args.workers,
        pool_backing=args.pool_backing,
        memory_bytes=args.memory_bytes,
        proxy=not args.no_proxy,
        proxy_port=args.proxy_port,
        auth_token=args.api_key,
        startup_timeout=args.wait,
        loom_serve=args.loom_serve,
    )
    run = RunStore(root).create(
        metadata={"repo": resolved.profile.repo, "source": "serve"}
    )
    plan = _plan(prepared, resolved, run, options)
    settings = plan.settings
    supervisor = Supervisor(
        plan,
        options,
        on_output=_plugin_output(args),
        state_root=root,
    )
    print(f"launching {resolved.profile.repo}")
    print(f"  run       {run.root}")
    print(f"  deployment {plan.deployment['name']}  device {plan.device_uri}")
    if plan.scenario is not None:
        print(
            "  scenario  running the package's own deployment "
            f"{plan.scenario.name} across {plan.scenario.device_text}"
        )
    print(
        f"  context   {settings.context_length} tokens, "
        f"max {settings.max_tokens} output"
        + (f", {settings.rows} rows" if settings.rows else "")
    )
    supervisor.start()
    client = supervisor.client()
    print(f"  endpoint  {supervisor.endpoint}")
    if options.proxy:
        print(f"  server    {supervisor.server_endpoint} (instrumented by the proxy)")
    print(f"  prepared in {run.metadata.get('prepare_seconds', 0):.1f}s; ^C stops it")
    _signal_termination()
    exit_code = supervisor.wait()
    if args.json:
        print(json.dumps({"run": run.describe(), "exit_code": exit_code}, indent=1))
    else:
        metrics = supervisor.metrics.snapshot()
        print(
            f"stopped: {metrics['requests']} request(s), "
            f"{metrics['completion_tokens']} completion tokens, "
            f"ttft p50 {metrics['ttft_seconds']['p50'] or '-'}s"
        )
        print(f"log: {run.serve_log}")
    del client
    return exit_code


#: Launcher options that set capacities or devices a scenario declares itself:
#: (argument name, flag, the `context_settings` keyword it feeds).
SERVE_CAPACITY_OPTIONS = (
    ("context", "--context", "context_length"),
    ("max_tokens", "--max-tokens", "max_tokens"),
    ("parallel", "--parallel", "parallel"),
    ("draft_depth", "--draft-depth", "draft_depth"),
    ("checkpoint_capacity", "--checkpoint-capacity", "checkpoint_capacity"),
    ("session_cache", "--session-cache", "session_cache_capacity"),
    ("prefix_cache", "--prefix-cache", "prefix_cache_capacity"),
    ("prefill_capacity", "--prefill-capacity", "prefill_capacity"),
)
#: What a launch started for `chat` or `bench` takes from the command line
#: (their `--max-tokens` is a per-turn output limit, not a capacity).
ENDPOINT_CAPACITY_OPTIONS = (
    ("context", "--context", "context_length"),
    ("parallel", "--parallel", "parallel"),
)


def _prepare(
    args: argparse.Namespace,
    resolved: Resolved,
    capacity_options: tuple[tuple[str, str, str], ...],
):
    """What a launch of `resolved` runs: its prepared scenario, or the
    capacities composed from the profile and the request.

    Options that contradict a scenario are refused before anything is started
    or written.
    """
    scenario = Scenario.detect(resolved.installed.path)
    if scenario is not None:
        given = [
            flag
            for name, flag, _keyword in capacity_options
            if getattr(args, name, None) is not None
        ]
        if getattr(args, "device", None):
            given.append("--device")
        scenario.refuse(given)
        return scenario
    return context_settings(
        resolved.profile,
        **{
            keyword: getattr(args, name, None)
            for name, _flag, keyword in capacity_options
        },
    )


def _plan(prepared, resolved: Resolved, run, options: LaunchOptions) -> LaunchPlan:
    """Plans the launch `_prepare` resolved."""
    if isinstance(prepared, Scenario):
        return plan_scenario(resolved.profile, prepared, run, options)
    return plan_launch(resolved.profile, resolved.card_path, prepared, run, options)


def _plugin_output(args: argparse.Namespace):
    if getattr(args, "quiet_output", False) or getattr(args, "json", False):
        return None
    return lambda line: (sys.stdout.write(f"  | {line}\n"), sys.stdout.flush())


def _detach(argv: list[str], root: Path, wait: float) -> int:
    """Runs `serve` in the background, then reports its endpoint."""
    tail = [argument for argument in argv if argument != "--detach"]
    command = [sys.executable, "-m", "loom.launch", *tail]
    log_directory = root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "detached.log"
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if process.poll() is not None:
            sys.stderr.write(f"error: the launch exited; see {log_path}\n")
            return 1
        record = read_current(root)
        if record and record.get("endpoint"):
            print(f"serving {record.get('repo')} at {record['endpoint']}")
            print(f"  deployment {record.get('deployment')}")
            print(f"  server     {record.get('server_endpoint')} (native listener)")
            print(f"  log        {log_path}")
            return 0
        time.sleep(0.2)
    # The child runs in its own session, so nothing else would stop it: end the
    # whole group rather than leave a server holding a device and a port.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except OSError:
        pass
    sys.stderr.write(f"error: the launch did not become ready; see {log_path}\n")
    return 1


# -- chat -----------------------------------------------------------------
def _cmd_chat(args: argparse.Namespace) -> int:
    target = _endpoint_for(args)
    endpoint, profile = target.url, target.profile
    thinking = None
    if args.think == "on":
        thinking = True
    elif args.think == "off":
        thinking = False
    session = ChatSession(
        Client(endpoint, token=args.api_key, timeout=args.request_timeout),
        profile,
        model=target.model,
        thinking=thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system=args.system,
        session=args.session,
        context_length=target.context_length,
        template_kwargs=_template_kwargs(args),
    )
    if args.prompt:
        result = one_shot(session, args.prompt)
        if args.json:
            print(json.dumps(describe_turn(result), indent=1))
        if args.save:
            _save_transcript(session, args.save)
        return 0 if not result.error else 1
    print(f"chatting with {profile.repo} at {endpoint}")
    print("type /help for commands, /exit or Ctrl-D to leave")
    code = run_repl(session)
    if args.save:
        _save_transcript(session, args.save)
    return code


def _template_kwargs(args: argparse.Namespace) -> dict:
    """Reads the JSON template controls a caller passed, or nothing."""
    text = getattr(args, "template_kwargs", None)
    if not text:
        return {}
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LauncherError(f"--template-kwargs is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise LauncherError("--template-kwargs must be a JSON object")
    return document


def _save_transcript(session: ChatSession, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(session.transcript(), file, indent=1)
    sys.stderr.write(f"transcript written to {path}\n")


# -- bench ----------------------------------------------------------------
def _cmd_bench(args: argparse.Namespace) -> int:
    root = _state_root(args)
    store = BenchmarkStore(root)
    if args.list:
        return _list_runs(store, args.json)
    if args.show:
        record = store.load(args.show)
        if record is None:
            sys.stderr.write(f"error: no stored run {args.show}\n")
            return 1
        return _print_record(record, args.json)
    resolved_endpoint = _endpoint_for(args)
    endpoint = resolved_endpoint.url
    profile = resolved_endpoint.profile
    supervisor = resolved_endpoint.supervisor
    runner = benchmarks.Runner(
        Client(endpoint, token=args.api_key, timeout=args.request_timeout),
        profile,
        model=resolved_endpoint.model,
        repeats=args.repeats,
        thinking=None if args.think == "auto" else args.think == "on",
        on_progress=None if args.json else _bench_progress,
        max_sessions=_deployment_sessions(resolved_endpoint, args),
    )
    print(f"benchmarking {profile.repo} ({args.suite} suite) at {endpoint}")
    record = runner.run(args.suite)
    record["endpoint"] = endpoint
    if not args.no_store:
        path = store.save(record)
        record["path"] = str(path)
    if supervisor is not None and not args.keep_server:
        supervisor.stop()
    if args.json:
        print(json.dumps(record, indent=1))
        return 0 if record.get("ok") else 1
    for line in benchmarks.format_metrics(record):
        print(line)
    if record.get("path"):
        print(f"\nstored: {record['path']}")
    return 0 if record.get("ok") else 1


def _deployment_sessions(endpoint: "Endpoint", args: argparse.Namespace) -> int | None:
    """How many sessions the deployment admits, for comparability notes."""
    if args.parallel is not None:
        return args.parallel
    return endpoint.profile.execution.rows if endpoint.profile.execution else None


def _bench_progress(message: str) -> None:
    sys.stdout.write(f"  {message}\n")
    sys.stdout.flush()


def _list_runs(store: BenchmarkStore, as_json: bool) -> int:
    records = store.list()
    if as_json:
        print(json.dumps({"runs": records}, indent=1))
        return 0
    if not records:
        print("no stored benchmark runs")
        return 0
    print(f"{'id':<26} {'profile':<44} {'suite':<10} {'ok':<5} started")
    for record in records:
        print(
            f"{str(record.get('id')):<26} "
            f"{str(record.get('profile')):<44} "
            f"{str(record.get('suite')):<10} "
            f"{str(record.get('ok')):<5} "
            f"{record.get('started_at')}"
        )
    return 0


def _print_record(record: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(record, indent=1))
        return 0
    print(
        f"{record.get('profile')}  suite {record.get('suite')}  {record.get('started_at')}"
    )
    for line in benchmarks.format_metrics(record):
        print(line)
    return 0


# -- endpoint resolution --------------------------------------------------
@dataclass
class Endpoint:
    """A client-facing endpoint and the profile that describes it."""

    url: str
    profile: Profile
    supervisor: Supervisor | None = None
    model: str | None = None
    context_length: int | None = None


def _endpoint_for(args: argparse.Namespace) -> Endpoint:
    """Resolves an endpoint: an explicit URL, a live launch, or a new server."""
    if args.base_url:
        profile = _profile_for_endpoint(args, args.base_url)
        return Endpoint(
            args.base_url,
            profile,
            model=_served_model(Client(args.base_url, timeout=10.0), profile),
        )
    root = _state_root(args)
    running = running_endpoint(root)
    if running is not None:
        endpoint, record = running
        sys.stderr.write(
            f"using the running launch of {record.get('repo')} at {endpoint}\n"
        )
        profile = _profile_for_endpoint(args, endpoint, record)
        return Endpoint(
            endpoint,
            profile,
            model=_served_model(Client(endpoint, timeout=10.0), profile, record),
            context_length=_recorded_context(record),
        )
    resolved = _resolve(args, install=True)
    prepared = _prepare(args, resolved, ENDPOINT_CAPACITY_OPTIONS)
    options = LaunchOptions(
        device=args.device,
        port=args.port,
        workers=args.workers,
        proxy=not args.no_proxy,
        auth_token=args.api_key,
        startup_timeout=args.wait,
        loom_serve=args.loom_serve,
    )
    run = RunStore(root).create(
        metadata={"repo": resolved.profile.repo, "source": "ephemeral"}
    )
    plan = _plan(prepared, resolved, run, options)
    settings = plan.settings
    supervisor = Supervisor(plan, options, on_output=None, state_root=root)
    supervisor.start()
    return Endpoint(
        supervisor.endpoint,
        resolved.profile,
        supervisor,
        model=_served_model(
            Client(supervisor.endpoint, timeout=10.0), resolved.profile
        ),
        context_length=settings.context_length,
    )


def _recorded_context(record: dict) -> int | None:
    """The context a live launch was configured with, when it recorded one."""
    value = record.get("context_length")
    return value if isinstance(value, int) else None


def _served_model(
    client: Client, profile: Profile, record: dict | None = None
) -> str | None:
    """The name an endpoint answers to, which need not be the card's model id.

    A deployment registers under its own name, and a request that names the
    checkpoint instead is rejected as unregistered: ask the endpoint what it
    serves and prefer the card's spelling when it matches.
    """
    names = client.model_names(timeout=10.0)
    if not names:
        return None
    candidates = [
        profile.model_id,
        profile.slug,
        profile.name,
        str_value((record or {}).get("deployment")),
    ]
    for candidate in candidates:
        if candidate and candidate in names:
            return candidate
    return names[0]


def _profile_for_endpoint(
    args: argparse.Namespace, endpoint: str, record: dict | None = None
) -> Profile:
    """The profile a conversation describes.

    With a target, the target's own documents are used. Without one, the model
    the endpoint reports is named and described by an otherwise empty profile:
    the endpoint owns generation, so only display and template previews need a
    local document.
    """
    target = getattr(args, "target", None)
    if target is None and record is not None:
        target = record.get("repo")
    if target is not None:
        local = _local_target(target)
        if local is not None:
            return Profile.from_local(local)
        return _profile_from_repo(args, target, getattr(args, "revision", None))
    client = Client(endpoint, token=getattr(args, "api_key", None), timeout=10.0)
    names = client.model_names(timeout=5.0)
    name = names[0] if names else "unknown"
    return Profile(repo=name, model_id=name, name=name)


# -- observation ----------------------------------------------------------
def _cmd_status(args: argparse.Namespace) -> int:
    root = _state_root(args)
    record = read_current(root)
    if args.json:
        print(json.dumps({"state": describe_state(root), "current": record}, indent=1))
        return 0
    if not record:
        print("no launch is running")
        print(f"state directory: {state_directory(root)}")
        return 1
    print(f"launch     {record.get('repo')} @ {record.get('endpoint')}")
    print(f"  deployment {record.get('deployment')}")
    print(f"  server    {record.get('server_endpoint')}")
    print(f"  pid       {record.get('pid')}")
    print(f"  started   {record.get('started_at')}")
    client = Client(str(record.get("endpoint")), timeout=5.0)
    try:
        status, document = client.get_json("/loom/status", timeout=5.0)
    except LauncherError as exc:
        print(f"  probe     failed: {exc}")
        return 1
    if status != 200:
        print(f"  probe     HTTP {status}")
        return 1
    metrics = ((document.get("proxy") or {}).get("metrics")) or {}
    print(
        f"  traffic   {metrics.get('requests', 0)} request(s), "
        f"{metrics.get('completion_tokens', 0)} completion tokens, "
        f"in flight {metrics.get('in_flight', 0)}"
    )
    ttft = metrics.get("ttft_seconds") or {}
    decode = metrics.get("decode_tokens_per_second") or {}

    def _seconds(value: object) -> str:
        return f"{value}s" if isinstance(value, (int, float)) else "-"

    print(
        f"  latency   ttft p50 {_seconds(ttft.get('p50'))} "
        f"p95 {_seconds(ttft.get('p95'))}, "
        f"decode mean {human_rate(decode.get('mean')) or '-'}"
    )
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    root = _state_root(args)
    runs = RunStore(root)
    run = runs.open(args.run) if args.run else runs.latest()
    if run is None:
        sys.stderr.write("error: no runs recorded\n")
        return 1
    path = run.serve_log
    if not path.is_file():
        sys.stderr.write(f"error: {path} does not exist\n")
        return 1
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-args.lines :]:
        print(line)
    if args.follow:
        try:
            with open(path, encoding="utf-8", errors="replace") as file:
                file.seek(0, os.SEEK_END)
                while True:
                    chunk = file.readline()
                    if not chunk:
                        time.sleep(0.2)
                        continue
                    print(chunk, end="")
        except KeyboardInterrupt:
            return 0
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    root = _state_root(args)
    record = read_current(root)
    if not record:
        print("no launch is running")
        return 1
    pid = record.get("pid")
    if not isinstance(pid, int):
        sys.stderr.write("error: the current record has no pid\n")
        return 1
    try:
        os.kill(pid, signal.SIGINT)
    except OSError as exc:
        sys.stderr.write(f"error: cannot signal {pid}: {exc}\n")
        return 1
    print(f"asked launch {pid} ({record.get('repo')}) to stop")
    return 0


def _cmd_hardware(args: argparse.Namespace) -> int:
    hardware = detect()
    if args.json:
        print(json.dumps(hardware.describe(), indent=1))
        return 0
    document = hardware.describe()
    print(f"system   {document['system']} {document['machine']}")
    print(f"cpu      {document['cpu'] or 'unknown'}")
    print(f"memory   {document['memory']}")
    gpu = document.get("gpu")
    if gpu:
        print(f"gpu      {gpu['name']} ({gpu['gfx_arch'] or 'arch unknown'})")
    else:
        print("gpu      none detected")
    print(f"npu      {'present' if document['npu'] else 'not detected'}")
    print(f"device   {document['default_device_uri']}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[dict] = []

    def check(name: str, ok: bool | None, detail: str) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})

    try:
        path = serve_command(getattr(args, "loom_serve", None))
        check("loom-serve", True, path)
    except LauncherError as exc:
        check("loom-serve", False, str(exc))
    try:
        command = fetch_command(getattr(args, "loom_fetch", None))
        check("loom-fetch", True, " ".join(command))
    except LauncherError as exc:
        check("loom-fetch", False, str(exc))
    hub = _hub(args)
    try:
        found = hub.profiles("loom", limit=1)
        check("hub", True, f"{hub.endpoint} ({len(found)} profile(s) sampled)")
    except LauncherError as exc:
        check("hub", None, f"{hub.endpoint} unreachable: {exc}")
    root = _state_root(args)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / "doctor.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        check("state", True, str(root))
    except OSError as exc:
        check("state", False, f"{root}: {exc}")
    hardware = detect()
    check(
        "device",
        bool(hardware.devices),
        hardware.default_device_uri()
        + ("" if hardware.devices else " (no AMD GPU detected)"),
    )
    if args.json:
        print(json.dumps({"checks": checks}, indent=1))
        return 0 if all(check["ok"] is not False for check in checks) else 1
    for entry in checks:
        marker = {True: "ok  ", False: "FAIL", None: "warn"}[entry["ok"]]
        print(f"{marker} {entry['check']:<12} {entry['detail']}")
    return 0 if all(check["ok"] is not False for check in checks) else 1


def _signal_termination() -> None:
    def handler(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
