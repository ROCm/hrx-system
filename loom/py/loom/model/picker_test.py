# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import copy
import io
import json
import os
import selectors
import signal
import subprocess
import sys
import unittest
from contextlib import contextmanager

import pytest

from loom.model import ManifestError, ModelCard
from loom.model.catalog import generate
from loom.model.fetch_picker import Choice, Picker
from loom.model.fetch_source import FetchError


def _card(identity="fixture/alpha"):
    return {
        "schema_version": 1,
        "model": {"id": identity, "name": identity},
        "programs": {"source": {"inputs": ["code.loom"], "libraries": []}},
        "services": {"chat": {"kind": "text", "version": 1}},
        "variants": {"small": {"services": {"chat": {"programs": ["source"]}}}},
    }


def test_selection_uses_compatible_services_variants_and_canonical_options():
    document = _card()
    document["services"]["chat"]["options"] = ["context"]
    document["services"]["speech"] = {"kind": "recognition", "version": 1}
    document["programs"]["archive"] = {"inputs": ["code.loombc"], "libraries": []}
    document["variants"]["small"]["services"]["chat"]["programs"].append("archive")
    document["variants"]["audio"] = {"services": {"speech": {"programs": ["source"]}}}
    document["presets"] = {
        "check": {
            "service": "chat",
            "variant": "small",
            "programs": "source",
            "options": {"context": 16},
        }
    }
    card = ModelCard(document)
    output = io.StringIO()
    picker = Picker(io.StringIO(""), output)
    selected = picker.selection(card, programs="archive")
    assert selected.service_name == "chat"
    assert selected.variant == "small"
    assert selected.programs == "archive"
    assert selected.settings["options"] == {"context": 16}
    assert set(selected.files) == {"code.loombc"}
    assert output.getvalue() == ""
    selected = picker.selection(card, variant="audio")
    assert selected.service_name == "speech"
    assert selected.programs == "source"
    with pytest.raises(ManifestError, match="unknown name"):
        picker.selection(card, service="missing")
    with pytest.raises(FetchError, match="no compatible choices"):
        picker.selection(card, variant="audio", programs="archive")


def test_plain_named_choices_filter_and_escape_metadata():
    output = io.StringIO()
    picker = Picker(io.StringIO("/second\nfirst\nsecond\n"), output)
    selected = picker.choose(
        "Model",
        [Choice("first", "First"), Choice("second", "Second\x1b]52;secret\x07")],
    )
    assert selected == "second"
    assert "Choose one of the displayed names" in output.getvalue()
    assert "\\u001b]52;secret\\u0007" in output.getvalue()
    assert "\x1b" not in output.getvalue()


def test_named_preset_supplies_choices_without_reprompting():
    document = _card()
    document["programs"]["archive"] = {"inputs": ["code.loombc"], "libraries": []}
    document["variants"]["small"]["services"]["chat"]["programs"].append("archive")
    document["variants"]["large"] = copy.deepcopy(document["variants"]["small"])
    document["services"]["speech"] = {"kind": "speech", "version": 1}
    document["variants"]["audio"] = {"services": {"speech": {"programs": ["source"]}}}
    document["presets"] = {
        "chat": {"service": "chat", "variant": "small", "programs": "source"}
    }
    output = io.StringIO()
    picker = Picker(io.StringIO(""), output)
    selected = picker.selection(ModelCard(document), preset="chat")
    assert (
        selected.preset,
        selected.service_name,
        selected.variant,
        selected.programs,
    ) == ("chat", "chat", "small", "source")
    selected = picker.selection(
        ModelCard(document), preset="chat", variant="large", programs="archive"
    )
    assert selected.variant == "large"
    assert set(selected.files) == {"code.loombc"}
    assert output.getvalue() == ""


def test_preset_picker_resolves_programs_without_another_question():
    document = _card()
    document["programs"]["archive"] = {"inputs": ["code.loombc"], "libraries": []}
    document["variants"]["small"]["services"]["chat"]["programs"].append("archive")
    document["presets"] = {
        name: {"service": "chat", "variant": "small", "programs": programs}
        for name, programs in (("development", "source"), ("release", "archive"))
    }
    output = io.StringIO()
    selected = Picker(io.StringIO("release\n"), output).selection(ModelCard(document))
    assert selected.preset == "release"
    assert set(selected.files) == {"code.loombc"}
    assert "Programs" not in output.getvalue()


def test_program_ambiguity_requires_a_preset_or_explicit_override():
    document = _card()
    document["programs"]["archive"] = {"inputs": ["code.loombc"], "libraries": []}
    document["variants"]["small"]["services"]["chat"]["programs"].append("archive")
    with pytest.raises(ManifestError, match="choose a preset or specify --programs"):
        Picker(io.StringIO(""), io.StringIO()).selection(ModelCard(document))


def test_preset_picker_displays_informational_memory_and_assumptions():
    document = _card()
    selection = {"variant": "small", "programs": "source", "service": "chat"}
    document["presets"] = {
        "average": selection
        | {"memory": {"weights_bytes": 1073741824, "workload": "One average request."}},
        "empty": selection
        | {
            "memory": {
                "weights_bytes": 0,
                "transient_bytes": 2**64 - 1,
                "workload": "Count boundaries.",
            }
        },
    }
    output = io.StringIO()
    selected = Picker(io.StringIO("average\n"), output).selection(ModelCard(document))
    assert selected.settings["memory"]["weights_bytes"] == 1073741824
    assert (
        "Estimated memory: weights 1.0 GiB; transients unknown. One average request."
        in output.getvalue()
    )
    assert (
        "Estimated memory: weights 0 bytes; transients 16.0 EiB." in output.getvalue()
    )


def test_picker_selects_named_partition_from_the_same_compatible_choices():
    document = _card()
    scope = document["variants"].pop("small")
    document["variants"]["small"] = {
        "partitions": {
            "node-0": scope,
            "other side": copy.deepcopy(scope),
        }
    }
    selected = Picker(io.StringIO("other side\n"), io.StringIO()).selection(
        ModelCard(document)
    )
    assert selected.partition == "other side"
    selected = Picker(io.StringIO(""), io.StringIO()).selection(
        ModelCard(document), partition="node-0"
    )
    assert selected.partition == "node-0"
    with pytest.raises(ManifestError, match="unknown name"):
        Picker(io.StringIO(""), io.StringIO()).selection(
            ModelCard(document), partition="absent"
        )


def test_empty_directory_does_not_accept_an_implicit_destination():
    output = io.StringIO()
    picker = Picker(io.StringIO("\n"), output)
    with pytest.raises(KeyboardInterrupt):
        picker.directory()
    assert "nothing has been installed" in output.getvalue()


def _publish(directory):
    cards = {}
    for identity in ("fixture/alpha", "fixture/beta", "fixture/tést"):
        path = identity.split("/")[1] + ".json"
        card = ModelCard(_card(identity))
        cards[path] = card
        (directory / path).write_text(json.dumps(card.document))
    (directory / "code.loom").write_bytes(b"fixture code")
    catalog = directory / "catalog.json"
    catalog.write_text(json.dumps(generate(cards).document))
    return catalog


class _Terminal:
    def __init__(self, process, master, slave):
        # Owned subprocess; stdout signals exit without depending on PTY EOF.
        self.process = process
        # PTY master used by the test as a terminal emulator.
        self.master = master
        # Retained slave for kernel size/mode observations.
        self.slave = slave
        # Unconsumed output after a readiness marker.
        self.buffer = b""

    def await_text(self, text):
        marker = text.encode()
        with selectors.DefaultSelector() as selector:
            selector.register(self.master, selectors.EVENT_READ, "terminal")
            selector.register(self.process.stdout, selectors.EVENT_READ, "result")
            while marker not in self.buffer:
                for key, _ in selector.select():
                    data = os.read(key.fd, 65536)
                    if key.data == "result" or not data:
                        raise AssertionError(
                            f"CLI exited before {text!r}: {self.buffer!r} {data!r}"
                        )
                    self.buffer += data
        end = self.buffer.index(marker) + len(marker)
        result, self.buffer = self.buffer[:end], self.buffer[end:]
        return result

    def send(self, text):
        os.write(self.master, text.encode())

    def resize(self, columns, rows):
        import termios

        termios.tcsetwinsize(self.slave, (rows, columns))


@contextmanager
def _terminal(arguments, *, term="xterm-256color", environment=None):
    if os.name == "nt":
        raise unittest.SkipTest("this case requires a POSIX PTY")
    import pty
    import termios

    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    termios.tcsetwinsize(slave, (24, 100))
    environment = dict(os.environ if environment is None else environment, TERM=term)
    try:
        with subprocess.Popen(
            arguments,
            stdin=slave,
            stderr=slave,
            stdout=subprocess.PIPE,
            env=environment,
        ) as process:
            try:
                yield _Terminal(process, master, slave)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def _arguments(directory, *arguments):
    return [
        sys.executable,
        "-m",
        "loom.model.fetch_main",
        "--catalog",
        str(_publish(directory)),
        "--cache",
        str(directory / "cache"),
        "--offline",
        "--no-lock",
        *arguments,
    ]


def test_terminal_catalog_filter_resize_and_acquisition(tmp_path):
    with _terminal(_arguments(tmp_path)) as terminal:
        terminal.await_text("Enter: select")
        terminal.send("tést")
        terminal.await_text("1/1 choices")
        terminal.resize(20, 8)
        terminal.await_text("Resize to at least")
        terminal.send("\rX")
        terminal.await_text("Resize to at least")
        assert not (tmp_path / "installed").exists()
        terminal.send("\x7f")
        terminal.resize(100, 24)
        terminal.await_text("Enter: select")
        terminal.send("\r")
        terminal.await_text("Install directory (Ctrl-C cancels): ")
        terminal.send(str(tmp_path / "installed") + "\n")
        result, _ = terminal.process.communicate()
        assert terminal.process.returncode == 0, result
        card = ModelCard.parse((tmp_path / "installed/model.json").read_text())
        assert card.select().model_id == "fixture/tést"
        assert (tmp_path / "installed/code.loom").read_bytes() == b"fixture code"
        assert (tmp_path / "installed/.loom-fetch.json").is_file()


@pytest.mark.parametrize("keys", ["\x1b[B", "\x1b[F\x1b[A", "\x1b[6~\x1b[5~\x1b[B"])
def test_terminal_navigation_selects_the_visible_model(tmp_path, keys):
    with _terminal(
        _arguments(tmp_path, "--output=" + str(tmp_path / "installed"))
    ) as terminal:
        terminal.await_text("Enter: select")
        terminal.send(keys)
        terminal.await_text("2/3 choices")
        terminal.send("\r")
        result, _ = terminal.process.communicate()
        assert terminal.process.returncode == 0, result
        card = ModelCard.parse((tmp_path / "installed/model.json").read_text())
        assert card.select().model_id == "fixture/beta"


@pytest.mark.parametrize("key", ["\x1b", "\x04", "interrupt"])
def test_terminal_cancel_restores_mode_without_materializing(tmp_path, key):
    with _terminal(
        _arguments(tmp_path, "--output=" + str(tmp_path / "installed"))
    ) as terminal:
        terminal.await_text("Enter: select")
        if key == "interrupt":
            terminal.process.send_signal(signal.SIGINT)
        else:
            terminal.send(key)
        terminal.await_text("\x1b[?25h\x1b[?1049l")
        terminal.process.communicate()
        assert terminal.process.returncode == 130
        assert not (tmp_path / "installed").exists()


def test_dumb_terminal_uses_named_choices(tmp_path):
    with _terminal(
        _arguments(tmp_path, "--output=" + str(tmp_path / "installed")), term="dumb"
    ) as terminal:
        output = terminal.await_text("Name: ")
        assert b"\x1b" not in output
        terminal.send("fixture/beta\n")
        result, _ = terminal.process.communicate()
        assert terminal.process.returncode == 0, result
        assert (
            ModelCard.parse((tmp_path / "installed/model.json").read_text())
            .select()
            .model_id
            == "fixture/beta"
        )


@pytest.mark.parametrize("flag", ["--json", "--no-input"])
def test_explicit_headless_modes_never_open_a_picker(tmp_path, flag):
    result = subprocess.run(
        _arguments(tmp_path, flag), input=b"fixture/alpha\n", capture_output=True
    )
    assert result.returncode == 2
    assert b"select a card or --model=ID" in result.stderr
    assert b"\x1b" not in result.stderr
