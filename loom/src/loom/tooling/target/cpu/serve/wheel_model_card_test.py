# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Installs the SDK and serves a fetched model with its packaged native tools."""

import functools
import http.server
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

from loom_serve_testing.http_harness import running_server
from model_card_test import check_model, check_text
from release import inspect_wheel, read_identity


def pick_model(arguments, environment, output):
    """Drives the installed command through a real terminal and restores it."""
    import pty
    import termios

    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    termios.tcsetwinsize(slave, (24, 100))
    try:
        with subprocess.Popen(
            [str(value) for value in arguments],
            stdin=slave,
            stderr=slave,
            stdout=subprocess.PIPE,
            env=dict(environment, TERM="xterm-256color"),
        ) as process:
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(master, selectors.EVENT_READ, "terminal")
                    selector.register(process.stdout, selectors.EVENT_READ, "result")
                    for marker, response in (
                        (b"Enter: select", b"\r"),
                        (
                            b"Install directory (Ctrl-C cancels): ",
                            str(output).encode() + b"\n",
                        ),
                    ):
                        content = b""
                        while marker not in content:
                            for key, _ in selector.select():
                                data = os.read(key.fd, 65536)
                                assert data and key.data == "terminal", (content, data)
                                content += data
                        os.write(master, response)
                result, _ = process.communicate()
                assert process.returncode == 0, result
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


def main(
    wheel,
    associated_wheel,
    deployment,
    elf_inspector,
    version_file,
    readme,
    expected_drivers=None,
):
    wheel = Path(wheel).absolute()
    associated_wheel = Path(associated_wheel).absolute()
    deployment = Path(deployment).absolute()
    elf_inspector = Path(elf_inspector).absolute()
    identity = read_identity(Path(version_file))
    artifact = inspect_wheel(wheel, identity, Path(readme).read_text(encoding="utf-8"))
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON")
        and key
        not in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "DYLD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
        )
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)

        def run(*arguments, expected=0):
            command = [str(argument) for argument in arguments]
            # Windows CreateProcess does not use the supplied environment's
            # PATH when locating the executable. Resolve our isolated PATH here.
            executable = shutil.which(command[0], path=environment.get("PATH"))
            assert executable is not None, command[0]
            command[0] = executable
            result = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert result.returncode == expected, (result.stdout, result.stderr)
            return result.stdout

        run(sys.executable, "-I", "-m", "venv", directory / "venv")
        scripts = directory / "venv" / ("Scripts" if os.name == "nt" else "bin")
        system_path = (
            str(Path(os.environ["SystemRoot"]) / "System32")
            if os.name == "nt"
            else os.defpath
        )
        environment["PATH"] = str(scripts) + os.pathsep + system_path
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        fetch = scripts / ("loom-fetch.exe" if os.name == "nt" else "loom-fetch")

        def install(artifact):
            run(
                python,
                "-I",
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--force-reinstall",
                artifact,
            )

        install(wheel)
        assert shutil.which("bazel", path=environment["PATH"]) is None
        assert shutil.which("bazelisk", path=environment["PATH"]) is None
        with ZipFile(wheel) as archive:
            native_names = {
                Path(entry.filename).name
                for entry in archive.infolist()
                if not entry.is_dir() and ".data/scripts/" in entry.filename
            }
        companions = (
            {"iree.dll", "loomc.dll"}
            if os.name == "nt"
            else {"libiree.so", "libloomc.so"}
        )
        assert companions <= native_names, native_names
        if os.name == "nt":
            symbols = {name for name in native_names if name.endswith(".pdb")}
            project_binaries = {
                name
                for name in native_names
                if name.endswith(".exe")
                or name in companions | {"amdf_wkmi_bridge.dll"}
            }
            expected_symbols = {
                str(Path(name).with_suffix(".pdb")) for name in project_binaries
            }
            if artifact.profile in ("debug", "asan"):
                assert symbols == expected_symbols, (symbols, expected_symbols)
            else:
                assert symbols <= expected_symbols, (symbols, expected_symbols)
        for name in sorted(native_names):
            installed = scripts / name
            with installed.open("rb") as stream:
                signature = b"\x7fELF"
                if os.name == "nt":
                    signature = (
                        b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\x00\x00\x00"
                        if name.endswith(".pdb")
                        else b"MZ"
                    )
                assert stream.read(len(signature)) == signature, name
            if os.name != "nt":
                runpath = run(elf_inspector, "--print-rpath", installed).strip()
                assert runpath == "$ORIGIN", (name, runpath)
        for command in (
            "loom-benchmark",
            "loom-check",
            "loom-compile",
            "loom-compile-report",
            "loom-export-cxx",
            "loom-fetch",
            "loom-format",
            "loom-import-cxx",
            "loom-link",
            "loom-lint",
            "loom-opt",
            "loom-run",
            "loom-serve",
            "loom-test",
        ):
            executable = shutil.which(command, path=environment["PATH"])
            assert executable and Path(executable).parent == scripts, command
            run(command, "--help")
        if expected_drivers is not None:
            driver_output = run("loom-run", "--list_drivers")
            drivers = {
                line.split(":", 1)[0].strip()
                for line in driver_output.splitlines()
                if line.strip() and not line.startswith("#")
            }
            assert drivers == set(expected_drivers.split(",")), driver_output
        source = directory / "program.cxx"
        source.write_text(
            "#include <loomcxx/math.h>\n"
            "float twice(float value) { return value + value; }\n"
        )
        module = directory / "program.loombc"
        run("loom-import-cxx", "--to=bc", "--output=" + str(module), source)
        text = run("loom-format", "--from=bc", "--to=text", module)
        assert "@twice" in text and "scalar.addf" in text, text
        run(
            python,
            "-I",
            "-c",
            """
import importlib.metadata
import importlib.util
import json
import sys
from importlib.resources import files
from pathlib import Path
from loom import LoomBuilder, default_ops, default_types
from loom.format.bytecode.reader import read_module
from loom.format.bytecode.writer import write_module
from loom.format.text.parser import Parser
from loom.model import ModelCard
from loom.model.fetch_catalog import installed_catalog

assert importlib.util.find_spec('huggingface_hub') is None
distribution = importlib.metadata.distribution('hrx-loom')
assert distribution.version == sys.argv[1], (distribution.version, sys.argv[1])
catalog = json.loads(sys.argv[2])
if catalog is not None:
    assert installed_catalog() == catalog
assert distribution.metadata.get_all('Provides-Extra') == ['hf']
assert distribution.requires == ["huggingface_hub<3,>=2.2; extra == 'hf'"], distribution.requires
default_ops()
default_types()
assert all(Path(module.__file__).is_relative_to(sys.prefix)
           for name, module in list(sys.modules.items())
           if name == 'loom' or name.startswith('loom.'))
for name in ('model-card', 'model-lock', 'model-catalog'):
    schema = json.loads(files('loom.model').joinpath(name + '.v1.schema.json').read_text())
    assert schema['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
""",
            wheel.name.split("-")[1],
            json.dumps(identity.catalog),
        )
        assert "--offline" in run(fetch, "--help")
        if identity.catalog is None:
            failure = json.loads(run(fetch, "--list", "--json", expected=1))
            assert "no associated model catalog" in failure["message"]
        install(associated_wheel)
        run(
            python,
            "-I",
            "-c",
            "import importlib.metadata, sys; "
            "assert importlib.metadata.version('hrx-loom') == sys.argv[1]; "
            "from loom.model.fetch_catalog import installed_catalog; "
            "assert installed_catalog() == 'github:fixture/library@compatible'",
            associated_wheel.name.split("-")[1],
        )
        # Acquisition and native output must use the final publication artifact.
        install(wheel)
        card = deployment.parent / "model.json"
        listing = json.loads(run(fetch, card, "--list", "--json"))
        assert listing["model"]["id"] == "loom-fixture/text"
        unsafe = directory / "display.json"
        listing["model"]["name"] = "model\x1b]52;secret\x07\nnext"
        unsafe.write_text(json.dumps(listing))
        displayed = run(fetch, unsafe, "--list")
        assert displayed.startswith("model\\u001b]52;secret\\u0007\\u000anext\n")
        assert (
            json.loads(run(fetch, unsafe, "--list", "--json"))["model"]["name"]
            == listing["model"]["name"]
        )
        installed = directory / "installed"
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(card.parent)
        )
        with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as origin:
            thread = threading.Thread(target=origin.serve_forever)
            thread.start()
            try:
                published_lock = directory / "publication.json.lock"
                publication = json.loads(
                    run(
                        fetch,
                        f"http://127.0.0.1:{origin.server_port}/model.json",
                        "--update-lock",
                        "--all-variants",
                        "--lockfile",
                        published_lock,
                        "--no-input",
                        "--json",
                        "--cache",
                        directory / "cache",
                    )
                )
                assert publication["lockfile"] == str(published_lock)
                assert len(publication["selections"]) == 1
                result = json.loads(
                    run(
                        fetch,
                        f"http://127.0.0.1:{origin.server_port}/model.json",
                        "--preset=check",
                        "--lockfile",
                        published_lock,
                        "--no-input",
                        "--json",
                        "--cache",
                        directory / "cache",
                        "--output",
                        installed,
                    )
                )
            finally:
                origin.shutdown()
                thread.join()
        assert (installed / "model.json.lock").is_file()
        run(
            fetch,
            installed / "model.json",
            "--preset=check",
            "--offline",
            "--no-input",
            "--cache",
            directory / "cache",
            "--output",
            directory / "offline",
        )
        assert Path(result["card"]) == installed / "model.json"
        catalog = installed / "catalog.json"
        run(
            python,
            "-I",
            "-m",
            "loom.model.catalog",
            "--output",
            catalog,
            "--card=model.json=" + str(installed / "model.json"),
        )
        listing = json.loads(run(fetch, "--catalog", catalog, "--list", "--json"))
        assert list(listing["models"]) == ["loom-fixture/text"]
        assert listing["snapshot"]["revision"] is None
        catalog_installation = directory / "catalog-installed"
        run(
            fetch,
            "--catalog",
            catalog,
            "--model=loom-fixture/text",
            "--preset=check",
            "--offline",
            "--no-input",
            "--cache",
            directory / "cache",
            "--output",
            catalog_installation,
        )
        receipt = json.loads((catalog_installation / ".loom-fetch.json").read_text())
        assert receipt["catalog"] == listing["snapshot"]
        assert receipt["card"]["location"] == str(installed / "model.json")
        if os.name != "nt":
            catalog_installation = directory / "picker-installed"
            pick_model(
                [
                    fetch,
                    "--catalog",
                    catalog,
                    "--preset=check",
                    "--offline",
                    "--cache",
                    directory / "cache",
                ],
                environment,
                catalog_installation,
            )
            receipt = json.loads(
                (catalog_installation / ".loom-fetch.json").read_text()
            )
            assert receipt["catalog"] == listing["snapshot"]
        server = shutil.which("loom-serve", path=environment["PATH"])
        model = str(catalog_installation / "model.json")
        with running_server(
            [
                server,
                "--device=task",
                "--workers=1",
                "--listen=127.0.0.1:0",
                "--model=" + model,
            ],
            directory / "peer.log",
            cwd=directory,
            environment=environment,
        ) as (peer, address):
            assert address is not None, (directory / "peer.log").read_text()
            check_model(str(server), model, "abab", environment=environment)
            # Retiring another server must not signal this server or the test
            # runner, including when Windows console control is broadcast.
            assert peer.poll() is None
            check_text(urlsplit("http://" + address), "loom-fixture/text", "abab")
        assert peer.returncode == 0


if __name__ == "__main__":
    main(*sys.argv[1:])
