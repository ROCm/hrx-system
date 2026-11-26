# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom SDK wheels with Python APIs and native commands from the same build."""

load("@rules_python//python:packaging.bzl", "py_wheel")

def _metadata_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(output, json.encode({"catalog": ctx.attr.catalog or None}))
    return [DefaultInfo(files = depset([output]))]

_metadata = rule(
    implementation = _metadata_impl,
    attrs = {"catalog": attr.string()},
)

def _distribution_version(version):
    if "+" in version:
        fail("Supply the base SDK version; the native configuration owns the local version suffix")
    versions = {
        Label("//loom/packaging/python:profile_release"): version,
        Label("//loom/packaging/python:profile_tracy"): version + "+tracy",
        Label("//loom/packaging/python:profile_debug"): version + "+debug",
        Label("//loom/packaging/python:profile_asan"): version + "+asan",
    }
    if ".dev" in version:
        versions[Label("//loom/packaging/python:profile_development")] = version
    return select(
        versions,
        no_match_error = (
            "Unsupported Loom wheel configuration: use --config=opt, " +
            "--config=opt --//runtime/config/base/tracing:provider=tracy, " +
            "--compilation_mode=dbg, or --config=asan. " +
            "Fastbuild requires a development version; combined instrumentation is not a wheel profile."
        ),
    )

def loom_python_wheel(name, version, catalog = "", testonly = False):
    """Builds an SDK wheel without publishing it.

    Args:
      name: Target name.
      version: Base Python distribution version, without a local suffix. Native
        tracing, debug, and ASAN configurations add their matching suffix.
      catalog: Compatible github:OWNER/REPOSITORY@TAG association; empty for a
        development SDK requiring an explicit discovery source.
      testonly: Whether this wheel is a test fixture.
    """
    _metadata(name = name + "_metadata", catalog = catalog, testonly = testonly)
    py_wheel(
        name = name,
        testonly = testonly,
        description_file = Label("//loom/packaging/python:PYPI_README.md"),
        distribution = "hrx-loom",
        data_files = select({
            "@platforms//os:linux": {
                Label("//loom/packaging/python:native"): "scripts/",
                Label("//loom/packaging/linux:loom-serve-sandbox"): "scripts/",
            },
            "//conditions:default": {
                Label("//loom/packaging/python:native"): "scripts/",
            },
        }),
        entry_points = {"console_scripts": [
            "loom-compile-report = loom.tools.compile_report:main",
            "loom-export-cxx = loom.tools.export_cxx:main",
            "loom-fetch = loom.model.fetch_main:main",
            "loom-lint = loom.tools.source_lint:main",
        ]},
        extra_distinfo_files = {
            Label("//:LICENSE"): "licenses/LICENSE",
            ":" + name + "_metadata": "loom.json",
        },
        extra_requires = {"hf": ["huggingface_hub>=2.2,<3"]},
        homepage = "https://github.com/ROCm/hrx-system",
        license = "Apache-2.0 WITH LLVM-exception",
        platform = select({
            Label("//loom/packaging/python:linux_x86_64"): "linux_x86_64",
            Label("//loom/packaging/python:linux_aarch64"): "linux_aarch64",
            Label("//loom/packaging/python:windows_x86_64"): "win_amd64",
        }),
        python_requires = ">=3.12",
        python_tag = "py3",
        stamp = 0,
        strip_path_prefixes = ["loom/py"],
        summary = "Loom compiler, serving and authoring tools, and Python SDK",
        tags = ["manual"],
        twine_binary = None,
        version = _distribution_version(version),
        deps = [
            Label("//loom/packaging/python:package"),
            Label("//loom/packaging/python:schemas"),
        ],
    )
