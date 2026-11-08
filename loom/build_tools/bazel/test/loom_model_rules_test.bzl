# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model composition retains source contracts through dependency analysis."""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:util.bzl", "TestingAspectInfo")
load("//loom/build_tools/bazel:defs.bzl", "LoomModelInfo", "loom_library", "loom_model")

_PACKAGE = "loom/build_tools/bazel/test/"
_CARD = "//loom/src/loom/serve/fixture/text:native.card.json"

def _declare_composition(name, form):
    loom_library(
        name = name + "_leaf",
        srcs = ["link_kernels.loom", "dependency_checks.loom"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_primary",
        srcs = ["model_provider.cxx"],
        data = ["model_provider.h"],
        input_format = "cxx",
        inputopts = ["cxx:fp-model=fast"],
        deps = [":" + name + "_leaf"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_branch",
        srcs = ["profile_benchmarks.loom"],
        deps = [":" + name + "_primary", ":" + name + "_leaf"],
        tags = ["manual"],
    )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_primary"],
        libraries = [":" + name + "_branch"],
        form = form,
        programs = "selected",
        preset = "check",
        model_name = "fixture",
        encoding = "preserve",
        tags = ["manual"],
    )

def _expect_composition(env, target, form):
    info = target[LoomModelInfo]
    stem = target.label.name.removesuffix("_subject")
    primary = {
        "data": [_PACKAGE + "model_provider.h"],
        "format": "loombc" if form == "module" else "cxx",
        "path": _PACKAGE + (stem + "_primary.loombc" if form == "module" else "model_provider.cxx"),
    }
    if form == "sources":
        primary["options"] = ["cxx:fp-model=fast"]
    libraries = [
        {"format": "loombc", "path": _PACKAGE + stem + "_branch.loombc"},
        {"format": "loombc", "path": _PACKAGE + stem + "_leaf.loombc"},
    ] if form == "module" else [
        _PACKAGE + "profile_benchmarks.loom",
        _PACKAGE + "link_kernels.loom",
        _PACKAGE + "dependency_checks.loom",
    ]
    expected = {"selected": {"inputs": [primary], "libraries": libraries}}
    if info.programs != expected:
        env.fail("expected %r, got %r" % (expected, info.programs))
    env.expect.that_str(info.card).equals("loom/src/loom/serve/fixture/text/native.card.json")
    paths = [file.short_path for file in target[OutputGroupInfo].model_files.to_list()]

    # The auxiliary file remains runtime data even when the owning source has
    # become bytecode. A primary reached through a library is emitted once.
    expected_files = [primary["path"], _PACKAGE + "model_provider.h"] + [
        item["path"] if type(item) == "dict" else item
        for item in libraries
    ]
    for path in expected_files:
        env.expect.that_collection(paths).contains(_PACKAGE + stem + "_subject/" + path)
    env.expect.that_int(len(paths)).equals(len(expected_files) + 2)

def _test_sources(name, **kwargs):
    _declare_composition(name, "sources")
    analysis_test(name = name, target = ":" + name + "_subject", impl = _sources_impl, **kwargs)

def _sources_impl(env, target):
    _expect_composition(env, target, "sources")

def _test_modules(name, **kwargs):
    _declare_composition(name, "module")
    analysis_test(name = name, target = ":" + name + "_subject", impl = _modules_impl, **kwargs)

def _modules_impl(env, target):
    _expect_composition(env, target, "module")

def _test_bytecode_is_an_original_source(name, **kwargs):
    loom_library(name = name + "_module", srcs = ["link_kernels.loom"], tags = ["manual"])
    loom_library(name = name + "_provider", srcs = [":" + name + "_module"], tags = ["manual"])
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_provider"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", impl = _bytecode_source_impl, **kwargs)

def _bytecode_source_impl(env, target):
    stem = target.label.name.removesuffix("_subject")
    env.expect.that_collection(target[LoomModelInfo].programs["source"]["inputs"]).contains_exactly([
        _PACKAGE + stem + "_module.loombc",
    ])

def _test_external_files_fail(name, **kwargs):
    loom_library(
        name = name + "_provider",
        srcs = ["@bazel_tools//tools/build_defs/repo:http.bzl"],
        input_format = "loom",
        tags = ["manual"],
    )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_provider"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", expect_failure = True, impl = _external_files_impl, **kwargs)

def _external_files_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("requires a local repository file"))

def _test_conflicting_importers_fail(name, **kwargs):
    for suffix, format_name in [("a", "cxx"), ("b", "loom")]:
        loom_library(
            name = name + "_" + suffix,
            srcs = ["model_provider.cxx"],
            input_format = format_name,
            tags = ["manual"],
        )
    loom_model(
        name = name + "_subject",
        card = _CARD,
        inputs = [":" + name + "_a"],
        libraries = [":" + name + "_b"],
        programs = "source",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(name = name, target = ":" + name + "_subject", expect_failure = True, impl = _conflicting_importers_impl, **kwargs)

def _conflicting_importers_impl(env, target):
    env.expect.that_target(target).failures().contains_predicate(matching.contains("conflicting model provider"))

def _find_action(env, target, mnemonic):
    for action in target[TestingAspectInfo].actions:
        if action.mnemonic == mnemonic:
            return action
    env.fail("expected %s action for %s" % (mnemonic, target.label))
    return None

def _executable_basename(path):
    return path.split("/")[-1].removesuffix(".exe")

def _expect_action_executable(env, target, mnemonic, basename):
    action = _find_action(env, target, mnemonic)
    env.expect.that_str(_executable_basename(action.argv[0])).equals(basename)
    return action

def _expect_model_linker(env, target, basename):
    action = _find_action(env, target, "LoomModel")
    linker_ordinal = action.argv.index("--linker") + 1
    env.expect.that_str(_executable_basename(action.argv[linker_ordinal])).equals(basename)
    return action

def _expect_action_input_suffix(env, action, suffix):
    for file in action.inputs.to_list():
        if file.basename.endswith(suffix):
            return
    env.fail("expected action input ending in %r, got %r" % (suffix, action.inputs.to_list()))

def _reject_action_input_suffix(env, action, suffix):
    for file in action.inputs.to_list():
        if file.basename.endswith(suffix):
            env.fail("action unexpectedly consumes %s" % file.path)

def _expect_generated_action_input_suffix(env, action, suffix):
    for file in action.inputs.to_list():
        if file.basename.endswith(suffix):
            if file.is_source:
                env.fail("action directly consumes source file %s" % file.path)
            return
    env.fail("expected generated action input ending in %r" % suffix)

def _test_cxx_import_precedes_link(name, **kwargs):
    loom_library(
        name = name + "_loom",
        srcs = ["link_kernels.loom"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_cxx",
        srcs = ["model_provider.cxx"],
        data = ["model_provider.h"],
        input_format = "cxx",
        inputopts = ["cxx:fp-model=fast"],
        tags = ["manual"],
    )
    loom_library(
        name = name + "_opaque_module",
        srcs = [":" + name + "_loom"],
        tags = ["manual"],
    )
    loom_model(
        name = name + "_model",
        card = _CARD,
        encoding = "bc",
        form = "sources",
        inputs = [":" + name + "_cxx"],
        programs = "selected",
        preset = "check",
        model_name = "fixture",
        tags = ["manual"],
    )
    analysis_test(
        name = name,
        impl = _cxx_import_precedes_link_impl,
        targets = {
            "cxx_library": ":" + name + "_cxx",
            "loom_library": ":" + name + "_loom",
            "model": ":" + name + "_model",
            "opaque_module": ":" + name + "_opaque_module",
        },
        **kwargs
    )

def _cxx_import_precedes_link_impl(env, targets):
    cxx_import = _expect_action_executable(
        env,
        targets.cxx_library,
        "LoomCxxImport",
        "loom-import-cxx",
    )
    env.expect.that_collection(cxx_import.argv).contains("--options=fp-model=fast")
    cxx_link = _expect_action_executable(
        env,
        targets.cxx_library,
        "LoomLibrary",
        "loom-link",
    )
    env.expect.that_collection(cxx_link.argv).not_contains("--input-format=cxx")
    _expect_action_input_suffix(env, cxx_link, "model_provider.cxx.loombc")
    _reject_action_input_suffix(env, cxx_link, "model_provider.cxx")
    _reject_action_input_suffix(env, cxx_link, "model_provider.h")
    _expect_action_executable(env, targets.loom_library, "LoomLibrary", "loom-link")
    for target in [targets.loom_library, targets.opaque_module]:
        for action in target[TestingAspectInfo].actions:
            if action.mnemonic == "LoomCxxImport":
                env.fail("Loom-only library unexpectedly imports C/C++")
    model = _expect_model_linker(env, targets.model, "loom-link")
    _expect_generated_action_input_suffix(env, model, "model_provider.cxx")

def loom_model_rules_test_suite(name):
    test_suite(name = name, tests = [
        _test_sources,
        _test_modules,
        _test_bytecode_is_an_original_source,
        _test_external_files_fail,
        _test_conflicting_importers_fail,
        _test_cxx_import_precedes_link,
    ])
