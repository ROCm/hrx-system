# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model-card selected providers and native deployment artifacts."""

load(":loom_linking.bzl", "LoomLibraryInfo")

_LINK_TOOLCHAIN = Label("//loom/build_tools/bazel:link_toolchain_type")

LoomModelInfo = provider(
    doc = "Analysis-owned model-card program declarations, without payload access.",
    fields = {
        "card": "Repository-relative path of the authored card.",
        "programs": "Named program sets with generated inputs/libraries.",
    },
)

def _repository_path(ctx, file):
    if file.owner.workspace_name != ctx.label.workspace_name:
        fail("%s: model composition requires a local repository file, got %s" % (ctx.label, file.owner))
    if file.is_directory:
        fail("%s: model composition requires individual files, got %s" % (ctx.label, file.short_path))
    return file.short_path.removeprefix("../%s/" % ctx.label.workspace_name)

def _mapped_artifacts(declarations):
    result = {}
    for target, name in declarations.items():
        artifacts = target[DefaultInfo].files.to_list()
        if len(artifacts) != 1 or artifacts[0].is_directory:
            fail("%s must supply exactly one file for %s" % (target.label, name))
        if name in result:
            fail("conflicting model path: %s" % name)
        result[name] = artifacts[0]
    return result

def _program_composition(ctx):
    if not ctx.attr.inputs and not ctx.attr.libraries:
        declarations = ctx.attr.files | ctx.attr.data | ctx.attr.assets
        if len(declarations) != len(ctx.attr.files) + len(ctx.attr.data) + len(ctx.attr.assets):
            fail("files, data, and assets must have disjoint targets")
        return struct(
            files = _mapped_artifacts(ctx.attr.files),
            data = _mapped_artifacts(ctx.attr.data),
            normalization_formats = {},
            programs = {},
        )
    if ctx.attr.files or ctx.attr.data:
        fail("provider inputs/libraries and explicit files/data are separate composition modes")
    if not ctx.attr.programs:
        fail("provider composition requires an explicit programs key")
    primary = [target[LoomLibraryInfo] for target in ctx.attr.inputs]
    libraries = [target[LoomLibraryInfo] for target in ctx.attr.libraries]
    primary_labels = {info.component.label: True for info in primary}
    dependencies = depset(
        direct = [info.component for info in libraries],
        transitive = [info.transitive_components for info in primary + libraries],
        order = "preorder",
    ).to_list()
    groups = [
        ("inputs", [info.component for info in primary]),
        ("libraries", [component for component in dependencies if component.label not in primary_labels]),
    ]
    program = {"inputs": [], "libraries": []}
    files = {}
    normalization_formats = {}
    data = {}
    providers = {}
    for role, components in groups:
        for component in components:
            if component.label.workspace_name != ctx.label.workspace_name:
                fail("%s: external repository provider %s is not supported in model composition" % (ctx.label, component.label))
            auxiliary = []
            for file in component.data:
                path = _repository_path(ctx, file)
                if path in data and data[path] != file:
                    fail("conflicting model auxiliary file: %s" % path)
                data[path] = file
                auxiliary.append(path)
            source_records = zip(component.sources, component.normalized_sources, component.normalized_formats) if ctx.attr.form == "sources" else [(component.module, component.module, "loombc")]
            for file, normalized_file, normalized_format in source_records:
                path = _repository_path(ctx, file)
                provider = {"path": path}
                if ctx.attr.form == "module":
                    provider["format"] = "loombc"
                else:
                    if component.input_format:
                        provider["format"] = "loombc" if component.input_format == "loom" and file.extension == "loombc" else component.input_format
                    if component.inputopts:
                        if any(["$(" in option for option in component.inputopts]):
                            fail("%s: source model import options must be portable literals, not Bazel location expressions" % component.label)
                        provider["options"] = list(component.inputopts)
                if auxiliary:
                    provider["data"] = auxiliary
                selected_file = file if ctx.attr.encoding == "preserve" else normalized_file
                if path in providers:
                    if providers[path] != provider or files[path] != selected_file:
                        fail("conflicting model provider: %s" % path)
                    continue
                providers[path] = provider
                files[path] = selected_file
                if ctx.attr.encoding != "preserve":
                    normalization_formats[path] = normalized_format
                program[role].append(provider if len(provider) > 1 else path)
    return struct(
        files = files,
        data = {path: file for path, file in data.items() if path not in files},
        normalization_formats = normalization_formats,
        programs = {ctx.attr.programs: program},
    )

def _loom_model_impl(ctx, link_tool = None):
    composition = _program_composition(ctx)
    preserve = ctx.attr.encoding == "preserve"
    card = ctx.actions.declare_file(ctx.label.name + "/model.json")
    deployment = ctx.actions.declare_file(ctx.label.name + "/serve.json")
    outputs = [card, deployment]
    runfiles = [card, deployment]
    inputs = [ctx.file.card]
    files = {}
    data = {}
    assets = {}
    asset_files = _mapped_artifacts(ctx.attr.assets)
    declarations = composition.files | composition.data | asset_files
    if len(declarations) != len(composition.files) + len(composition.data) + len(asset_files):
        fail("files, data, and assets must have disjoint paths")
    for name, artifact in declarations.items():
        if not name or any([part in ["", ".", ".."] for part in name.split("/")]) or "\\" in name or ":" in name:
            fail("expected normalized relative model path: %s" % name)
        if name.split("/")[0] in ["model.json", "serve.json"]:
            fail("conflicting model path: %s" % name)

        # Normalization retains the card's logical provider path even when an
        # upstream importer produced the bytes under a generated module path.
        staged = artifact
        if not preserve and name not in asset_files:
            staged = ctx.actions.declare_file(ctx.label.name + ".input/" + name)
            ctx.actions.symlink(output = staged, target_file = artifact)
            inputs.append(staged)
        output = ctx.actions.declare_file(ctx.label.name + "/" + name)
        runfiles.append(output)
        if name in composition.files:
            files[name] = {"input": staged.path, "output": output.path}
            if name in composition.normalization_formats:
                files[name]["format"] = composition.normalization_formats[name]
            if preserve:
                ctx.actions.symlink(output = output, target_file = artifact)
            else:
                outputs.append(output)
        else:
            ctx.actions.symlink(output = output, target_file = artifact)
            if name in composition.data:
                data[name] = output.path
            else:
                assets[name] = output.path
    plan = ctx.actions.declare_file(ctx.label.name + ".plan.json")
    ctx.actions.write(plan, json.encode({
        "assets": assets,
        "card": ctx.file.card.path,
        "data": data,
        "encoding": ctx.attr.encoding,
        "files": files,
        "input_root": card.dirname + ".input",
        "name": ctx.attr.model_name,
        "output_card": card.path,
        "output_deployment": deployment.path,
        "partition": ctx.attr.partition if ctx.attr.partition else None,
        "preset": ctx.attr.preset,
        "program_declarations": composition.programs.get(ctx.attr.programs),
        "programs": ctx.attr.programs if ctx.attr.programs else None,
    }))
    arguments = ["--plan", plan.path]
    tools = []
    if not preserve:
        if link_tool == None:
            fail("provider normalization requires the Loom linker toolchain")
        arguments += ["--linker", link_tool.files_to_run.executable.path]
        tools.append(link_tool.files_to_run)
    ctx.actions.run(
        executable = ctx.executable._builder,
        arguments = arguments,
        inputs = depset(inputs + [plan]),
        tools = tools,
        outputs = outputs,
        mnemonic = "LoomModel",
        progress_message = "Building model preset %s" % ctx.label,
    )

    # Test identities include authored choices and externally provisioned assets.
    # These metadata dependencies are not provider inputs or publication files:
    # a normalized package needs its own lock for the bytes it publishes.
    test_inputs = [ctx.file.card]
    if ctx.file.lockfile:
        test_inputs.append(ctx.file.lockfile)
    return ([LoomModelInfo(card = _repository_path(ctx, ctx.file.card), programs = composition.programs)] if composition.programs else []) + [
        DefaultInfo(files = depset([deployment]), runfiles = ctx.runfiles(files = runfiles + test_inputs)),
        OutputGroupInfo(model_card = depset([card]), model_files = depset(runfiles)),
    ]

def _loom_preserved_model_impl(ctx):
    return _loom_model_impl(ctx)

def _loom_normalized_model_impl(ctx):
    return _loom_model_impl(ctx, ctx.toolchains[_LINK_TOOLCHAIN].tool)

_LOOM_MODEL_DOC = """Packages one model-card selection from declared compiler providers.

Provider composition uses inputs and libraries of loom_library targets. Each
provider owns its sources, data, importer settings, and direct dependencies;
analysis derives the complete ordered closure with repository-relative paths.
form selects original sources or merged modules for the complete composition.
Sources may themselves be bytecode. Primary providers reached again as
dependencies retain their primary role and appear only once.

LoomModelInfo exposes the generated inputs/libraries for card synchronization
through cquery without executing any build action. The package action checks
that the authored card matches that same composition. Origins, presets,
execution directories, variants, and runtime assets remain card-owned.

Provider composition accepts local repository files and literal importer
options. External repositories and location-expanded importer options fail
explicitly rather than exporting execroot paths. Explicit files/data mappings
remain available as a separate input mode for independently authored cards;
normalized explicit files must already be Loom modules.
"""

def _loom_model_attrs():
    return {
        "assets": attr.label_keyed_string_dict(allow_files = True, doc = "Optional runtime assets, such as small fixture parameters or tokenizers, mapped to card paths. Omitted assets retain their acquisition descriptors for loom-fetch. Supplied assets are runfiles, never linker inputs."),
        "card": attr.label(mandatory = True, allow_single_file = [".json"]),
        "data": attr.label_keyed_string_dict(allow_files = True, doc = "Exactly the selected providers' auxiliary files, mapped to card paths. These are compiler inputs and runtime runfiles, separate from parameter/tokenizer acquisition."),
        "encoding": attr.string(default = "bc", values = ["text", "bc", "preserve"], doc = "Normalize providers to Loom text/bytecode, or preserve their bytes, formats, and import options."),
        "files": attr.label_keyed_string_dict(allow_files = True, doc = "Exactly the selected inputs and libraries, mapped to card paths; preserved as symlinks or normalized with the configured linker."),
        "form": attr.string(default = "sources", values = ["sources", "module"], doc = "Select original declared files or merged modules for every provider."),
        "inputs": attr.label_list(providers = [LoomLibraryInfo], doc = "Ordered primary providers. Their direct and transitive dependencies become libraries automatically."),
        "libraries": attr.label_list(providers = [LoomLibraryInfo], doc = "Additional runtime-selected library roots; their dependency closure is included automatically."),
        "lockfile": attr.label(allow_single_file = True, doc = "Authored acquisition lock carried as a test dependency. Changes invalidate dependent tests without reading runtime assets or recompiling providers. This is not the publication lock of normalized outputs."),
        "model_name": attr.string(mandatory = True),
        "partition": attr.string(doc = "Executable partition override within the selected variant."),
        "preset": attr.string(mandatory = True),
        "programs": attr.string(),
        "_builder": attr.label(default = Label("//loom/py/loom/model:build"), executable = True, cfg = "exec"),
    }

_loom_preserved_model = rule(
    implementation = _loom_preserved_model_impl,
    attrs = _loom_model_attrs(),
    doc = _LOOM_MODEL_DOC,
)

_loom_normalized_model = rule(
    implementation = _loom_normalized_model_impl,
    attrs = _loom_model_attrs(),
    doc = _LOOM_MODEL_DOC,
    toolchains = [_LINK_TOOLCHAIN],
)

def loom_model(name, encoding = "bc", **kwargs):
    """Packages one model-card selection from declared compiler providers."""
    model_rule = _loom_preserved_model if encoding == "preserve" else _loom_normalized_model
    model_rule(name = name, encoding = encoding, **kwargs)

def _loom_model_documentation_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".md")
    arguments = ctx.actions.args()
    arguments.add("--card", ctx.file.card)
    arguments.add("--output", output)
    inputs = [ctx.file.card]
    if ctx.file.overview:
        arguments.add("--overview", ctx.file.overview)
        inputs.append(ctx.file.overview)
    if ctx.file.lockfile:
        arguments.add("--lockfile", ctx.file.lockfile)
        inputs.append(ctx.file.lockfile)
    if ctx.attr.card_url:
        arguments.add("--card-url", ctx.attr.card_url)
    ctx.actions.run(
        executable = ctx.executable._renderer,
        arguments = [arguments],
        inputs = inputs,
        outputs = [output],
        mnemonic = "LoomModelDocumentation",
        progress_message = "Documenting model card %s" % ctx.label,
    )
    return [DefaultInfo(files = depset([output]), runfiles = ctx.runfiles(files = [output]))]

loom_model_documentation = rule(
    implementation = _loom_model_documentation_impl,
    doc = "Generates a model reference from metadata alone, without acquiring or compiling payloads.",
    attrs = {
        "card": attr.label(mandatory = True, allow_single_file = [".json"]),
        "card_url": attr.string(doc = "Published card URL used to resolve card-relative source links."),
        "lockfile": attr.label(allow_single_file = True, doc = "Optional lock covering all card choices, supplying immutable source revisions and output commitments."),
        "overview": attr.label(allow_single_file = [".md"], doc = "Publisher-authored Markdown overview included before the generated contract. Use absolute publication links so the result stands alone."),
        "_renderer": attr.label(default = Label("//loom/py/loom/model:documentation_generate"), executable = True, cfg = "exec"),
    },
)
