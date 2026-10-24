# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Model-card selected providers and native deployment artifacts."""

_LINK_TOOLCHAIN = Label("//loom/build_tools/bazel:link_toolchain_type")

def _loom_model_impl(ctx):
    preserve = ctx.attr.encoding == "preserve"
    card = ctx.actions.declare_file(ctx.label.name + "/model.json")
    deployment = ctx.actions.declare_file(ctx.label.name + "/serve.json")
    outputs = [card, deployment]
    runfiles = [card, deployment]
    inputs = [ctx.file.card]
    files = {}
    data = {}
    assets = {}
    names = {}
    declarations = ctx.attr.files | ctx.attr.data | ctx.attr.assets
    if len(declarations) != len(ctx.attr.files) + len(ctx.attr.data) + len(ctx.attr.assets):
        fail("files, data, and assets must have disjoint targets")
    for target, name in declarations.items():
        if not name or any([part in ["", ".", ".."] for part in name.split("/")]) or "\\" in name or ":" in name:
            fail("expected normalized relative model path: %s" % name)
        if name.split("/")[0] in ["model.json", "serve.json"] or name in names:
            fail("conflicting model path: %s" % name)
        names[name] = True
        artifacts = target[DefaultInfo].files.to_list()
        if len(artifacts) != 1 or artifacts[0].is_directory:
            fail("%s must supply exactly one file for %s" % (target.label, name))
        artifact = artifacts[0]

        # Importers resolve auxiliary inputs relative to the provider's physical
        # path. Staging preserves the card namespace without copying input bytes.
        staged = artifact
        if not preserve and target not in ctx.attr.assets:
            staged = ctx.actions.declare_file(ctx.label.name + ".input/" + name)
            ctx.actions.symlink(output = staged, target_file = artifact)
            inputs.append(staged)
        output = ctx.actions.declare_file(ctx.label.name + "/" + name)
        runfiles.append(output)
        if target in ctx.attr.files:
            files[name] = {"input": staged.path, "output": output.path}
            if preserve:
                ctx.actions.symlink(output = output, target_file = artifact)
            else:
                outputs.append(output)
        else:
            ctx.actions.symlink(output = output, target_file = artifact)
            if target in ctx.attr.data:
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
        "programs": ctx.attr.programs if ctx.attr.programs else None,
    }))
    arguments = ["--plan", plan.path]
    tools = []
    if not preserve:
        linker = ctx.toolchains[_LINK_TOOLCHAIN].tool
        arguments += ["--linker", linker.files_to_run.executable.path]
        tools.append(linker.files_to_run)
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
    return [
        DefaultInfo(files = depset([deployment]), runfiles = ctx.runfiles(files = runfiles + test_inputs)),
        OutputGroupInfo(model_card = depset([card]), model_files = depset(runfiles)),
    ]

loom_model = rule(
    implementation = _loom_model_impl,
    attrs = {
        "assets": attr.label_keyed_string_dict(allow_files = True, doc = "Optional runtime assets, such as small fixture parameters or tokenizers, mapped to card paths. Omitted assets retain their acquisition descriptors for loom-fetch. Supplied assets are runfiles, never linker inputs."),
        "card": attr.label(mandatory = True, allow_single_file = [".json"]),
        "data": attr.label_keyed_string_dict(allow_files = True, doc = "Exactly the selected providers' auxiliary files, mapped to card paths. These are compiler inputs and runtime runfiles, separate from parameter/tokenizer acquisition."),
        "encoding": attr.string(default = "bc", values = ["text", "bc", "preserve"], doc = "Normalize providers to Loom text/bytecode, or preserve their bytes, formats, and import options."),
        "files": attr.label_keyed_string_dict(allow_files = True, doc = "Exactly the selected inputs and libraries, mapped to card paths; preserved as symlinks or normalized with the configured linker."),
        "lockfile": attr.label(allow_single_file = True, doc = "Authored acquisition lock carried as a test dependency. Changes invalidate dependent tests without reading runtime assets or recompiling providers. This is not the publication lock of normalized outputs."),
        "model_name": attr.string(mandatory = True),
        "partition": attr.string(doc = "Executable partition override within the selected variant."),
        "preset": attr.string(mandatory = True),
        "programs": attr.string(),
        "_builder": attr.label(default = Label("//loom/py/loom/model:build"), executable = True, cfg = "exec"),
    },
    toolchains = [_LINK_TOOLCHAIN],
)

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
