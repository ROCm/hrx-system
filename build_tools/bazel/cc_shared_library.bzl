# Copyright 2021 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared composition preserving the selected producers' link metadata.

Derived from rules_cc's shared-library composition. Graph/provider identities
and partition/ordering logic remain those of the selected rules_cc module;
whole-archive selection changes metadata without creating new backend actions.
"""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_helper.bzl", "cc_helper")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_cc//cc/common:cc_shared_library_info.bzl", "CcSharedLibraryInfo")
load("@rules_cc//cc/common:semantics.bzl", "semantics")

# The shared graph and library constructors must use the selected module's
# provider identities, including when it is unpatched in an external consumer.
# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private:graph_node_info.bzl", "GraphNodeInfo")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/link:create_library_to_link.bzl", "make_library_to_link")

# buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/rules_impl:cc_shared_library.bzl", "graph_structure_aspect")

# buildifier: disable=bzl-visibility
load(
    "@rules_cc//cc/private/rules_impl:cc_shared_library_impl.bzl",
    _add_unused_dynamic_deps = "add_unused_dynamic_deps",
    _build_exports_map_from_only_dynamic_deps = "build_exports_map_from_only_dynamic_deps",
    _build_link_once_static_libs_map = "build_link_once_static_libs_map",
    _merge_cc_shared_library_infos = "merge_cc_shared_library_infos",
    _separate_static_and_dynamic_link_libraries = "separate_static_and_dynamic_link_libraries",
    _sort_linker_inputs = "sort_linker_inputs",
    _throw_linked_but_not_exported_errors = "throw_linked_but_not_exported_errors",
)

def _create_linker_context(linker_inputs):
    return cc_common.create_linking_context(
        linker_inputs = depset(linker_inputs, order = "topological"),
    )

def _is_dynamic_only(library_to_link):
    if (library_to_link.static_library == None and
        library_to_link.pic_static_library == None and
        (library_to_link.objects == None or len(library_to_link.objects) == 0) and
        (library_to_link.pic_objects == None or len(library_to_link.pic_objects) == 0)):
        return True
    return False

def _wrap_static_library_with_alwayslink(linker_input):
    new_libraries_to_link = []
    for old_library_to_link in linker_input.libraries:
        if _is_dynamic_only(old_library_to_link):
            new_libraries_to_link.append(old_library_to_link)
            continue
        new_library_to_link = make_library_to_link(
            static_library = old_library_to_link.static_library,
            objects = old_library_to_link.objects,
            pic_static_library = old_library_to_link.pic_static_library,
            pic_objects = old_library_to_link.pic_objects,
            _library_identifier = old_library_to_link._library_identifier,
            _contains_objects = old_library_to_link._contains_objects,
            _must_keep_debug = old_library_to_link._must_keep_debug,
            _disable_whole_archive = old_library_to_link._disable_whole_archive,
            _lto_compilation_context = old_library_to_link._lto_compilation_context,
            _pic_lto_compilation_context = old_library_to_link._pic_lto_compilation_context,
            _shared_non_lto_backends = old_library_to_link._shared_non_lto_backends,
            _pic_shared_non_lto_backends = old_library_to_link._pic_shared_non_lto_backends,
            alwayslink = True,
        )
        new_libraries_to_link.append(new_library_to_link)

    return cc_common.create_linker_input(
        owner = linker_input.owner,
        libraries = depset(direct = new_libraries_to_link),
        user_link_flags = linker_input.user_link_flags,
        additional_inputs = depset(direct = linker_input.additional_inputs),
        linkstamps = depset(linker_input.linkstamps),
    )

def _check_if_target_under_path(value, pattern):
    if pattern.workspace_name != value.workspace_name:
        return False
    if pattern.name == "__pkg__":
        return pattern.package == value.package
    if pattern.name == "__subpackages__":
        return _same_package_or_above(pattern, value)

    return pattern.package == value.package and pattern.name == value.name

def _check_if_target_should_be_exported_with_filter(target, current_label, exports_filter):
    for export_filter in exports_filter:
        export_filter_label = current_label.relative(export_filter)
        if _check_if_target_under_path(target, export_filter_label):
            return True

    return False

def _contains_code_to_link(linker_input):
    """Whether this input owns static code, rather than only link metadata."""
    for library in linker_input.libraries:
        if (library.static_library != None or
            library.pic_static_library != None or
            len(library.objects) or len(library.pic_objects)):
            return True

    return False

def _find_top_level_linker_input_labels(
        deps_graph_nodes,
        linker_inputs_to_be_linked_statically_map,
        targets_to_be_linked_dynamically_set):
    top_level_linker_input_labels_set = {}
    nodes_to_check = list(deps_graph_nodes)

    seen_nodes_set = {}
    for i in range(2147483647):
        if i == len(nodes_to_check):
            break

        node = nodes_to_check[i]
        must_add_children = False
        node_str_owners = []
        for owner in node.owners:
            owner_str = str(owner)
            node_str_owners.append(owner_str)
            if owner_str in linker_inputs_to_be_linked_statically_map:
                must_add_children = True
                for linker_input in linker_inputs_to_be_linked_statically_map[owner_str]:
                    if _contains_code_to_link(linker_input):
                        top_level_linker_input_labels_set[owner_str] = True
                        must_add_children = False
                        break
            elif owner_str not in targets_to_be_linked_dynamically_set:
                # This can happen when there was a target in the graph that exported other libraries'
                # linker_inputs but didn't contribute any linker_input of its own.
                must_add_children = True

        node_key = tuple(node_str_owners)
        if must_add_children and node_key not in seen_nodes_set:
            nodes_to_check.extend(node.children)
            seen_nodes_set[node_key] = True

    return top_level_linker_input_labels_set

def _filter_inputs(
        ctx,
        deps,
        transitive_exports,
        link_once_static_libs_map):
    curr_link_once_static_libs_set = {}

    deps_root_tree_nodes = []
    dependency_linker_inputs_sets = []
    direct_deps_set = {}
    for dep in deps:
        direct_deps_set[str(dep.label)] = True
        dependency_linker_inputs_sets.append(dep[CcInfo].linking_context.linker_inputs)
        deps_root_tree_nodes.append(dep[GraphNodeInfo])

    dependency_linker_inputs = depset(transitive = dependency_linker_inputs_sets, order = "topological").to_list()

    can_be_linked_dynamically = {}
    for linker_input in dependency_linker_inputs:
        owner = str(linker_input.owner)
        if owner in transitive_exports:
            can_be_linked_dynamically[owner] = True

    # The targets_to_be_linked_statically_map points to whether the target to
    # be linked statically can be linked more than once.
    # Entries in unused_dynamic_linker_inputs will be marked None if they are
    # used
    (
        targets_to_be_linked_statically_map,
        targets_to_be_linked_dynamically_set,
        topologically_sorted_labels,
        unused_dynamic_linker_inputs,
    ) = _separate_static_and_dynamic_link_libraries(
        ctx.attr.dynamic_deps,
        deps_root_tree_nodes,
        can_be_linked_dynamically,
    )

    linker_inputs_to_be_linked_statically_map = {}
    for linker_input in dependency_linker_inputs:
        owner = str(linker_input.owner)
        if owner in targets_to_be_linked_statically_map:
            linker_inputs_to_be_linked_statically_map.setdefault(owner, []).append(linker_input)

    top_level_linker_input_labels_set = _find_top_level_linker_input_labels(
        deps_root_tree_nodes,
        linker_inputs_to_be_linked_statically_map,
        targets_to_be_linked_dynamically_set,
    )

    # We keep track of precompiled_only_dynamic_libraries, so that we can add
    # them to runfiles.
    precompiled_only_dynamic_libraries = []
    exports = {}
    linker_inputs_seen = {}
    linker_inputs_count = 0
    label_to_linker_inputs = {}

    def _add_linker_input_to_dict(owner, linker_input):
        label_to_linker_inputs.setdefault(owner, []).append(linker_input)

    # We use this dictionary to give an error if a target containing only
    # precompiled dynamic libraries is placed directly in roots. If such a
    # precompiled dynamic library is needed it would be because a target in the
    # parallel cc_library graph actually needs it. Therefore the precompiled
    # dynamic library should be made a dependency of that cc_library instead.
    dynamic_only_roots = {}
    linked_statically_but_not_exported = {}
    for linker_input in dependency_linker_inputs:
        stringified_linker_input = cc_helper.stringify_linker_input(linker_input)
        if stringified_linker_input in linker_inputs_seen:
            continue
        linker_inputs_seen[stringified_linker_input] = True
        owner = str(linker_input.owner)
        if owner in targets_to_be_linked_dynamically_set:
            unused_dynamic_linker_inputs[transitive_exports[owner].owner] = None

            # Link the library in this iteration dynamically,
            # transitive_exports contains the artifacts produced by a
            # cc_shared_library
            _add_linker_input_to_dict(linker_input.owner, transitive_exports[owner])
            linker_inputs_count += 1
        elif owner in targets_to_be_linked_statically_map:
            if not linker_input.libraries:
                # A header-only owner may still supply native flags, scripts,
                # or linkstamps. It owns no code subject to link-once checks.
                if linker_input.user_link_flags or linker_input.additional_inputs or linker_input.linkstamps:
                    _add_linker_input_to_dict(linker_input.owner, linker_input)
                    linker_inputs_count += 1
                continue
            if owner in link_once_static_libs_map:
                # We are building a dictionary that will allow us to give
                # proper errors for libraries that have been linked multiple
                # times elsewhere but haven't been exported. The values in the
                # link_once_static_libs_map dictionary are the
                # cc_shared_library targets. In this iteration we know of at
                # least one target (i.e. owner) which is being linked
                # statically by the cc_shared_library
                # link_once_static_libs_map[owner] but is not being exported
                linked_statically_but_not_exported.setdefault(link_once_static_libs_map[owner], []).append(owner)

            dynamic_only_libraries = []
            static_libraries = []
            for library in linker_input.libraries:
                if _is_dynamic_only(library):
                    dynamic_only_libraries.append(library)
                else:
                    static_libraries.append(library)

            if len(dynamic_only_libraries):
                precompiled_only_dynamic_libraries.extend(dynamic_only_libraries)
                if not len(static_libraries):
                    if owner in direct_deps_set:
                        dynamic_only_roots[owner] = True
                    _add_linker_input_to_dict(linker_input.owner, linker_input)
                    linker_inputs_count += 1
                    continue
            if len(static_libraries) and owner in dynamic_only_roots:
                dynamic_only_roots.pop(owner)

            linker_input_to_be_linked_statically = linker_input
            if owner in top_level_linker_input_labels_set:
                linker_input_to_be_linked_statically = _wrap_static_library_with_alwayslink(linker_input)
            if _check_if_target_should_be_exported_with_filter(
                linker_input.owner,
                ctx.label,
                ctx.attr.exports_filter,
            ):
                exports[owner] = True

            _add_linker_input_to_dict(linker_input.owner, linker_input_to_be_linked_statically)
            linker_inputs_count += 1

            if not targets_to_be_linked_statically_map[owner]:
                curr_link_once_static_libs_set[owner] = True

    if dynamic_only_roots:
        message = ("Do not place libraries which only contain a " +
                   "precompiled dynamic library in roots. The following " +
                   "libraries only have precompiled dynamic libraries:\n")
        for dynamic_only_root in dynamic_only_roots:
            message += dynamic_only_root + "\n"
        fail(message)

    linker_inputs_count += _add_unused_dynamic_deps(ctx, unused_dynamic_linker_inputs, _add_linker_input_to_dict, topologically_sorted_labels, link_indirect_deps = False)

    linker_inputs = _sort_linker_inputs(
        topologically_sorted_labels,
        label_to_linker_inputs,
        linker_inputs_count,
    )

    _throw_linked_but_not_exported_errors(linked_statically_but_not_exported)
    return (exports, linker_inputs, curr_link_once_static_libs_set.keys(), precompiled_only_dynamic_libraries, dependency_linker_inputs)

def _same_package_or_above(label_a, label_b):
    if label_a.workspace_name != label_b.workspace_name:
        return False
    package_a_tokenized = label_a.package.split("/")
    package_b_tokenized = label_b.package.split("/")
    if len(package_b_tokenized) < len(package_a_tokenized):
        return False

    if package_a_tokenized[0] != "":
        for i in range(len(package_a_tokenized)):
            if package_a_tokenized[i] != package_b_tokenized[i]:
                return False

    return True

def _cc_shared_library_impl(ctx):
    deps = ctx.attr.deps
    if not deps:
        fail("Shared libraries require at least one implementation in deps", attr = "deps")

    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features + ["force_no_whole_archive"],
        unsupported_features = ctx.disabled_features,
    )

    merged_cc_shared_library_infos = _merge_cc_shared_library_infos(ctx)

    # Small performance tweak to avoid flattening merged_cc_shared_library_infos twice:
    merged_cc_shared_library_infos_list = merged_cc_shared_library_infos.to_list()
    exports_map = _build_exports_map_from_only_dynamic_deps(merged_cc_shared_library_infos_list)
    for export in deps:
        # Do not check for overlap between targets matched by the current
        # rule's exports_filter and what is in exports_map. A library in roots
        # will have to be linked in statically into the current rule with 100%
        # guarantee and it will also have to be exported. Therefore, we must
        # check it's not already exported by a different shared library. On the
        # other hand, a library in the transitive closure of the current rule
        # may be matched by the exports_filter but if it's already exported by
        # a dynamic_dep then it won't be linked statically (therefore not give
        # an error either) in the current target. The rule will intentionally
        # not throw an error in these cases.
        if str(export.label) in exports_map:
            fail("Trying to export a library already exported by a different shared library: " +
                 str(export.label))

    link_once_static_libs_map = _build_link_once_static_libs_map(merged_cc_shared_library_infos_list)

    (exports, linker_inputs, curr_link_once_static_libs_set, precompiled_only_dynamic_libraries, public_linker_inputs) = _filter_inputs(
        ctx,
        deps,
        exports_map,
        link_once_static_libs_map,
    )

    linking_context = _create_linker_context(linker_inputs)

    cc_runtimes_deps = cc_helper.get_cc_runtimes(ctx, True)
    runtimes_linking_contexts = cc_helper.get_linking_contexts_from_deps(cc_runtimes_deps)

    user_link_flags = []
    for user_link_flag in ctx.attr.user_link_flags:
        user_link_flags.append(ctx.expand_location(user_link_flag, targets = ctx.attr.additional_linker_inputs))

    main_output = None
    if ctx.attr.shared_lib_name:
        main_output = ctx.actions.declare_file(ctx.attr.shared_lib_name)

    additional_inputs = []
    additional_outputs = []
    link_variables = {}
    additional_output_groups = {}

    pdb_file = None
    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "generate_pdb_file"):
        if ctx.attr.shared_lib_name:
            pdb_file = ctx.actions.declare_file(paths.replace_extension(ctx.attr.shared_lib_name, ".pdb"))
        else:
            pdb_file = ctx.actions.declare_file(ctx.label.name + ".pdb")
        additional_outputs.append(pdb_file)
        additional_output_groups["pdb_file"] = depset([pdb_file])

    if cc_common.is_enabled(feature_configuration = feature_configuration, feature_name = "targets_windows"):
        object_files = []
        for linker_input in linking_context.linker_inputs.to_list():
            for library in linker_input.libraries:
                if library.pic_static_library != None:
                    if library.pic_objects != None:
                        object_files.extend(library.pic_objects)
                elif library.static_library != None:
                    if library.objects != None:
                        object_files.extend(library.objects)

        generated_def_file = cc_helper.generate_def_file(
            ctx,
            ctx.file._def_parser,
            object_files,
            ctx.label.name,
            cc_toolchain,
            feature_configuration,
        )
        if generated_def_file != None:
            additional_output_groups["def_file"] = depset([generated_def_file])
        custom_win_def_file = ctx.file.win_def_file
        win_def_file = cc_helper.get_windows_def_file_for_linking(ctx, custom_win_def_file, generated_def_file, feature_configuration)
        link_variables["def_file_path"] = win_def_file.path
        additional_inputs.append(win_def_file)

    additional_inputs.extend(ctx.files.additional_linker_inputs)
    linking_outputs = cc_common.link(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        linking_contexts = [linking_context] + runtimes_linking_contexts,
        user_link_flags = user_link_flags,
        additional_inputs = additional_inputs,
        name = ctx.label.name,
        output_type = "dynamic_library",
        main_output = main_output,
        variables_extension = link_variables,
        additional_outputs = additional_outputs,
    )

    runfiles_files = []
    if linking_outputs.library_to_link.resolved_symlink_dynamic_library != None:
        runfiles_files.append(linking_outputs.library_to_link.resolved_symlink_dynamic_library)

    # This is different to cc_binary(linkshared=1). Bazel never handles the
    # linking implicitly for a cc_binary(linkshared=1) but it does so for a cc_shared_library,
    # for which it will use the symlink in the solib directory. If we don't add it, a dependent
    # linked against it would fail.
    runfiles_files.append(linking_outputs.library_to_link.dynamic_library)
    runfiles = ctx.runfiles(
        files = runfiles_files,
    )
    for dep in ctx.attr.dynamic_deps:
        runfiles = runfiles.merge(dep[DefaultInfo].data_runfiles)

    precompiled_only_dynamic_libraries_runfiles = []
    for precompiled_dynamic_library in precompiled_only_dynamic_libraries:
        # precompiled_dynamic_library.dynamic_library could be None if the library to link just contains
        # an interface library which is valid if the actual library is obtained from the system.
        if precompiled_dynamic_library.dynamic_library != None:
            precompiled_only_dynamic_libraries_runfiles.append(precompiled_dynamic_library.dynamic_library)

    runfiles = runfiles.merge(ctx.runfiles(files = precompiled_only_dynamic_libraries_runfiles))

    for export in deps:
        exports[str(export.label)] = True

    if not semantics.get_experimental_link_static_libraries_once(ctx):
        curr_link_once_static_libs_set = {}

    library = []
    if linking_outputs.library_to_link.resolved_symlink_dynamic_library != None:
        library.append(linking_outputs.library_to_link.resolved_symlink_dynamic_library)
    else:
        library.append(linking_outputs.library_to_link.dynamic_library)

    interface_library = []
    if linking_outputs.library_to_link.resolved_symlink_interface_library != None:
        interface_library.append(linking_outputs.library_to_link.resolved_symlink_interface_library)
    elif linking_outputs.library_to_link.interface_library != None:
        interface_library.append(linking_outputs.library_to_link.interface_library)
    else:
        interface_library = library

    return [
        DefaultInfo(
            files = depset(library),
            runfiles = runfiles,
        ),
        OutputGroupInfo(
            main_shared_library_output = depset(library),
            interface_library = depset(interface_library),
            **additional_output_groups
        ),
        CcSharedLibraryInfo(
            dynamic_deps = merged_cc_shared_library_infos,
            exports = exports.keys(),
            link_once_static_libs = curr_link_once_static_libs_set,
            linker_input = cc_common.create_linker_input(
                owner = ctx.label,
                libraries = depset([linking_outputs.library_to_link] + precompiled_only_dynamic_libraries),
                # CcInfo link options are transitive usage requirements. Inline
                # header code still needs them after its implementation owner
                # becomes dynamic. Preserve their declared inputs as well;
                # neither linkstamps nor this DSO's private user_link_flags and
                # additional_linker_inputs belong in a consumer's link.
                user_link_flags = [flag for input in public_linker_inputs for flag in input.user_link_flags],
                additional_inputs = depset([file for input in public_linker_inputs for file in input.additional_inputs]),
            ),
        ),
    ]

def _initializer(**kwargs):
    filters = kwargs.get("exports_filter", [])
    if type(filters) == "list":
        return kwargs | {"exports_filter": [str(native.package_relative_label(value)) for value in filters]}
    return kwargs

iree_cc_shared_library = rule(
    implementation = _cc_shared_library_impl,
    initializer = _initializer,
    doc = """Links implementation roots and substitutes exported dynamic dependencies.

Direct deps are whole-archived without recreating their compiler backends or
discarding linkstamps. Export ownership, duplicate-link checks, ordering and
consumer providers follow rules_cc, allowing ordinary cc_binary/cc_test users.
Transitive CcInfo link options and their inputs remain public usage requirements;
user_link_flags and additional_linker_inputs affect only this shared library.
""",
    attrs = {
        "additional_linker_inputs": attr.label_list(allow_files = True, doc = "Declared linker scripts and other non-code inputs."),
        "deps": attr.label_list(providers = [CcInfo], aspects = [graph_structure_aspect], doc = "Implementation roots exported by this shared library."),
        "dynamic_deps": attr.label_list(providers = [CcSharedLibraryInfo], doc = "Shared libraries providing transitive implementations."),
        "exports_filter": attr.string_list(doc = "Additional exported owner labels, packages, or subpackages."),
        "shared_lib_name": attr.string(doc = "Output filename, or the toolchain's default when empty."),
        "user_link_flags": attr.string_list(doc = "Linker flags; locations expand against additional_linker_inputs."),
        "win_def_file": attr.label(allow_single_file = [".def"], doc = "Explicit Windows exports."),
        "_def_parser": semantics.get_def_parser(),
    },
    provides = [CcSharedLibraryInfo],
    toolchains = use_cc_toolchain() + semantics.get_runtimes_toolchain(),
    fragments = ["cpp"] + semantics.additional_fragments(),
)
