# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Native products from source-resolved C/C++ declarations.

Toolchain settings and authored interfaces remain native expressions. Source
classification, paths, aliases, and the BUILD interface are already resolved.
"""


def arguments(values):
    return " ".join(f'"{value}"' for value in values)


def command(name, *parts):
    return f"{name}({' '.join(parts)})\n"


def generated_dependencies(name, generated, producers):
    lines = []
    producer_targets = dict.fromkeys(
        producers[source] for source in generated if source in producers
    )
    if producer_targets:
        lines.append(command("add_dependencies", name, " ".join(producer_targets)))
    for source in generated:
        if source not in producers:
            lines.append(
                command("iree_generated_output_add_consumer", arguments((source, name)))
            )
    return "".join(lines)


def response(key, artifacts):
    return command(
        "set", f"_IREE_LINK_INPUTS_{key}", arguments(artifacts), "PARENT_SCOPE"
    )


def executable_inputs(
    name, interface, artifacts, native, generated, producers, *, response, cxx, msvc
):
    """Projects a final product's source closure without walking native libraries."""
    lines = []
    for operation, scope, values in (
        ("target_include_directories", "PRIVATE", interface.includes),
        ("target_include_directories", "SYSTEM PRIVATE", interface.system_includes),
        ("target_compile_definitions", "PRIVATE", interface.defines),
        ("target_compile_options", "PRIVATE", interface.options),
        ("target_link_options", "PRIVATE", interface.link_options),
    ):
        if values:
            lines.append(command(operation, name, scope, arguments(values)))
    links = list(native)
    if artifacts:
        relative = f"CMakeFiles/iree/link/{response}.rsp"
        response_file = "${CMAKE_BINARY_DIR}/" + relative
        if msvc:
            lines.append(
                command(
                    "target_link_options",
                    name,
                    "PRIVATE",
                    arguments(('SHELL:@\\"' + relative + '\\"',)),
                )
            )
        else:
            links.insert(0, '"-Wl,@\\"' + relative + '\\""')
        lines.append(
            command(
                "set_property",
                "TARGET",
                name,
                "APPEND PROPERTY LINK_DEPENDS",
                f'"${{_IREE_LINK_INPUTS_{response}}}" "{response_file}"',
            )
        )
    if cxx:
        lines.append(
            command("set_property", "TARGET", name, "PROPERTY LINKER_LANGUAGE CXX")
        )
    lines.append(
        command(
            "set_property",
            "TARGET",
            name,
            "PROPERTY LINK_LIBRARIES",
            " ".join(
                (
                    *links,
                    "${IREE_DEFAULT_LINK_LIBRARIES}",
                    "${IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES}",
                )
            ),
        )
    )
    lines.append(generated_dependencies(name, generated, producers))
    return "".join(lines)


def library(
    declaration,
    resolved,
    inherited,
    links,
    *,
    sources,
    headers,
    generated_headers,
    generated_inputs,
    producers,
    compile_options,
    cxx_standard,
    cxx_features,
    data,
    sanitizer_suppressions,
    alwayslink,
    testonly,
    linkable_more_than_once,
    msvc,
):
    """Emits a static or interface library owned by a configured BUILD rule."""
    name = declaration.name.replace("::", "_")
    compiled = bool(declaration.compiled)
    compile_target = name + ".objects" if compiled and alwayslink else name
    interface = not compiled or alwayslink
    scope = "INTERFACE" if interface else "PUBLIC"
    dependencies = (*declaration.dependencies, "${IREE_IMPLICIT_DEFS_CC_DEPS}")
    lines = []

    def emit(operation, *parts):
        lines.append(command(operation, *parts))

    def property(target, key, *values):
        emit("set_property", "TARGET", target, "PROPERTY", key, arguments(values))

    def requirements(operation, target, visibility, values):
        if values:
            emit(operation, target, visibility, arguments(values))

    if compiled:
        emit("add_library", compile_target, "OBJECT" if alwayslink else "STATIC")
        emit(
            "target_sources", compile_target, "PRIVATE", arguments((*sources, *headers))
        )
        emit("iree_set_source_object_names", arguments(declaration.compiled))
        if alwayslink:
            emit("add_library", name, "INTERFACE")
            emit("add_dependencies", name, compile_target)
        elif msvc:
            emit(
                "set_target_properties",
                name,
                "PROPERTIES COMPILE_PDB_NAME compile COMPILE_PDB_OUTPUT_DIRECTORY",
                f'"${{CMAKE_CURRENT_BINARY_DIR}}/CMakeFiles/{name}.dir"',
            )
        requirements(
            "target_compile_options",
            compile_target,
            "PRIVATE",
            ("${IREE_DEFAULT_COPTS}", *compile_options, *inherited.options),
        )
        emit(
            "iree_set_cxx_options",
            compile_target,
            arguments((cxx_standard or "", *cxx_features)),
        )
        property(compile_target, "FOLDER", "${IREE_IDE_FOLDER}")
        property(name, "IREE_CC_OBJECT_TARGET", compile_target)
        property(name, "IREE_CC_OBJECTS", f"$<TARGET_OBJECTS:{compile_target}>")
        property(
            name,
            "INTERFACE_IREE_TRANSITIVE_OBJECTS",
            f"$<GENEX_EVAL:$<TARGET_PROPERTY:{name},IREE_CC_OBJECTS>>",
        )
    else:
        emit("add_library", name, "INTERFACE")
        if generated_headers:
            emit("target_sources", name, "PRIVATE", arguments(generated_headers))

    # Opaque imports keep their owning native scope and late-bound requirements.
    native_links = [*links, "${IREE_IMPLICIT_DEFS_CC_DEPS}"]
    if compiled:
        native_links.extend(("${IREE_THREADS_DEPS}", "${IREE_DEFAULT_LINK_LIBRARIES}"))
    if compiled and alwayslink:
        native_links.insert(
            0, f'"$<GENEX_EVAL:$<TARGET_PROPERTY:{name},IREE_CC_OBJECTS>>"'
        )
    emit("target_link_libraries", name, "INTERFACE", " ".join(native_links))
    if compiled:
        emit(
            "set_property",
            "TARGET",
            compile_target,
            "PROPERTY LINK_LIBRARIES",
            " ".join((*resolved.native, "${IREE_IMPLICIT_DEFS_CC_DEPS}")),
        )
    requirements(
        "target_include_directories",
        name,
        scope,
        (
            "$<BUILD_INTERFACE:${IREE_SOURCE_DIR};${IREE_BINARY_DIR};"
            + ";".join(resolved.includes)
            + ">",
        ),
    )
    requirements(
        "target_include_directories",
        name,
        "SYSTEM " + scope,
        ("$<BUILD_INTERFACE:" + ";".join(resolved.system_includes) + ">",),
    )
    requirements("target_compile_definitions", name, scope, resolved.defines)
    requirements("target_compile_options", name, "INTERFACE", resolved.options)
    requirements("target_link_options", name, "INTERFACE", resolved.link_options)
    requirements(
        "target_link_options",
        name,
        "INTERFACE" if interface else "PRIVATE",
        ("${IREE_DEFAULT_LINKOPTS}",),
    )
    if compiled and alwayslink:
        for operation, visibility, key in (
            ("target_include_directories", "PUBLIC", "INTERFACE_INCLUDE_DIRECTORIES"),
            (
                "target_include_directories",
                "SYSTEM PUBLIC",
                "INTERFACE_SYSTEM_INCLUDE_DIRECTORIES",
            ),
            ("target_compile_definitions", "PUBLIC", "INTERFACE_COMPILE_DEFINITIONS"),
        ):
            requirements(
                operation,
                compile_target,
                visibility,
                (f"$<TARGET_PROPERTY:{name},{key}>",),
            )

    # Authored shared/unified products still consume these native interfaces.
    # The source graph owns ordinary closure resolution above this boundary.
    emit("_iree_cc_library_add_object_deps", name, " ".join(dependencies))
    lines.append(generated_dependencies(compile_target, generated_inputs, producers))
    if not testonly:
        emit("iree_install_targets", "TARGETS", name, "HDRS", arguments(headers))
    namespace, leaf = declaration.name.rsplit("::", 1)
    aliases = [declaration.name]
    if namespace.rsplit("::", 1)[-1] == leaf:
        aliases.append(namespace)
    for alias in aliases:
        emit("iree_add_alias_library", alias, name)
    metadata_dependencies = dependencies + (
        ("${IREE_THREADS_DEPS}", "${IREE_DEFAULT_LINK_LIBRARIES}") if compiled else ()
    )
    emit(
        "set_property",
        "TARGET",
        name,
        "PROPERTY IREE_CC_DEPS",
        " ".join(metadata_dependencies),
    )
    property(name, "IREE_CC_LINKOPTS", ";".join(declaration.interface.link_options))
    property(
        name,
        "IREE_CC_LINKABLE_MORE_THAN_ONCE",
        "TRUE" if linkable_more_than_once else "FALSE",
    )
    if data:
        emit(
            "iree_add_data_dependencies",
            "NAME",
            name,
            "DATA",
            " ".join(data),
            "OUT_TARGET_DATA _IREE_NATIVE_DATA",
        )
    emit(
        "iree_target_sanitizer_suppressions",
        name,
        "DEPS",
        " ".join(dependencies),
        "${_IREE_NATIVE_DATA}" if data else "",
        "SUPPRESSIONS",
        " ".join(sanitizer_suppressions),
    )
    return "".join(lines) + "\n"
