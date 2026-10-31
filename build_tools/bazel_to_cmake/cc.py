# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile interfaces resolved from configured BUILD declarations.

Only source-owned libraries participate in this graph. Authored and imported
native dependencies remain target edges, evaluated by CMake in their owning
scope. No native target properties or generator expressions are interpreted.
"""

import hashlib
import os
import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import native_cc
from conditions import Option


@dataclass(frozen=True)
class Interface:
    # Public include directories, with source/binary roots already qualified.
    includes: tuple[str, ...] = ()
    # Public system include directories.
    system_includes: tuple[str, ...] = ()
    # Public preprocessor definitions; native expressions remain unevaluated.
    defines: tuple[str, ...] = ()
    # Public compiler options, excluding the library's private options.
    options: tuple[str, ...] = ()
    # Link options inherited by final executables and shared libraries.
    link_options: tuple[str, ...] = ()
    # Native dependencies at the boundary of the source-owned graph.
    native: tuple[str, ...] = ()

    def with_dependencies(self, dependencies):
        return Interface(
            **{
                name: tuple(
                    dict.fromkeys(
                        value
                        for interface in (self, *dependencies)
                        for value in getattr(interface, name)
                    )
                )
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class Library:
    # Canonical native target name used by this source declaration's consumers.
    name: str
    # Ordered direct dependencies, including opaque native target expressions.
    dependencies: tuple[str, ...]
    # Compile and link requirements contributed by this declaration.
    interface: Interface
    # Native compilation inputs, excluding source-only headers.
    compiled: tuple[str, ...]
    # Compiled objects retained regardless of ordinary archive symbol selection.
    alwayslink: bool
    # Generated sources and headers required before this declaration compiles.
    generated: tuple[str, ...]
    # Repository-relative package owning this declaration's artifact directory.
    package: str

    @cached_property
    def requires_cxx_linker(self):
        """Whether this library contributes C++ objects to a final link."""
        return any(
            re.search(r"\.(cc|cpp|cxx|c[+][+]|C)$", source) for source in self.compiled
        )


@dataclass
class Graph:
    # Effective native output settings; parent/toolchain overrides remain native.
    configuration: dict[str, str]
    # Native directory owning each source package and its private imports.
    native_scopes: dict[str, str] = field(default_factory=dict)
    # Source-owned declarations indexed by native name and default-package alias.
    libraries: dict[str, Library] = field(default_factory=dict)
    # Resolved transitive interfaces, computed once after source collection.
    resolved: dict[str, Interface] = field(default_factory=dict)
    # Opaque requirements reachable in each library's native directory scope.
    public_native: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Private interfaces preserving opaque dependencies in their native scope.
    native_interfaces: dict[tuple[str, str], str] = field(default_factory=dict)
    # Interfaces awaiting emission in the source package that first owns them.
    pending_native_interfaces: dict[str, list[tuple[str, str]]] = field(
        default_factory=dict
    )
    # Active source dependency chain for a useful BUILD-cycle diagnostic.
    resolving: list[str] = field(default_factory=list)
    # Ordered source link closure; dependents precede their providers.
    links: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Generated compilation prerequisites inherited through source dependencies.
    inputs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Declared generator target owning each fully qualified output path.
    producers: dict[str, str] = field(default_factory=dict)
    # Final link inputs interned once instead of restated for every consumer.
    responses: dict[tuple[str, ...], str] = field(default_factory=dict)

    def response(self, artifacts):
        if artifacts not in self.responses:
            key = hashlib.sha256("\0".join(artifacts).encode()).hexdigest()[:20]
            self.responses[artifacts] = key
        return self.responses[artifacts]

    def response_files(self):
        outputs = {}
        if not self.responses:
            return outputs
        binary_root = Path(self.configuration["CMAKE_BINARY_DIR"])
        owned_root = self.configuration["IREE_BINARY_DIR"]
        relative_root = os.path.relpath(owned_root, binary_root).replace("\\", "/")
        for artifacts, key in self.responses.items():
            output = binary_root / "CMakeFiles/iree/link" / (key + ".rsp")
            content = "".join(
                '"' + artifact.replace("${IREE_BINARY_DIR}", relative_root) + '"\n'
                for artifact in artifacts
            )
            outputs[output] = content
        return outputs

    def add(self, library, aliases=()):
        for name in (library.name, *aliases):
            if name in self.libraries:
                raise ValueError(f"Duplicate source library: {name}")
            self.libraries[name] = library

    def native_interface(self, package, dependency):
        key = (self.native_scopes[package], dependency)
        if key not in self.native_interfaces:
            name = (
                "_iree_native_"
                + hashlib.sha256("\0".join(key).encode()).hexdigest()[:16]
            )
            self.native_interfaces[key] = name
            self.pending_native_interfaces.setdefault(package, []).append(
                (name, dependency)
            )
        return self.native_interfaces[key]

    def take_native_interfaces(self, package):
        return self.pending_native_interfaces.pop(package, ())

    def resolve(self, name):
        library = self.libraries.get(name)
        if library is None:
            return Interface(native=(name,))
        if library.name in self.resolved:
            return self.resolved[library.name]
        if library.name in self.resolving:
            raise ValueError(
                "Source library dependency cycle: "
                + " -> ".join([*self.resolving, library.name])
            )
        self.resolving.append(library.name)
        inherited = [
            self._dependency(library, dependency) for dependency in library.dependencies
        ]
        interface = replace(
            library.interface,
            native=tuple(
                self.native_interface(library.package, dependency)
                for dependency in library.interface.native
            ),
        )
        # Source compilation can cross native directories through the private
        # interfaces. Public/exported interfaces retain ordinary native edges,
        # so private imported names never escape their owning lookup scope.
        public_native = list(library.interface.native)
        for dependency in library.dependencies:
            if self._shares_native_scope(library, dependency):
                provider = self.libraries[dependency]
                public_native.extend(self.public_native[provider.name])
            else:
                public_native.append(dependency)
        self.public_native[library.name] = tuple(dict.fromkeys(public_native))
        result = interface.with_dependencies(inherited)
        self.resolving.pop()
        self.resolved[library.name] = result
        return result

    def inherited(self, name):
        library = self.libraries[name]
        return Interface().with_dependencies(
            [self._dependency(library, dep) for dep in library.dependencies]
        )

    def _shares_native_scope(self, library, dependency):
        provider = self.libraries.get(dependency)
        return provider is not None and (
            self.native_scopes[provider.package] == self.native_scopes[library.package]
        )

    def _dependency(self, library, dependency):
        if dependency in self.libraries:
            return self.resolve(dependency)
        return Interface(native=(self.native_interface(library.package, dependency),))

    def link_dependencies(self, name):
        library = self.libraries[name]
        self.resolve(name)
        return tuple(
            dict.fromkeys(
                [
                    f"$<LINK_ONLY:{dependency}>"
                    if self._shares_native_scope(library, dependency)
                    else dependency
                    for dependency in library.dependencies
                ]
                + list(self.public_native[library.name])
            )
        )

    def link_inputs(self, name):
        if name not in self.links:
            library = self.libraries[name]
            self.links[name] = self._ordered_links(
                library.package, (*library.dependencies, *library.interface.native)
            )
        return self.links[name]

    def _ordered_links(self, package, dependencies):
        ordered = {}
        for dependency in dependencies:
            provider = self.libraries.get(dependency)
            if provider:
                sequence = (provider.name, *self.link_inputs(dependency))
            else:
                sequence = (self.native_interface(package, dependency),)
            for item in sequence:
                # A shared provider moves after every dependent that reaches
                # it. Child closures already retain this invariant.
                ordered.pop(item, None)
                ordered[item] = None
        return tuple(ordered)

    def executable_inputs(self, package, dependencies):
        interfaces, generated = [], {}
        for dependency in dependencies:
            provider = self.libraries.get(dependency)
            if provider:
                interfaces.append(self.resolve(dependency))
                generated.update(dict.fromkeys(self.generated_inputs(dependency)))
            else:
                interfaces.append(
                    Interface(native=(self.native_interface(package, dependency),))
                )
        artifacts, native = [], []
        cxx = False
        for dependency in self._ordered_links(package, dependencies):
            provider = self.libraries.get(dependency)
            if provider:
                if provider.compiled:
                    target = provider.name.replace("::", "_")
                    cxx |= provider.requires_cxx_linker
                    if provider.alwayslink:
                        native.append(
                            f"$<GENEX_EVAL:$<TARGET_PROPERTY:{target},IREE_CC_OBJECTS>>"
                        )
                    elif self.archive_pattern is None:
                        native.append(provider.name)
                    else:
                        artifacts.append(self.archive(provider))
            else:
                native.append(dependency)
        return (
            Interface().with_dependencies(interfaces),
            tuple(artifacts),
            tuple(native),
            tuple(generated),
            cxx,
        )

    @cached_property
    def archive_pattern(self):
        """The configured toolchain's ordinary static product layout."""
        if (
            # Recursive generators need native target edges to enter each
            # provider's build. Ninja can follow the owned artifact file edges.
            self.configuration.get("CMAKE_GENERATOR") != "Ninja"
            or not (
                Option("MSVC").evaluate(self.configuration)
                or self.configuration.get("CMAKE_SYSTEM_NAME") == "Linux"
            )
            or self.configuration.get("CMAKE_CONFIGURATION_TYPES")
            or any(
                key.startswith("CMAKE_ARCHIVE_OUTPUT_DIRECTORY")
                for key in self.configuration
            )
            or any(
                key.startswith("CMAKE_") and key.endswith("_POSTFIX") and value
                for key, value in self.configuration.items()
            )
            or any(
                key in self.configuration
                for key in (
                    "CMAKE_STATIC_LIBRARY_PREFIX_C",
                    "CMAKE_STATIC_LIBRARY_PREFIX_CXX",
                    "CMAKE_STATIC_LIBRARY_SUFFIX_C",
                    "CMAKE_STATIC_LIBRARY_SUFFIX_CXX",
                )
            )
        ):
            return None
        prefix = self.configuration["CMAKE_STATIC_LIBRARY_PREFIX"]
        suffix = self.configuration["CMAKE_STATIC_LIBRARY_SUFFIX"]
        return "${IREE_BINARY_DIR}/%s/" + prefix + "%s" + suffix

    def archive(self, library):
        name = library.name.replace("::", "_")
        if self.archive_pattern is None:
            return f"$<TARGET_LINKER_FILE:{name}>"
        return self.archive_pattern % (library.package, name)

    def generated_inputs(self, name):
        if name not in self.inputs:
            library = self.libraries[name]
            inputs = dict.fromkeys(library.generated)
            for dependency in library.dependencies:
                if dependency in self.libraries:
                    inputs.update(dict.fromkeys(self.generated_inputs(dependency)))
            self.inputs[name] = tuple(inputs)
        return self.inputs[name]


class BuildFileFunctions:
    """Source C/C++ rules and their native projections."""

    def _emit_executable(
        self,
        invocation,
        name,
        deps,
        target_compatible_with,
        *,
        enabled=True,
        implicit=True,
    ):
        if (
            self._libraries is None
            or not enabled
            or not all(
                self._select_condition(label).evaluate(self._configuration)
                for label in target_compatible_with or []
            )
        ):
            self._converter.body += invocation
            return
        target = self._convert_single_target(self._current_target_label(name))
        namespace = target.rsplit("::", 1)[0]
        dependencies = tuple(
            namespace + native if native.startswith("::") else native
            for native in sorted(
                {
                    native
                    for dependency in deps or []
                    for native in self._convert_target(dependency)
                    if native
                }
            )
        )
        if implicit:
            dependencies += ("${IREE_IMPLICIT_DEFS_CC_DEPS}",)

        def render():
            package = self._current_package()
            interface, artifacts, native, generated, cxx = (
                self._libraries.executable_inputs(package, dependencies)
            )
            interfaces = native_cc.interfaces(
                self._libraries.take_native_interfaces(package)
            )
            return (
                invocation
                + interfaces
                + native_cc.executable_inputs(
                    target.replace("::", "_"),
                    interface,
                    artifacts,
                    native,
                    generated,
                    self._libraries.producers,
                    response=self._libraries.response(artifacts) if artifacts else None,
                    cxx=cxx,
                    msvc=Option("MSVC").evaluate(self._configuration),
                )
            )

        self._converter.defer(render)

    def _native_source_paths(self, values):
        paths, generated = [], []
        package = self._current_package()
        for value in sorted(values or []):
            if (
                value.startswith("$")
                or os.path.isabs(value)
                and not value.startswith("//")
            ):
                path = value
                is_generated = value.startswith(
                    ("${IREE_BINARY_DIR}", "${CMAKE_CURRENT_BINARY_DIR}")
                )
            else:
                if value.startswith("//"):
                    relative = value[2:].removeprefix(":").replace(":", "/")
                else:
                    relative = package + "/" + value.removeprefix(":")
                if not os.path.splitext(relative)[1]:
                    relative += ".stamp"
                is_generated = not os.path.exists(
                    os.path.join(self._repo_root, relative)
                )
                root = "IREE_BINARY_DIR" if is_generated else "IREE_SOURCE_DIR"
                path = f"${{{root}}}/{relative}"
            paths.append(path)
            if is_generated:
                generated.append(path)
        return tuple(paths), tuple(generated)

    def _record_library_interface(
        self,
        name,
        srcs,
        deps,
        defines,
        includes,
        system_includes,
        linkopts,
        alwayslink,
        shared,
        testonly,
        target_compatible_with,
        cmake_package,
        headers,
    ):
        if self._libraries is None:
            return None
        if shared or Option("BUILD_SHARED_LIBS").evaluate(self._configuration):
            return None
        if testonly and not Option("IREE_BUILD_TESTS").evaluate(self._configuration):
            return None
        if any(
            not self._select_condition(label).evaluate(self._configuration)
            for label in target_compatible_with or []
        ):
            return None
        label = self._current_target_label(name)
        # Explicit mappings may name a native adapter instead of the declaration
        # (for example the installed library's public header interface).
        if not cmake_package and label in self._targets._explicit_target_mapping:
            return None
        native_name = (
            cmake_package + "::" + name
            if cmake_package
            else self._convert_single_target(label)
        )
        package = self._current_package()
        source_root = "${IREE_SOURCE_DIR}/" + package
        binary_root = "${IREE_BINARY_DIR}/" + package

        def qualify(value):
            return value.replace("${_IREE_PACKAGE_SOURCE_DIR}", source_root).replace(
                "${CMAKE_CURRENT_BINARY_DIR}", binary_root
            )

        namespace = native_name.rsplit("::", 1)[0]
        dependencies = tuple(
            namespace + dependency if dependency.startswith("::") else dependency
            for dependency in sorted(
                {
                    native
                    for dependency in deps or []
                    for native in self._convert_target(dependency)
                    if native
                }
            )
        )
        compiled = tuple(
            source
            for source in self._native_source_paths(srcs)[0]
            if not re.search(
                r"\.(h|hh|h[+][+]|hm|hpp|hxx|H|HPP|in|txx|inc|inl)$", source
            )
        )
        interface = Interface(
            includes=tuple(
                f"$<BUILD_INTERFACE:{root}/{directory}>"
                for directory in includes or []
                for root in (source_root, binary_root)
            ),
            system_includes=tuple(qualify(value) for value in system_includes or []),
            defines=tuple(qualify(value) for value in defines or []),
            options=("${IREE_INTERFACE_COPTS}",) if compiled and not alwayslink else (),
            link_options=tuple(
                qualify(value)
                for value in self._convert_linkopt_location_args(linkopts or [])
            ),
            native=("${IREE_THREADS_DEPS}", "${IREE_DEFAULT_LINK_LIBRARIES}")
            if compiled
            else (),
        )
        aliases = (namespace,) if name == namespace.rsplit("::", 1)[-1] else ()
        self._libraries.add(
            Library(
                native_name,
                dependencies,
                interface,
                compiled,
                bool(alwayslink),
                (
                    *self._native_source_paths(srcs)[1],
                    *self._native_source_paths(headers)[1],
                ),
                package,
            ),
            aliases,
        )
        return native_name

    def cc_library(
        self,
        name,
        hdrs=None,
        textual_hdrs=None,
        srcs=None,
        linkstamp=None,
        copts=None,
        conlyopts=None,
        cxxopts=None,
        cxx_standard=None,
        cxx_features=None,
        defines=None,
        data=None,
        deps=None,
        testonly=None,
        linkopts=None,
        includes=None,
        system_includes=None,
        alwayslink=None,
        shared=None,
        sanitizer_suppressions=None,
        target_compatible_with=None,
        cmake_package=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        if cxx_standard is not None and cxx_standard not in ("c++17", "c++20", "c++23"):
            raise ValueError(f"Unsupported C++ standard: {cxx_standard}")
        cxx_standard_block = self._convert_string_arg_block(
            "CXX_STANDARD", cxx_standard[3:] if cxx_standard else None, quote=False
        )
        cxx_features_block = self._convert_string_list_block(
            "CXX_FEATURES", cxx_features
        )
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        textual_hdrs_block = self._convert_srcs_block(
            textual_hdrs, block_name="TEXTUAL_HDRS"
        )
        if linkstamp:
            srcs = (srcs or []) + [linkstamp]
        srcs_block, platform_srcs_block = self._convert_platform_select_srcs(name, srcs)
        copts_block, platform_copts_block = self._convert_compile_options(
            name, copts, conlyopts, cxxopts
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        alwayslink_block = self._convert_option_block("ALWAYSLINK", alwayslink)
        shared_block = self._convert_option_block("SHARED", shared)
        duplicate_safe_block = self._convert_option_block(
            "LINKABLE_MORE_THAN_ONCE",
            "LINKABLE_MORE_THAN_ONCE" in (kwargs.get("tags") or []),
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts, expand_locations=True
        )
        includes_block = self._convert_includes_block(includes)
        system_includes_block = self._convert_string_list_block(
            "SYSTEM_INCLUDES", system_includes
        )
        native_library = self._record_library_interface(
            name,
            srcs,
            deps,
            defines,
            includes,
            system_includes,
            linkopts,
            alwayslink,
            shared,
            testonly,
            target_compatible_with,
            cmake_package,
            (hdrs or []) + (textual_hdrs or []),
        )

        sanitizer_suppressions_block = self._convert_sanitizer_suppressions_block(
            sanitizer_suppressions
        )

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_srcs_block:
            self._converter.body += platform_srcs_block
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        if platform_linkopts_block:
            self._converter.body += platform_linkopts_block

        def render():
            if native_library:
                resolved = self._libraries.resolve(native_library)
                inherited = self._libraries.inherited(native_library)
                sources, _ = self._native_source_paths(srcs)
                headers, generated_headers = self._native_source_paths(
                    (hdrs or []) + (textual_hdrs or [])
                )
                options = list(copts or [])
                for language, values in (("C", conlyopts), ("CXX", cxxopts)):
                    options.extend(
                        f"$<$<COMPILE_LANGUAGE:{language}>:{value}>"
                        for value in values or []
                    )
                body = native_cc.library(
                    self._libraries.libraries[native_library],
                    resolved,
                    inherited,
                    self._libraries.link_dependencies(native_library),
                    sources=sources,
                    headers=headers,
                    generated_headers=generated_headers,
                    generated_inputs=self._libraries.generated_inputs(native_library),
                    producers=self._libraries.producers,
                    compile_options=tuple(options),
                    cxx_standard=cxx_standard[3:] if cxx_standard else None,
                    cxx_features=tuple(cxx_features or []),
                    data=tuple(self._convert_data_locations(data) or ()),
                    sanitizer_suppressions=tuple(
                        value
                        for sanitizer, label in sorted(
                            (sanitizer_suppressions or {}).items()
                        )
                        for value in (
                            sanitizer,
                            self._sanitizer_suppression_name(label),
                        )
                    ),
                    alwayslink=alwayslink,
                    testonly=testonly,
                    linkable_more_than_once="LINKABLE_MORE_THAN_ONCE"
                    in (kwargs.get("tags") or []),
                    msvc=Option("MSVC").evaluate(self._configuration),
                    wasm=self._configuration.get("CMAKE_SYSTEM_PROCESSOR") == "wasm32",
                )
                return (
                    native_cc.interfaces(
                        self._libraries.take_native_interfaces(self._current_package())
                    )
                    + body
                )
            interface_blocks = defines_block + includes_block + system_includes_block
            return (
                f"iree_cc_library(\n"
                f"{name_block}"
                + self._convert_string_arg_block("PACKAGE", cmake_package)
                + f"{copts_block}"
                f"{cxx_standard_block}"
                f"{cxx_features_block}"
                f"{hdrs_block}"
                f"{textual_hdrs_block}"
                f"{srcs_block}"
                f"{data_block}"
                f"{deps_block}"
                f"{interface_blocks}"
                f"{testonly_block}"
                f"{alwayslink_block}"
                f"{shared_block}"
                f"{duplicate_safe_block}"
                f"{sanitizer_suppressions_block}"
                f"{linkopts_block}"
                f"  PUBLIC\n)\n\n"
            )

        if native_library:
            self._converter.defer(render)
        else:
            self._converter.body += render()
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_library(self, **kwargs):
        self.cc_library(**kwargs)

    def iree_cc_library_group(self, **kwargs):
        self.cc_library(**kwargs)

    def cc_test(
        self,
        name,
        hdrs=None,
        srcs=None,
        copts=None,
        conlyopts=None,
        cxxopts=None,
        cxx_standard=None,
        cxx_features=None,
        defines=None,
        linkopts=None,
        data=None,
        deps=None,
        timeout=None,
        args=None,
        env=None,
        tags=None,
        includes=None,
        group=None,
        resource_group=None,
        sanitizer_suppressions=None,
        target_compatible_with=None,
        _cmake_rule="iree_cc_test",
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        # Extract resource_group from tags if provided via Bazel tag convention.
        # The iree_runtime_cc_test .bzl macro encodes it as "resource_group:name".
        if not resource_group and tags:
            for tag in tags:
                if tag.startswith("resource_group:"):
                    resource_group = tag[len("resource_group:") :]
                    break
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        if cxx_standard is not None and cxx_standard not in ("c++17", "c++20", "c++23"):
            raise ValueError(f"Unsupported C++ standard: {cxx_standard}")
        cxx_standard_block = self._convert_string_arg_block(
            "CXX_STANDARD", cxx_standard[3:] if cxx_standard else None, quote=False
        )
        cxx_features_block = self._convert_string_list_block(
            "CXX_FEATURES", cxx_features
        )
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        srcs_block, platform_srcs_block = self._convert_platform_select_srcs(name, srcs)
        copts_block, platform_copts_block = self._convert_compile_options(
            name, copts, conlyopts, cxxopts
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts, expand_locations=True
        )
        data_block = self._convert_data_list_block(data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        args_block = self._convert_string_list_block(
            "ARGS",
            self._convert_cc_test_args(args),
            sort=False,
        )
        env_block = self._convert_string_list_block(
            "ENV", self._convert_test_env(env), sort=False
        )
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)
        includes_block = self._convert_includes_block(includes)
        group_block = self._convert_string_arg_block("GROUP", group)
        resource_group_block = self._convert_string_arg_block(
            "RESOURCE_GROUP", resource_group, quote=False
        )
        sanitizer_suppressions_block = self._convert_sanitizer_suppressions_block(
            sanitizer_suppressions
        )

        self._emit_platform_guard_begin(target_compatible_with)
        did_emit_testdata_guard = self._emit_optional_local_cts_testdata_guard_begin(
            deps
        )
        if platform_srcs_block:
            self._converter.body += platform_srcs_block
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        if platform_linkopts_block:
            self._converter.body += platform_linkopts_block
        invocation = (
            f"{_cmake_rule}(\n"
            f"{name_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{cxx_standard_block}"
            f"{cxx_features_block}"
            f"{defines_block}"
            f"{linkopts_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{args_block}"
            f"{env_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f"{includes_block}"
            f"{group_block}"
            f"{resource_group_block}"
            f"{sanitizer_suppressions_block}"
            f")\n\n"
        )
        self._emit_executable(
            invocation,
            name,
            deps,
            target_compatible_with,
            enabled=_cmake_rule == "iree_cc_test"
            and Option("IREE_BUILD_TESTS").evaluate(self._configuration or {}),
            implicit=False,
        )
        self._emit_optional_local_cts_testdata_guard_end(did_emit_testdata_guard)
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_test(self, **kwargs):
        self.cc_test(**kwargs)

    def cc_binary(
        self,
        name,
        srcs=None,
        data=None,
        deps=None,
        copts=None,
        conlyopts=None,
        cxxopts=None,
        defines=None,
        linkopts=None,
        linkshared=None,
        testonly=None,
        includes=None,
        target_compatible_with=None,
        _cmake_rule="iree_cc_binary",
        _cmake_options=(),
        _native_linking=False,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        self._target_file_labels.add(self._current_target_label(name))
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        copts_block, platform_copts_block = self._convert_compile_options(
            name, copts, conlyopts, cxxopts
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts, expand_locations=True
        )
        srcs_block, platform_srcs_block = self._convert_platform_select_srcs(name, srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        includes_block = self._convert_includes_block(includes)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_srcs_block:
            self._converter.body += platform_srcs_block
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        if platform_linkopts_block:
            self._converter.body += platform_linkopts_block
        if linkshared:
            self._converter.body += (
                f"iree_cc_library(\n"
                f"{name_block}"
                f"{copts_block}"
                f"{srcs_block}"
                f"{data_block}"
                f"{deps_block}"
                f"{defines_block}"
                f"{linkopts_block}"
                f"{testonly_block}"
                f"{includes_block}"
                f"  SHARED\n"
                f")\n\n"
            )
            self._emit_platform_guard_end(target_compatible_with)
            return
        invocation = (
            f"{_cmake_rule}(\n"
            + "".join(f"  {option}\n" for option in _cmake_options)
            + f"{name_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{linkopts_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"{includes_block}"
            f")\n\n"
        )
        self._emit_executable(
            invocation,
            name,
            deps,
            target_compatible_with,
            enabled=not _native_linking
            and (
                not testonly
                or Option("IREE_BUILD_TESTS").evaluate(self._configuration or {})
            ),
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_binary(self, **kwargs):
        self.cc_binary(**kwargs)

    def iree_cc_fuzz(
        self,
        name,
        srcs=None,
        data=None,
        deps=None,
        copts=None,
        conlyopts=None,
        cxxopts=None,
        defines=None,
        linkopts=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block, platform_srcs_block = self._convert_platform_select_srcs(name, srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        copts_block, platform_copts_block = self._convert_compile_options(
            name, copts, conlyopts, cxxopts
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts, expand_locations=True
        )
        labels_block = self._convert_string_list_block("LABELS", tags)

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_srcs_block:
            self._converter.body += platform_srcs_block
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        if platform_linkopts_block:
            self._converter.body += platform_linkopts_block
        invocation = (
            f"iree_cc_fuzz(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{linkopts_block}"
            f"{labels_block}"
            f")\n\n"
        )
        self._emit_executable(
            invocation,
            name,
            deps,
            target_compatible_with,
            enabled=Option("IREE_BUILD_TESTS").evaluate(self._configuration or {})
            and Option("IREE_ENABLE_FUZZING").evaluate(self._configuration or {}),
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_compiler_cc_fuzz(self, **kwargs):
        self.iree_cc_fuzz(**kwargs)

    def cc_binary_benchmark(
        self,
        name,
        srcs=None,
        data=None,
        deps=None,
        copts=None,
        defines=None,
        linkopts=None,
        args=None,
        tags=None,
        resource_group=None,
        testonly=True,
        target_compatible_with=None,
        # unused
        size="small",
        timeout=None,
        visibility=None,
    ):
        if self._should_skip_target(tags=tags):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_data_list_block(data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        linkopts_block = self._convert_string_list_block("LINKOPTS", linkopts)
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_test_location_args(args), sort=False
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        labels_block = self._convert_string_list_block("LABELS", tags)
        resource_group_block = self._convert_string_arg_block(
            "RESOURCE_GROUP", resource_group, quote=False
        )

        self._emit_platform_guard_begin(target_compatible_with)
        invocation = (
            f"iree_cc_binary_benchmark(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{linkopts_block}"
            f"{args_block}"
            f"{testonly_block}"
            f"{labels_block}"
            f"{resource_group_block}"
            f")\n\n"
        )
        self._emit_executable(
            invocation,
            name,
            deps,
            target_compatible_with,
            enabled=Option("IREE_BUILD_BENCHMARKS").evaluate(self._configuration or {})
            and (
                not testonly
                or Option("IREE_BUILD_TESTS").evaluate(self._configuration or {})
            ),
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_benchmark(self, **kwargs):
        self.cc_binary_benchmark(**kwargs)
