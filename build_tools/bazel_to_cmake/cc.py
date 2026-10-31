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

import re
from dataclasses import dataclass, field

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
    # Closest authored native directory; its private imports stay in this scope.
    native_scope: str
    # Ordered direct dependencies, including opaque native target expressions.
    dependencies: tuple[str, ...]
    # Compile and link requirements contributed by this declaration.
    interface: Interface


@dataclass
class Graph:
    # Source-owned declarations indexed by native name and default-package alias.
    libraries: dict[str, Library] = field(default_factory=dict)
    # Resolved transitive interfaces, computed once after source collection.
    resolved: dict[str, Interface] = field(default_factory=dict)
    # Active source dependency chain for a useful BUILD-cycle diagnostic.
    resolving: list[str] = field(default_factory=list)

    def add(self, library, aliases=()):
        for name in (library.name, *aliases):
            if name in self.libraries:
                raise ValueError(f"Duplicate source library: {name}")
            self.libraries[name] = library

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
        result = library.interface.with_dependencies(inherited)
        self.resolving.pop()
        self.resolved[library.name] = result
        return result

    def inherited(self, name):
        library = self.libraries[name]
        return Interface().with_dependencies(
            [self._dependency(library, dep) for dep in library.dependencies]
        )

    def _owns_dependency(self, library, dependency):
        provider = self.libraries.get(dependency)
        return provider is not None and provider.native_scope == library.native_scope

    def _dependency(self, library, dependency):
        if self._owns_dependency(library, dependency):
            return self.resolve(dependency)
        return Interface(native=(dependency,))

    def link_dependencies(self, name):
        library = self.libraries[name]
        return tuple(
            dict.fromkeys(
                [
                    f"$<LINK_ONLY:{dependency}>"
                    if self._owns_dependency(library, dependency)
                    else dependency
                    for dependency in library.dependencies
                ]
                + list(self.resolve(name).native)
            )
        )


class BuildFileFunctions:
    """Source C/C++ rules and their native projections."""

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
        compiled = any(
            not re.search(r"\.(h|hh|h[+][+]|hm|hpp|hxx|H|HPP|in|txx|inc|inl)$", source)
            for source in srcs or []
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
            Library(native_name, self._native_scope, dependencies, interface),
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
            interface_blocks = defines_block + includes_block + system_includes_block
            if native_library:
                resolved = self._libraries.resolve(native_library)
                inherited = self._libraries.inherited(native_library)
                interface_blocks = "  RESOLVED\n" + "".join(
                    self._convert_string_list_block(
                        keyword, values, sort=False, quote=keyword != "LINK_LIBRARIES"
                    )
                    for keyword, values in (
                        ("DEFINES", resolved.defines),
                        ("INCLUDES", resolved.includes),
                        ("SYSTEM_INCLUDES", resolved.system_includes),
                        ("TRANSITIVE_COPTS", inherited.options),
                        ("TRANSITIVE_LINKOPTS", inherited.link_options),
                        (
                            "LINK_LIBRARIES",
                            self._libraries.link_dependencies(native_library),
                        ),
                    )
                )
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
        self._converter.body += (
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
        self._converter.body += (
            f"iree_cc_binary(\n"
            f"{name_block}"
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
        self._converter.body += (
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
        self._converter.body += (
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
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_benchmark(self, **kwargs):
        self.cc_binary_benchmark(**kwargs)
