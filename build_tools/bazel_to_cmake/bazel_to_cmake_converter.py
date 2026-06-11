# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Converter class for converting Bazel BUILD files to CMakeLists.txt files.

See bazel_to_cmake.py for usage.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=exec-used

import itertools
import os
import re

import bazel_to_cmake_config
import bazel_to_cmake_targets

# Maps Bazel string_flag labels to CMake variable names. Used by
# flag_values in iree_hal_executable rules and CTS macros to resolve
# {PLACEHOLDER} template variables to ${CMAKE_VARIABLE} references.
_BUILD_SETTING_CMAKE_VARIABLES = {}

_BUILD_SETTING_CMAKE_LIST_VARIABLES = {
    "//runtime/src/iree/hal/drivers/amdgpu:targets": "_IREE_HAL_AMDGPU_EXACT_TARGETS",
}

_LOCATION_PATTERN = re.compile(r"\$\((location|locations) ([^)]+)\)")

# Maps Bazel platform labels (from both select() conditions and
# target_compatible_with) to CMake platform condition values.
_PLATFORM_CMAKE_SYSTEM_NAME = {
    # select() condition labels (config_setting in build_tools/bazel/).
    "//build_tools/bazel:iree_is_android": "Android",
    "//build_tools/bazel:iree_is_linux": "Linux",
    "//build_tools/bazel:iree_is_macos": "Darwin",
    "//build_tools/bazel:iree_is_wasm": "wasm_32",
    "//build_tools/bazel:iree_is_windows": "Windows",
    # target_compatible_with constraint labels.
    "@platforms//os:android": "Android",
    "@platforms//os:emscripten": "Emscripten",
    "@platforms//os:linux": "Linux",
    "@platforms//os:macos": "Darwin",
    "@platforms//os:windows": "Windows",
    # CPU architecture constraints.
    "@platforms//cpu:wasm32": "wasm_32",
}

_RUNTIME_HAL_DRIVER_CMAKE_OPTIONS = {
    "//runtime/config/hal:driver_amdgpu": "IREE_HAL_DRIVER_AMDGPU",
    "//runtime/config/hal:driver_cuda": "IREE_HAL_DRIVER_CUDA",
    "//runtime/config/hal:driver_hip": "IREE_HAL_DRIVER_HIP",
    "//runtime/config/hal:driver_local_sync": "IREE_HAL_DRIVER_LOCAL_SYNC",
    "//runtime/config/hal:driver_local_task": "IREE_HAL_DRIVER_LOCAL_TASK",
    "//runtime/config/hal:driver_null": "IREE_HAL_DRIVER_NULL",
    "//runtime/config/hal:driver_vulkan": "IREE_HAL_DRIVER_VULKAN",
    "//runtime/config/hal:driver_webgpu": "IREE_HAL_DRIVER_WEBGPU",
    "//runtime/config/hal:executable_loader_embedded_elf": "IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF",
    "//runtime/config/hal:executable_loader_system_library": "IREE_HAL_EXECUTABLE_LOADER_SYSTEM_LIBRARY",
    "//runtime/config/hal:executable_loader_vmvx_module": "IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE",
    "//runtime/src/iree/hal/drivers/hip:rccl_enabled": "IREE_HAL_DRIVER_HIP_RCCL",
}

_LOOM_CONFIG_CMAKE_OPTIONS = {
    "//loom/config/target:amdgpu_artifacts": "LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU",
    "//loom/config/target:spirv_artifacts": "LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV",
    "//loom/config/target:spirv_vulkan_artifacts": "LOOM_TARGET_ARCH_SPIRV AND LOOM_EMIT_SPIRV AND IREE_HAL_DRIVER_VULKAN",
}


def _append_unique_condition(conditions, condition):
    if condition and condition not in conditions:
        conditions.append(condition)


class ConditionSelect:
    """Represents a supported conditional value from a Bazel select().

    Carries the full condition→value mapping so that the rule handler can emit
    CMake if/elseif/else blocks for platform- or option-specific values.
    """

    def __init__(self, conditions):
        # conditions: dict mapping condition labels to value lists.
        # Includes "//conditions:default" if present.
        self.conditions = conditions

    def __radd__(self, other):
        """Support: unconditional_list + ConditionSelect(...)."""
        if isinstance(other, list):
            return MixedDeps(unconditional=list(other), selects=[self])
        return NotImplemented

    def __add__(self, other):
        """Support: ConditionSelect(...) + unconditional_list."""
        if isinstance(other, list):
            return MixedDeps(unconditional=list(other), selects=[self])
        if isinstance(other, ConditionSelect):
            return MixedDeps(unconditional=[], selects=[self, other])
        if isinstance(other, MixedDeps):
            return MixedDeps(
                unconditional=list(other.unconditional),
                selects=[self] + list(other.selects),
            )
        return NotImplemented


class MixedDeps:
    """A deps value containing both unconditional entries and ConditionSelects.

    Created by list + ConditionSelect concatenation in BUILD file exec context.
    The rule handler splits this into a normal DEPS block (unconditional) plus
    CMake variable(s) for the conditional portions.
    """

    def __init__(self, unconditional, selects):
        self.unconditional = unconditional  # List of dep labels.
        self.selects = selects  # List of ConditionSelect objects.

    def __add__(self, other):
        """Support: MixedDeps + list (e.g., cfg.py injecting extra deps)."""
        if isinstance(other, list):
            return MixedDeps(
                unconditional=self.unconditional + other,
                selects=list(self.selects),
            )
        if isinstance(other, ConditionSelect):
            return MixedDeps(
                unconditional=list(self.unconditional),
                selects=list(self.selects) + [other],
            )
        return NotImplemented

    def __radd__(self, other):
        """Support: list + MixedDeps."""
        if isinstance(other, list):
            return MixedDeps(
                unconditional=other + self.unconditional,
                selects=list(self.selects),
            )
        if isinstance(other, ConditionSelect):
            return MixedDeps(
                unconditional=list(self.unconditional),
                selects=[other] + list(self.selects),
            )
        return NotImplemented


_SPIRV_TOOL_TARGET_COMPATIBLE_WITH = ConditionSelect(
    {
        "//loom/config/target:spirv_artifacts": [],
        "//runtime/config/hal:driver_vulkan": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }
)


class _SelectsModule:
    """No-op subset of @bazel_skylib//lib:selects.bzl used in BUILD files."""

    def config_setting_group(self, *args, **kwargs):
        pass


class BuildFileFunctions(object):
    """Object passed to `exec` that has handlers for BUILD file functions."""

    def __init__(
        self,
        *,
        converter: "Converter",
        targets: bazel_to_cmake_targets.TargetConverter,
        build_dir: str,
        repo_root: str = "",
    ):
        self._converter = converter
        self._targets = targets
        self._repo_root = os.path.abspath(repo_root) if repo_root else ""
        if self._repo_root and build_dir and not os.path.isabs(build_dir):
            self._build_dir = os.path.join(self._repo_root, build_dir)
        else:
            self._build_dir = build_dir
        self._filegroup_srcs = {}
        self._target_file_labels = set()
        self._target_file_paths = {}
        self.selects = _SelectsModule()
        self._custom_initialize()

    def _custom_initialize(self):
        pass

    # ------------------------------------------------------------------------- #
    # Conversion utilities, written to reduce boilerplate and allow for reuse   #
    # between similar rule conversions (e.g. cc_library and cc_binary).         #
    # ------------------------------------------------------------------------- #

    def _expand_cmake_var(self, var):
        return "${" + var + "}"

    def _cmake_variable_name(self, name):
        return re.sub(r"[^A-Za-z0-9_]", "_", name)

    def _convert_string_arg_block(self, name, value, quote=True):
        #  NAME
        #    "value"
        if value is None:
            return ""
        if quote:
            return f'  {name}\n    "{value}"\n'
        else:
            return f"  {name}\n    {value}\n"

    # Match Bazel's timeout values
    # https://docs.bazel.build/versions/main/test-encyclopedia.html
    _timeout_map = {
        "short": 60,
        "moderate": 300,
        "long": 900,
        "eternal": 3600,
    }

    def _should_skip_target(self, tags=None, **kwargs):
        if tags and "skip-bazel_to_cmake" in tags:
            return True
        return False

    def _check_no_unhandled_kwargs(self, rule_name, kwargs):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise NotImplementedError(f"{rule_name} attributes: {names}")

    def _convert_platform_condition(self, constraint_label):
        """Returns a CMake condition string for a platform constraint label."""
        if constraint_label == "//build_tools/bazel:iree_is_wasm":
            return 'IREE_ARCH STREQUAL "wasm_32"'
        cmake_name = _PLATFORM_CMAKE_SYSTEM_NAME.get(constraint_label)
        if not cmake_name:
            return None
        # CPU architecture constraints use IREE_ARCH; OS constraints use
        # CMAKE_SYSTEM_NAME.
        if constraint_label.startswith("@platforms//cpu:"):
            return f'IREE_ARCH STREQUAL "{cmake_name}"'
        return f'CMAKE_SYSTEM_NAME STREQUAL "{cmake_name}"'

    def _convert_select_condition(self, label):
        """Returns a CMake condition string for a supported select condition."""
        cmake_condition = getattr(label, "cmake_condition", None)
        if cmake_condition:
            return cmake_condition
        condition = self._convert_platform_condition(label)
        if condition:
            return condition
        return _RUNTIME_HAL_DRIVER_CMAKE_OPTIONS.get(
            label
        ) or _LOOM_CONFIG_CMAKE_OPTIONS.get(label)

    def _condition_select_compatibility_condition(self, condition_select):
        compatible_conditions = []
        for label, value in condition_select.conditions.items():
            if label == "//conditions:default":
                continue
            # Empty list means compatible for this condition.
            if value == []:
                condition = self._convert_select_condition(label)
                _append_unique_condition(compatible_conditions, condition)
        if not compatible_conditions:
            return None
        return " OR ".join(compatible_conditions)

    def _target_compatible_condition(self, target_compatible_with):
        if not target_compatible_with:
            return None

        if isinstance(target_compatible_with, ConditionSelect):
            return self._condition_select_compatibility_condition(
                target_compatible_with
            )

        if isinstance(target_compatible_with, MixedDeps):
            conditions = []
            for label in target_compatible_with.unconditional:
                condition = self._convert_select_condition(label)
                if not condition:
                    raise NotImplementedError(f"target_compatible_with: {label}")
                _append_unique_condition(conditions, condition)
            for condition_select in target_compatible_with.selects:
                condition = self._condition_select_compatibility_condition(
                    condition_select
                )
                _append_unique_condition(conditions, condition)
            if not conditions:
                return None
            return " AND ".join(
                f"({condition})" if " OR " in condition else condition
                for condition in conditions
            )

        conditions = []
        for label in target_compatible_with:
            condition = self._convert_select_condition(label)
            if condition:
                _append_unique_condition(conditions, condition)
            else:
                raise NotImplementedError(f"target_compatible_with: {label}")
        return " AND ".join(conditions)

    def _emit_platform_guard_begin(self, target_compatible_with):
        """Emits platform guards for target_compatible_with."""
        condition = self._target_compatible_condition(target_compatible_with)
        if condition:
            self._converter.body += f"if({condition})\n"

    def _emit_platform_guard_end(self, target_compatible_with):
        """Emits endif() to close a target_compatible_with guard."""
        if self._target_compatible_condition(target_compatible_with):
            # Strip trailing blank line from the target body so endif() is
            # adjacent to the closing paren.
            self._converter.body = self._converter.body.rstrip("\n") + "\n"
            self._converter.body += "endif()\n\n"

    def _convert_platform_select_deps(self, name, deps):
        """Handles deps that may contain ConditionSelect entries.

        If deps is a plain list, returns (converted_deps_block, "").
        If deps is a MixedDeps or ConditionSelect, emits a CMake variable
        with if/elseif/else blocks before the target and returns
        (converted_deps_block_with_variable, variable_block).
        """
        if deps is None:
            return self._convert_target_list_block("DEPS", None), ""
        if isinstance(deps, ConditionSelect):
            deps = MixedDeps(unconditional=[], selects=[deps])
        if not isinstance(deps, MixedDeps):
            return self._convert_target_list_block("DEPS", deps), ""

        # Emit a CMake variable for the conditional deps.
        var_name = f"_{self._cmake_variable_name(name)}_platform_deps"
        var_block = f'set({var_name} "")\n'

        for ps in deps.selects:
            first = True
            for label, values in ps.conditions.items():
                if label == "//conditions:default":
                    continue
                cond = self._convert_select_condition(label)
                if not cond:
                    raise NotImplementedError(f"select condition: {label}")
                keyword = "if" if first else "elseif"
                var_block += f"{keyword}({cond})\n"
                cmake_targets = []
                for t in values:
                    cmake_targets.extend(self._convert_target(t))
                for ct in sorted(cmake_targets):
                    var_block += f"  list(APPEND {var_name} {ct})\n"
                first = False
            # Default branch.
            default_values = ps.conditions.get("//conditions:default", [])
            if default_values:
                var_block += "else()\n"
                cmake_targets = []
                for t in default_values:
                    cmake_targets.extend(self._convert_target(t))
                for ct in sorted(cmake_targets):
                    var_block += f"  list(APPEND {var_name} {ct})\n"
            var_block += "endif()\n"

        # Build the DEPS block: unconditional deps + the variable reference.
        deps_block = self._convert_target_list_block("DEPS", deps.unconditional)
        # Append the variable reference to the deps block.
        if deps_block:
            # Insert the variable ref before the closing of the DEPS block.
            deps_block = deps_block.rstrip("\n") + f"\n    ${{{var_name}}}\n"
        else:
            deps_block = f"  DEPS\n    ${{{var_name}}}\n"

        return deps_block, var_block

    def _convert_platform_select_strings(
        self, name, block_name, values, quote=True, sort=False
    ):
        """Handles string lists that may contain ConditionSelect entries.

        If values is a plain list, returns (converted_block, ""). If values is a
        MixedDeps or ConditionSelect, emits a CMake list variable with
        if/elseif/else blocks before the target and returns the normal argument
        block with an additional variable reference.
        """
        if values is None:
            return self._convert_string_list_block(block_name, None), ""
        if isinstance(values, ConditionSelect):
            values = MixedDeps(unconditional=[], selects=[values])
        if not isinstance(values, MixedDeps):
            return self._convert_string_list_block(
                block_name, values, quote=quote, sort=sort
            ), ""

        var_name = f"_{self._cmake_variable_name(name)}_platform_{block_name.lower()}"
        var_block = f'set({var_name} "")\n'

        def append_value(value):
            if quote:
                return f'  list(APPEND {var_name} "{value}")\n'
            return f"  list(APPEND {var_name} {value})\n"

        for ps in values.selects:
            first = True
            for label, selected_values in ps.conditions.items():
                if label == "//conditions:default":
                    continue
                cond = self._convert_select_condition(label)
                if not cond:
                    raise NotImplementedError(f"select condition: {label}")
                keyword = "if" if first else "elseif"
                var_block += f"{keyword}({cond})\n"
                if sort:
                    selected_values = sorted(selected_values)
                for value in selected_values:
                    var_block += append_value(value)
                first = False
            default_values = ps.conditions.get("//conditions:default", [])
            if default_values:
                var_block += "else()\n"
                if sort:
                    default_values = sorted(default_values)
                for value in default_values:
                    var_block += append_value(value)
            var_block += "endif()\n"

        unconditional = values.unconditional
        if sort:
            unconditional = sorted(unconditional)
        string_block = self._convert_string_list_block(
            block_name, unconditional, quote=quote, sort=False
        )
        if string_block:
            string_block = string_block.rstrip("\n") + f"\n    ${{{var_name}}}\n"
        else:
            string_block = f"  {block_name}\n    ${{{var_name}}}\n"

        return string_block, var_block

    def _convert_timeout_arg_block(self, name, value):
        if value is None:
            return ""
        value = self._timeout_map[value]
        return f"  {name}\n    {value}\n"

    def _convert_string_list_block(self, name, values, quote=True, sort=False):
        # Note this deliberately distinguishes between an empty list (argument
        # explicitly specified) and None (argument left as default).
        if values is None:
            return ""

        if sort:
            values = sorted(values)

        if quote:
            values_list = "\n".join([f'    "{v}"' for v in values])
        else:
            values_list = "\n".join([f"    {v}" for v in values])

        return f"  {name}\n{values_list}\n"

    def _convert_sanitizer_suppressions_block(self, sanitizer_suppressions):
        if not sanitizer_suppressions:
            return ""
        entries = []
        for sanitizer, label in sorted(sanitizer_suppressions.items()):
            suppression_name = self._sanitizer_suppression_name(label)
            entries.extend([sanitizer, suppression_name])
        return self._convert_string_list_block(
            "SANITIZER_SUPPRESSIONS", entries, quote=False
        )

    def _sanitizer_suppression_name(self, label):
        suffix = label.rsplit(":", 1)[-1]
        if suffix.startswith("lsan_suppressions_") and suffix.endswith(".txt"):
            return suffix[len("lsan_suppressions_") : -len(".txt")]
        raise NotImplementedError(f"sanitizer suppression label: {label}")

    def _convert_option_block(self, option, option_value):
        if option_value:
            # Note: this is a truthiness check as well as an existence check, e.g.
            # Bazel `testonly = False` will be handled correctly by this condition.
            return f"  {option}\n"
        else:
            return ""

    def _convert_target_block(self, name, target):
        if target is None:
            return ""

        # Convert the target name from its Bazel name to the corresponding CMake name.
        # The specific conversion pattern depends on the target location. In general,
        # Bazel targets are fully qualified and use slashes as delimiters, while
        # targets in CMake are rooted on subtrees and use _ (with :: aliases).
        cmake_aliases = self._targets.convert_target(target)
        if len(cmake_aliases) != 1:
            raise ValueError(
                f"Expected a CMake alias from {target}. Got {cmake_aliases}"
            )
        target = cmake_aliases[0]
        # Replace aliased :: target names with their explicit _ names.
        target = target.replace("::", "_")
        return self._convert_string_arg_block(name, target, quote=False)

    def _filegroup_dep_filename(self, src):
        return f"{src}.stamp"

    def _normalize_label(self, src):
        """
        Convert label to file path suitable for CMake to use as a dependency.
        """

        # Bazel allows srcs to reference targets in the current package (leading
        # ':') or in other packages (leading '//'). We map that to paths by:
        # - dropping any leading ':' as in:
        #      ':generated.c' -> 'generated.c'
        # - replacing any leading '//' by '${PROJECT_SOURCE_DIR}/' or
        #   '${PROJECT_BINARY_DIR}/' and any internal ':' by '/', as in:
        #      '//path/to/package:source.c'
        #      -> '${PROJECT_SOURCE_DIR}/path/to/package/source.c'
        #      '//path/to/package:generated.c'
        #      -> '${PROJECT_BINARY_DIR}/path/to/package/generated.c'
        pkg_root_relative_label = src.startswith("//")
        src = src.lstrip("/").lstrip(":").replace(":", "/")
        if not pkg_root_relative_label:
            return src
        # Repo-root-relative labels (//pkg:file) resolve from the repo root,
        # not from the current package directory.
        check_dir = self._repo_root if self._repo_root else self._build_dir
        if os.path.exists(os.path.join(check_dir, src)):
            return f"${{PROJECT_SOURCE_DIR}}/{src}"
        else:
            return f"${{PROJECT_BINARY_DIR}}/{src}"

    def _current_package(self):
        if not self._repo_root:
            return ""
        return os.path.relpath(self._build_dir, self._repo_root).replace("\\", "/")

    def _canonical_location_label(self, label):
        package, name = self._split_location_label(label)
        if package:
            return f"//{package}:{name}"
        return f":{name}"

    def _current_target_label(self, name):
        package = self._current_package()
        if package:
            return f"//{package}:{name}"
        return f":{name}"

    def _should_emit_python_target(self):
        return False

    def _python_package_dirs(self):
        return []

    def _convert_location_arg(self, arg):
        def replace_location(match):
            return " ".join(self._cmake_location_paths(match.group(2)))

        return [_LOCATION_PATTERN.sub(replace_location, arg)]

    def _convert_location_args(self, args):
        if args is None:
            return None
        converted_args = []
        for arg in args:
            converted_args.extend(self._convert_location_arg(arg))
        return converted_args

    def _convert_native_test_location_arg(self, arg):
        def replace_location(match):
            paths = self._cmake_location_paths(match.group(2))
            if len(paths) != 1:
                return " ".join(paths)
            return "{{%s}}" % paths[0]

        return _LOCATION_PATTERN.sub(replace_location, arg)

    def _convert_native_test_location_args(self, args):
        if args is None:
            return None
        return [self._convert_native_test_location_arg(arg) for arg in args]

    def _location_label_keys(self, args):
        if args is None:
            return set()
        labels = set()
        for arg in args:
            for match in _LOCATION_PATTERN.finditer(arg):
                labels.add(self._split_location_label(match.group(2)))
        return labels

    def _cmake_location_paths(self, label):
        package, name = self._split_location_label(label)
        if package == self._current_package() and name in self._filegroup_srcs:
            return [
                self._cmake_source_location_path(src)
                for src in self._filegroup_srcs[name]
            ]
        canonical_label = self._canonical_location_label(label)
        if canonical_label in self._target_file_paths:
            path = self._target_file_paths[canonical_label]
            if package == self._current_package():
                return [path]
            return [f"${{PROJECT_BINARY_DIR}}/{package}/{path}"]
        source_path = self._cmake_source_location_path(label)
        if self._repo_root:
            concrete_source_path = os.path.join(self._repo_root, package, name)
            if os.path.exists(concrete_source_path):
                return [source_path]
        try:
            cmake_targets = self._targets.convert_target("//%s:%s" % (package, name))
        except (KeyError, ValueError):
            return [source_path]
        if len(cmake_targets) == 1 and cmake_targets[0]:
            target = cmake_targets[0].replace("::", "_")
            return [f"$<TARGET_FILE:{target}>"]
        return [source_path]

    def _split_location_label(self, label):
        if label.startswith("//"):
            package_and_name = label[2:]
            if ":" in package_and_name:
                package, name = package_and_name.split(":", 1)
                return package, name
            return package_and_name, package_and_name.rsplit("/", 1)[-1]
        if label.startswith(":"):
            return self._current_package(), label[1:]
        if ":" in label and not os.path.splitext(label)[1]:
            package, name = label.split(":", 1)
            return f"{self._current_package()}/{package}", name
        return self._current_package(), label

    def _cmake_source_location_path(self, label):
        if label.startswith("${"):
            return label
        package, name = self._split_location_label(label)
        if name in self._filegroup_srcs and package == self._current_package():
            raise ValueError(f"filegroup label {label} was not expanded")
        return f"${{PROJECT_SOURCE_DIR}}/{package}/{name}"

    def _canonical_python_label(self, label, current_package=None):
        if label.startswith("@"):
            return label
        package = (
            current_package if current_package is not None else self._current_package()
        )
        if label.startswith(":"):
            return "//%s%s" % (package, label)
        if label.startswith("//"):
            return label
        if ":" in label:
            return "//%s/%s" % (package, label)
        return "//%s:%s" % (package, label)

    def _split_python_label(self, label, current_package=None):
        canonical_label = self._canonical_python_label(label, current_package)
        if canonical_label.startswith("@"):
            raise KeyError(f"external Python target '{label}' is not converted")
        if not canonical_label.startswith("//"):
            raise KeyError(f"unsupported Python target label '{label}'")
        package_and_name = canonical_label[2:]
        if ":" in package_and_name:
            package, name = package_and_name.split(":", 1)
        else:
            package = package_and_name
            name = package.rsplit("/", 1)[-1]
        return package, name

    def _python_cmake_target_name(self, label, current_package=None):
        package, name = self._split_python_label(label, current_package)
        return self._targets.convert_target("//%s:%s" % (package, name))[0]

    def _python_file_cmake_path(self, label, current_package=None):
        package, name = self._split_python_label(label, current_package)
        repo_relative_path = os.path.join(package, name)
        return os.path.relpath(
            os.path.join(self._repo_root, repo_relative_path), self._build_dir
        ).replace("\\", "/")

    def _python_file_basename(self, label):
        if label.startswith("//") or label.startswith(":"):
            label = label.rsplit(":", 1)[-1]
        return os.path.basename(label)

    def _python_local_targets(self, targets):
        if not targets:
            return []
        return [
            target
            for target in targets
            if isinstance(target, str) and not target.startswith("@")
        ]

    def _has_only_external_targets(self, targets):
        if not targets:
            return True
        if isinstance(targets, ConditionSelect):
            return all(
                self._has_only_external_targets(values)
                for values in targets.conditions.values()
            )
        if isinstance(targets, MixedDeps):
            return self._has_only_external_targets(targets.unconditional) and all(
                self._has_only_external_targets(values)
                for select in targets.selects
                for values in select.conditions.values()
            )
        return all(
            isinstance(target, str) and target.startswith("@") for target in targets
        )

    def _convert_python_targets(self, targets):
        return [
            self._python_cmake_target_name(target)
            for target in self._python_local_targets(targets)
        ]

    def _convert_python_target_list_blocks(self, target_name, list_name, targets):
        if not targets:
            return "", ""
        if isinstance(targets, ConditionSelect):
            targets = MixedDeps(unconditional=[], selects=[targets])
        if isinstance(targets, MixedDeps):
            converted_targets = self._convert_python_targets(targets.unconditional)
            var_block = ""
            var_name = (
                f"_{self._cmake_variable_name(target_name)}_python_{list_name.lower()}"
            )
            for select in targets.selects:
                first = True
                select_has_local_targets = any(
                    self._python_local_targets(values)
                    for values in select.conditions.values()
                )
                if not select_has_local_targets:
                    continue
                if not var_block:
                    var_block = f'set({var_name} "")\n'
                for values in select.conditions.values():
                    for target in values:
                        if not isinstance(target, str):
                            raise NotImplementedError(f"{list_name} with select()")
                default_targets = []
                for label, values in select.conditions.items():
                    if label == "//conditions:default":
                        default_targets = self._convert_python_targets(values)
                        continue
                    local_targets = self._convert_python_targets(values)
                    cond = self._convert_select_condition(label)
                    if not cond:
                        raise NotImplementedError(f"select condition: {label}")
                    keyword = "if" if first else "elseif"
                    var_block += f"{keyword}({cond})\n"
                    for target in local_targets:
                        var_block += f"  list(APPEND {var_name} {target})\n"
                    first = False
                if default_targets:
                    var_block += "else()\n"
                    for target in default_targets:
                        var_block += f"  list(APPEND {var_name} {target})\n"
                var_block += "endif()\n"
            deps_block = self._convert_string_list_block(
                list_name, converted_targets, sort=False, quote=False
            )
            if var_block:
                if deps_block:
                    deps_block = deps_block.rstrip("\n") + f"\n    ${{{var_name}}}\n"
                else:
                    deps_block = f"  {list_name}\n    ${{{var_name}}}\n"
            elif not converted_targets:
                deps_block = ""
            return deps_block, var_block
        converted_targets = self._convert_python_targets(targets)
        if not converted_targets:
            return "", ""
        return self._convert_string_list_block(
            list_name, converted_targets, sort=False, quote=False
        ), ""

    def _convert_python_target_list_block(self, list_name, targets):
        block, var_block = self._convert_python_target_list_blocks(
            list_name.lower(), list_name, targets
        )
        if var_block:
            raise NotImplementedError(f"{list_name} with select()")
        return block

    def _convert_srcs_block(self, srcs, is_generated=False, block_name="SRCS"):
        if not srcs:
            return ""

        srcs = [
            (
                self._normalize_label(s)
                if s.startswith("$") or os.path.splitext(s)[1]
                else self._filegroup_dep_filename(self._normalize_label(s))
            )
            for s in srcs
        ]

        return self._convert_string_list_block(block_name, srcs, sort=True)

    def _convert_data_srcs_block(self, srcs, block_name="SRCS"):
        if not srcs:
            return ""

        converted_srcs = []
        for src in srcs:
            if src.startswith(":") or src.startswith("//"):
                paths = self._cmake_location_paths(src)
                if self._canonical_location_label(
                    src
                ) in self._target_file_labels or not any(
                    path.startswith("$<TARGET_FILE:") for path in paths
                ):
                    converted_srcs.extend(paths)
                else:
                    converted_srcs.append(self._normalize_label(src))
            elif src.startswith("$") or os.path.splitext(src)[1]:
                converted_srcs.append(self._normalize_label(src))
            else:
                converted_srcs.append(
                    self._filegroup_dep_filename(self._normalize_label(src))
                )

        return self._convert_string_list_block(block_name, converted_srcs, sort=True)

    def _convert_target(self, target):
        """Returns a list of targets that correspond to the specified Bazel target.
        Note that this must be a list because some targets have a one to many mapping.
        """
        return self._targets.convert_target(target)

    def _convert_single_target(self, target):
        replacement_targets = self._convert_target(target)
        if len(replacement_targets) != 1:
            raise RuntimeError(
                f"Expected single target replacement for {target},"
                f" but got multiple: {replacement_targets}"
            )
        return replacement_targets[0]

    def _convert_single_target_block(self, name, target):
        mapped_target = self._convert_single_target(target)
        return self._convert_string_arg_block(name, mapped_target, quote=False)

    def _convert_target_list_block(self, list_name, targets, omit_empty=False):
        if targets is None:
            return ""

        #  DEPS
        #    package1::target1
        #    package1::target2
        #    package2::target
        targets = [self._convert_target(t) for t in targets]
        # Flatten lists
        targets = list(itertools.chain.from_iterable(targets))
        # Remove duplicates
        targets = set(targets)
        # Remove Falsey (None and empty string) values
        targets = list(filter(None, targets))
        if omit_empty and not targets:
            return ""

        return self._convert_string_list_block(
            list_name, targets, sort=True, quote=False
        )

    def _local_cts_testdata_lib_deps(self, deps):
        if isinstance(deps, MixedDeps):
            deps = deps.unconditional
        if not deps:
            return []
        return [
            dep
            for dep in deps
            if isinstance(dep, str)
            and dep.startswith(":testdata_")
            and dep.endswith("_lib")
        ]

    def _emit_optional_local_cts_testdata_guard_begin(self, deps):
        testdata_deps = self._local_cts_testdata_lib_deps(deps)
        if not testdata_deps:
            return False

        optional_targets = []
        for dep in testdata_deps:
            for target in self._convert_target(dep):
                if target.startswith("::"):
                    target = "${_IREE_OPTIONAL_TESTDATA_PACKAGE_NS}" + target
                optional_targets.append(target)

        if not optional_targets:
            return False

        conditions = " AND ".join(
            f"TARGET {target}" for target in sorted(set(optional_targets))
        )
        self._converter.body += (
            f"iree_package_ns(_IREE_OPTIONAL_TESTDATA_PACKAGE_NS)\nif({conditions})\n"
        )
        return True

    def _emit_optional_local_cts_testdata_guard_end(self, did_emit_guard):
        if did_emit_guard:
            self._converter.body += "endif()\n\n"

    def _convert_includes_block(self, includes):
        if not includes:
            return ""
        dirs = []
        for include in includes:
            dirs.append(
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/%s>" % (include,)
            )
            dirs.append(
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/%s>" % (include,)
            )
        return self._convert_string_list_block("INCLUDES", dirs, sort=False, quote=True)

    # ------------------------------------------------------------------------- #
    # Function handlers that convert BUILD definitions to CMake definitions.    #
    #                                                                           #
    # Names and signatures must match 1:1 with those expected in BUILD files    #
    # except that default values for optional arguments should generally be     #
    # `None` so we don't set them unnecessarily in the CMakeLists.txt files.    #
    # Each function that may be found in a BUILD file must be listed here.      #
    # ------------------------------------------------------------------------- #

    # Functions with no mapping to CMake. Just ignore these.
    def alias(self, *args, **kwargs):
        pass

    def bool_flag(self, *args, **kwargs):
        pass

    def string_list_flag(self, *args, **kwargs):
        pass

    def declare_requirements(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        """Attempts to bind constants from loaded .bzl files.

        Bazel load() imports names from .bzl files into the BUILD file's
        namespace. The converter can evaluate simple .bzl files that contain
        only Python-compatible constant assignments (lists, dicts, strings)
        and bind the requested names. Complex .bzl files with Starlark-
        specific constructs silently fall back to no-op behavior.
        """
        if len(args) < 2:
            return
        bzl_label = args[0]
        names = args[1:]

        # Resolve the .bzl file path from its label.
        if bzl_label.startswith(":"):
            abs_path = os.path.join(self._build_dir, bzl_label[1:])
        elif bzl_label.startswith("//"):
            # "//path/to/pkg:file.bzl" -> "path/to/pkg/file.bzl"
            rel = bzl_label[2:].replace(":", "/")
            abs_path = os.path.join(self._repo_root, rel)
        else:
            return  # External repositories — can't resolve.

        if not os.path.isfile(abs_path):
            return

        try:
            namespace = {}
            with open(abs_path) as f:
                exec(f.read(), namespace)
            for name in names:
                if name in namespace and hasattr(self, "_exec_namespace"):
                    # Only bind names not already provided by converter
                    # handlers. This avoids overwriting converter methods
                    # (like enforce_glob) with Starlark implementations that
                    # reference native.glob() and other unavailable builtins.
                    if name not in self._exec_namespace:
                        self._exec_namespace[name] = namespace[name]
        except Exception:
            if hasattr(self, "_exec_namespace"):
                for name in names:
                    if name == "REQUIREMENTS" and name not in self._exec_namespace:
                        self._exec_namespace[name] = []
            pass  # .bzl uses Starlark features — fall back to no-op.

    def package(self, **kwargs):
        pass

    def iree_build_test(self, **kwargs):
        pass

    def iree_assert_no_dependency(self, **kwargs):
        pass

    def test_suite(self, **kwargs):
        pass

    def config_setting(self, **kwargs):
        pass

    def exports_files(self, *args, **kwargs):
        pass

    def iree_td_library(self, *args, **kwargs):
        """Ignores iree_td_library - no CMake equivalent needed.

        Transitive .td dependencies are automatically tracked via depfiles
        generated by tablegen, so we don't need explicit td_library targets.
        """
        pass

    def iree_py_binary(
        self,
        name,
        srcs=None,
        deps=None,
        main=None,
        imports=None,
        data=None,
        tags=None,
        target_compatible_with=None,
        testonly=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if not self._should_emit_python_target():
            return
        self._check_no_unhandled_kwargs("iree_py_binary", kwargs)
        if data and not self._has_only_external_targets(data):
            raise NotImplementedError(f"iree_py_binary data: {name}")
        source_list = list(srcs or [])
        if main and main not in source_list:
            source_list.append(main)
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        main_block = self._convert_string_arg_block("MAIN", main)
        source_block = self._convert_string_list_block("SRCS", source_list, sort=False)
        imports_block = self._convert_string_list_block("IMPORTS", imports, sort=False)
        deps_block, deps_var_block = self._convert_python_target_list_blocks(
            name, "DEPS", deps
        )
        self._emit_platform_guard_begin(target_compatible_with)
        if deps_var_block:
            self._converter.body += deps_var_block
        self._converter.body += (
            "iree_py_library(\n"
            f"{name_block}"
            f"{main_block}"
            f"{source_block}"
            f"{imports_block}"
            f"{deps_block}"
            ")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_py_library(
        self,
        name,
        srcs=None,
        deps=None,
        imports=None,
        data=None,
        tags=None,
        target_compatible_with=None,
        testonly=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if not self._should_emit_python_target():
            return
        self._check_no_unhandled_kwargs("iree_py_library", kwargs)
        if data and not self._has_only_external_targets(data):
            raise NotImplementedError(f"iree_py_library data: {name}")
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        source_block = self._convert_string_list_block("SRCS", srcs, sort=False)
        imports_block = self._convert_string_list_block("IMPORTS", imports, sort=False)
        deps_block, deps_var_block = self._convert_python_target_list_blocks(
            name, "DEPS", deps
        )
        self._emit_platform_guard_begin(target_compatible_with)
        if deps_var_block:
            self._converter.body += deps_var_block
        self._converter.body += (
            "iree_py_library(\n"
            f"{name_block}"
            f"{source_block}"
            f"{imports_block}"
            f"{deps_block}"
            ")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_py_test(
        self,
        name,
        srcs=None,
        args=None,
        data=None,
        deps=None,
        env=None,
        imports=None,
        main=None,
        package_dirs=None,
        tags=None,
        target_compatible_with=None,
        testonly=None,
        timeout=None,
        visibility=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if not self._should_emit_python_target():
            return
        self._check_no_unhandled_kwargs("iree_py_test", kwargs)
        if env:
            raise NotImplementedError(f"iree_py_test env: {name}")
        if data:
            if not isinstance(data, list):
                if self._has_only_external_targets(data):
                    data = None
                else:
                    raise NotImplementedError(f"iree_py_test data: {name}")
        if data:
            location_labels = self._location_label_keys(args)
            data_labels = {self._split_location_label(label) for label in data}
            if not data_labels.issubset(location_labels):
                raise NotImplementedError(f"iree_py_test data: {name}")
        source_list = list(srcs or [])
        main_source = None
        if main:
            for source in source_list:
                if self._python_file_basename(source) == main:
                    main_source = source
                    break
            if main_source is None:
                main_source = main
        elif len(source_list) == 1:
            main_source = source_list[0]
        else:
            raise ValueError(f"iree_py_test {name} requires a main source")

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        source_block = self._convert_string_arg_block(
            "SRCS", self._python_file_cmake_path(main_source)
        )
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_location_args(args), sort=False
        )
        deps_block, deps_var_block = self._convert_python_target_list_blocks(
            name, "DEPS", deps
        )
        imports_block = self._convert_string_list_block("IMPORTS", imports, sort=False)
        labels_block = self._convert_string_list_block("LABELS", tags)
        package_dirs_block = self._convert_string_list_block(
            "PACKAGE_DIRS",
            package_dirs or self._python_package_dirs(),
            sort=False,
        )
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)
        self._emit_platform_guard_begin(target_compatible_with)
        if deps_var_block:
            self._converter.body += deps_var_block
        self._converter.body += (
            "iree_py_test(\n"
            f"{name_block}"
            f"{source_block}"
            f"{args_block}"
            f"{deps_block}"
            f"{imports_block}"
            f"{labels_block}"
            f"{package_dirs_block}"
            f"{timeout_block}"
            ")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def py_binary(self, **kwargs):
        self.iree_py_binary(**kwargs)

    def py_library(self, **kwargs):
        self.iree_py_library(**kwargs)

    def py_test(self, **kwargs):
        self.iree_py_test(**kwargs)

    def filegroup(self, name, srcs, **kwargs):
        if not srcs:
            return
        self._filegroup_srcs[name] = list(srcs)

        # Converting a dependency on a filegroup requires either using the
        # transitive dependency to the actual file or creating a similar
        # abstraction in CMake.
        #
        # One way of doing the transitive dependency is peeking in the build
        # file that defines a given filegroup but goes against the current
        # design where each build file is processed independently.
        #
        # Alternatively, the build file that defines a filegroup could set a
        # variable with the list of all the files in the filegroup which the
        # CMakeLists.txt corresponding to the using build file would use.
        # However that requires the variable to be defined before the
        # add_directory() for the corresponding using CMakeLists.txt which is
        # not a given.
        #
        # Instead, we generate a custom command that creates a stamp file that
        # acts as an abstraction to the filegroup. The using CMakeLists.txt
        # then creates a file dependency on that stamp file. We also need a
        # custom target in the same CMakeLists.txt to ensure a rule for the
        # custom command is actually created as per add_custom_command
        # documentation.
        depends_block = self._convert_srcs_block(srcs, block_name="DEPENDS")
        stamp_file = self._filegroup_dep_filename(name)
        self._converter.body += (
            f"add_custom_command(OUTPUT {stamp_file}\n"
            f"    COMMAND ${{CMAKE_COMMAND}} -E touch {stamp_file}\n"
            f"{depends_block}"
            f")\n\n"
            f"add_custom_target({name}\n"
            f"    DEPENDS {stamp_file}\n"
            f")\n\n"
        )

    def sh_binary(self, name, **kwargs):
        if self._should_skip_target(**kwargs):
            return
        raise NotImplementedError(f"sh_binary: {name}")

    def wasm_cc_library(
        self,
        name,
        srcs=None,
        module=None,
        deps=None,
        testonly=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_string_list_block("SRCS", srcs, sort=True)
        module_block = self._convert_string_arg_block("MODULE", module)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_wasm_cc_library(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{module_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_wasm_cc_library(self, **kwargs):
        """Direct handler for iree_wasm_cc_library (loaded from .bzl)."""
        self.wasm_cc_library(**kwargs)

    def wasm_entry(
        self,
        name,
        main=None,
        srcs=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        main_block = self._convert_string_arg_block("MAIN", main)
        srcs_block = self._convert_string_list_block("SRCS", srcs, sort=True)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_wasm_entry(\n{name_block}{main_block}{srcs_block}  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_wasm_entry(self, **kwargs):
        """Direct handler for iree_wasm_entry (loaded from .bzl)."""
        self.wasm_entry(**kwargs)

    def wasm_cc_binary(
        self,
        name,
        main=None,
        srcs=None,
        deps=None,
        copts=None,
        defines=None,
        linkopts=None,
        testonly=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        # Emit as a standard iree_cc_binary for the C side, plus a
        # wasm bundle step. For now, emit the cc_binary and add a comment
        # noting the JS bundling is handled at build time.
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        main_block = self._convert_string_arg_block("MAIN", main)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_wasm_cc_binary(\n"
            f"{name_block}"
            f"{main_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_wasm_cc_binary(self, **kwargs):
        """Direct handler for iree_wasm_cc_binary (loaded from .bzl)."""
        self.wasm_cc_binary(**kwargs)

    def wasm_cc_test(
        self,
        name,
        main=None,
        srcs=None,
        deps=None,
        copts=None,
        defines=None,
        linkopts=None,
        testonly=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        deps_block = self._convert_target_list_block("DEPS", deps)
        main_block = self._convert_string_arg_block("MAIN", main)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_wasm_cc_test(\n"
            f"{name_block}"
            f"{main_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{deps_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_wasm_cc_test(self, **kwargs):
        """Direct handler for iree_wasm_cc_test (loaded from .bzl)."""
        self.wasm_cc_test(**kwargs)

    def enforce_glob(self, files, **kwargs):
        return files

    def glob(self, include, exclude=None, exclude_directories=1):
        if exclude_directories != 1:
            raise NotImplementedError("glob with exclude_directories != 1")
        if exclude is None:
            exclude = []

        glob_vars = []
        for pattern in include:
            if "**" in pattern:
                # bazel's glob has some specific restrictions about crossing package
                # boundaries. We have no uses of recursive globs. Rather than try to
                # emulate them or silently give different behavior, just error out.
                # See https://bazel.build/reference/be/functions.html#glob
                raise NotImplementedError("Recursive globs not supported")
            # Bazel `*.mlir` glob -> CMake Variable `_GLOB_X_MLIR`
            var = "_GLOB_" + pattern.replace("*", "X").replace(".", "_").upper()
            glob_vars.append(var)
            self._converter.body += (
                f"file(GLOB {var} LIST_DIRECTORIES false"
                f" RELATIVE {self._expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
                f" CONFIGURE_DEPENDS {pattern})\n"
            )
        for pattern in exclude:
            if "**" in pattern:
                raise NotImplementedError("Recursive globs not supported")
            exclude_var = "_GLOB_" + pattern.replace("*", "X").replace(".", "_").upper()
            self._converter.body += (
                f"file(GLOB {exclude_var} LIST_DIRECTORIES false"
                f" RELATIVE {self._expand_cmake_var('CMAKE_CURRENT_SOURCE_DIR')}"
                f" CONFIGURE_DEPENDS {pattern})\n"
            )
            for glob_var in glob_vars:
                self._converter.body += f"list(REMOVE_ITEM {glob_var} {self._expand_cmake_var(exclude_var)})\n"
        return [self._expand_cmake_var(var) for var in glob_vars]

    # TODO(gcmn) implement these types of functions in a less hard-coded way
    def platform_trampoline_deps(self, basename, path="base"):
        return [f"//{path}/internal:{basename}_internal"]

    def select(self, d):
        # Check if all condition keys (except //conditions:default) are known
        # conditions. If so, return a ConditionSelect that the rule handler can
        # convert to CMake if/elseif/else blocks.
        non_default_keys = [k for k in d if k != "//conditions:default"]
        if non_default_keys and all(
            self._convert_select_condition(k) for k in non_default_keys
        ):
            return ConditionSelect(d)
        raise NotImplementedError(f"select: {d}")

    def iree_select(self, selector):
        return self.select(selector)

    def defaulting_select(self, selector):
        """Defined in build_defs.oss.bzl as a scoped alternative to select."""
        default_value = selector.get("//conditions:default")
        if default_value is None:
            raise ValueError("bazel_to_cmake can only convert selects with a default")
        return default_value

    def cc_library(
        self,
        name,
        hdrs=None,
        textual_hdrs=None,
        srcs=None,
        copts=None,
        defines=None,
        data=None,
        deps=None,
        testonly=None,
        linkopts=None,
        includes=None,
        system_includes=None,
        alwayslink=None,
        shared=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        textual_hdrs_block = self._convert_srcs_block(
            textual_hdrs, block_name="TEXTUAL_HDRS"
        )
        srcs_block = self._convert_srcs_block(srcs)
        copts_block, platform_copts_block = self._convert_platform_select_strings(
            name, "COPTS", copts, sort=False
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        alwayslink_block = self._convert_option_block("ALWAYSLINK", alwayslink)
        shared_block = self._convert_option_block("SHARED", shared)
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts
        )
        includes_block = self._convert_includes_block(includes)
        system_includes_block = self._convert_string_list_block(
            "SYSTEM_INCLUDES", system_includes
        )

        self._emit_platform_guard_begin(target_compatible_with)
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        if platform_linkopts_block:
            self._converter.body += platform_linkopts_block
        self._converter.body += (
            f"iree_cc_library(\n"
            f"{name_block}"
            f"{copts_block}"
            f"{hdrs_block}"
            f"{textual_hdrs_block}"
            f"{srcs_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{defines_block}"
            f"{testonly_block}"
            f"{alwayslink_block}"
            f"{shared_block}"
            f"{linkopts_block}"
            f"{includes_block}"
            f"{system_includes_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_cc_library(self, **kwargs):
        self.cc_library(**kwargs)

    def iree_compiler_register_plugin(self, plugin_id, target):
        plugin_id_block = self._convert_string_arg_block(
            "PLUGIN_ID", plugin_id, quote=False
        )
        target_block = self._convert_single_target_block("TARGET", target)
        self._converter.body += (
            f"iree_compiler_register_plugin(\n{plugin_id_block}{target_block})\n\n"
        )

    def cc_test(
        self,
        name,
        hdrs=None,
        srcs=None,
        copts=None,
        defines=None,
        data=None,
        deps=None,
        timeout=None,
        args=None,
        tags=None,
        includes=None,
        group=None,
        resource_group=None,
        sanitizer_suppressions=None,
        target_compatible_with=None,
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
        hdrs_block = self._convert_srcs_block(hdrs, block_name="HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        copts_block, platform_copts_block = self._convert_platform_select_strings(
            name, "COPTS", copts, sort=False
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        data_block = self._convert_target_list_block("DATA", data, omit_empty=True)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        args_block = self._convert_string_list_block("ARGS", args)
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
        if platform_copts_block:
            self._converter.body += platform_copts_block
        if platform_defines_block:
            self._converter.body += platform_defines_block
        if platform_deps_block:
            self._converter.body += platform_deps_block
        self._converter.body += (
            f"iree_cc_test(\n"
            f"{name_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{defines_block}"
            f"{data_block}"
            f"{deps_block}"
            f"{args_block}"
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
        copts_block, platform_copts_block = self._convert_platform_select_strings(
            name, "COPTS", copts, sort=False
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts
        )
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        includes_block = self._convert_includes_block(includes)

        self._emit_platform_guard_begin(target_compatible_with)
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
        defines=None,
        linkopts=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block, platform_deps_block = self._convert_platform_select_deps(name, deps)
        copts_block, platform_copts_block = self._convert_platform_select_strings(
            name, "COPTS", copts, sort=False
        )
        defines_block, platform_defines_block = self._convert_platform_select_strings(
            name, "DEFINES", defines
        )
        linkopts_block, platform_linkopts_block = self._convert_platform_select_strings(
            name, "LINKOPTS", linkopts
        )
        labels_block = self._convert_string_list_block("LABELS", tags)

        self._emit_platform_guard_begin(target_compatible_with)
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

    def iree_c_embed_data(
        self,
        name,
        srcs,
        c_file_output,
        h_file_output,
        testonly=None,
        strip_prefix=None,
        flatten=None,
        identifier=None,
        deps=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(**kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_data_srcs_block(srcs)
        c_file_output_block = self._convert_string_arg_block(
            "C_FILE_OUTPUT", c_file_output
        )
        h_file_output_block = self._convert_string_arg_block(
            "H_FILE_OUTPUT", h_file_output
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        strip_prefix_block = self._convert_option_block("STRIP_PREFIX", strip_prefix)
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        flatten_block = self._convert_option_block("FLATTEN", flatten)
        deps_block = self._convert_target_list_block("DEPS", deps)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_c_embed_data(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{deps_block}"
            f"{c_file_output_block}"
            f"{h_file_output_block}"
            f"{identifier_block}"
            f"{testonly_block}"
            f"{strip_prefix_block}"
            f"{flatten_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _iree_amdgpu_binary(
        self,
        name,
        target,
        arch,
        srcs,
        internal_hdrs=[],
        copts=[],
        linkopts=[],
        out=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        out_block = self._convert_string_arg_block("OUT", out)
        target_block = self._convert_string_arg_block("TARGET", target, quote=False)
        arch_block = self._convert_string_arg_block("ARCH", arch, quote=False)
        hdrs_block = self._convert_srcs_block(internal_hdrs, block_name="INTERNAL_HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        linkopts_block = self._convert_string_list_block(
            "LINKOPTS", linkopts, sort=False
        )

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_amdgpu_binary(\n"
            f"{name_block}"
            f"{out_block}"
            f"{target_block}"
            f"{arch_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{linkopts_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _iree_amdgpu_hal_cts_testdata(
        self,
        name,
        srcs,
        target_selectors_flag,
        format_name,
        format_string,
        identifier,
        backend_name="amdgpu",
        target="amdgcn-amd-amdhsa",
        internal_hdrs=None,
        testonly=True,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        target_block = self._convert_string_arg_block("TARGET", target, quote=False)
        targets_block = self._convert_amdgpu_target_selectors_block(
            target_selectors_flag
        )
        format_name_block = self._convert_string_arg_block(
            "FORMAT_NAME", format_name, quote=False
        )
        if "${" in format_string:
            escaped_format_string = format_string.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            format_string_block = f'  FORMAT_STRING\n    "{escaped_format_string}"\n'
        else:
            format_string_block = f"  FORMAT_STRING\n    [=[{format_string}]=]\n"
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        backend_name_block = self._convert_string_arg_block(
            "BACKEND_NAME", backend_name
        )
        hdrs_block = self._convert_srcs_block(internal_hdrs, block_name="INTERNAL_HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_amdgpu_hal_cts_testdata(\n"
            f"{name_block}"
            f"{target_block}"
            f"{targets_block}"
            f"{format_name_block}"
            f"{format_string_block}"
            f"{identifier_block}"
            f"{backend_name_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _convert_amdgpu_target_selectors_block(self, target_selectors_flag):
        if target_selectors_flag != "//runtime/src/iree/hal/drivers/amdgpu:targets":
            raise NotImplementedError(
                f"AMDGPU target selector flag: {target_selectors_flag}"
            )
        return self._convert_string_list_block(
            "TARGETS", ["${IREE_HAL_AMDGPU_TARGETS}"], quote=False
        )

    def _iree_amdgpu_binary_variants(
        self,
        name,
        target,
        srcs,
        target_selectors_flag,
        binary_name_prefix=None,
        internal_hdrs=None,
        copts=None,
        linkopts=None,
        testonly=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        target_block = self._convert_string_arg_block("TARGET", target, quote=False)
        targets_block = self._convert_amdgpu_target_selectors_block(
            target_selectors_flag
        )
        binary_name_prefix_block = self._convert_string_arg_block(
            "BINARY_NAME_PREFIX", binary_name_prefix, quote=False
        )
        hdrs_block = self._convert_srcs_block(internal_hdrs, block_name="INTERNAL_HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        linkopts_block = self._convert_string_list_block(
            "LINKOPTS", linkopts, sort=False
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_amdgpu_binary_variants(\n"
            f"{name_block}"
            f"{target_block}"
            f"{targets_block}"
            f"{binary_name_prefix_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{linkopts_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _iree_amdgpu_binary_variants_embed_data(
        self,
        name,
        target,
        srcs,
        target_selectors_flag,
        binary_name_prefix=None,
        c_file_output=None,
        h_file_output=None,
        identifier=None,
        flatten=None,
        internal_hdrs=None,
        copts=None,
        linkopts=None,
        deps=None,
        testonly=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        target_block = self._convert_string_arg_block("TARGET", target, quote=False)
        targets_block = self._convert_amdgpu_target_selectors_block(
            target_selectors_flag
        )
        binary_name_prefix_block = self._convert_string_arg_block(
            "BINARY_NAME_PREFIX", binary_name_prefix, quote=False
        )
        c_file_output_block = self._convert_string_arg_block(
            "C_FILE_OUTPUT", c_file_output
        )
        h_file_output_block = self._convert_string_arg_block(
            "H_FILE_OUTPUT", h_file_output
        )
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        hdrs_block = self._convert_srcs_block(internal_hdrs, block_name="INTERNAL_HDRS")
        srcs_block = self._convert_srcs_block(srcs)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        linkopts_block = self._convert_string_list_block(
            "LINKOPTS", linkopts, sort=False
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)
        flatten_block = self._convert_option_block("FLATTEN", flatten)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_amdgpu_binary_variants_embed_data(\n"
            f"{name_block}"
            f"{target_block}"
            f"{targets_block}"
            f"{binary_name_prefix_block}"
            f"{c_file_output_block}"
            f"{h_file_output_block}"
            f"{identifier_block}"
            f"{hdrs_block}"
            f"{srcs_block}"
            f"{copts_block}"
            f"{linkopts_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"{flatten_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _iree_hal_amdgpu_source_device_binaries(self):
        self._converter.body += (
            "# Source-built AMDGPU device binary targets are wired manually by\n"
            "# runtime/src/iree/hal/drivers/amdgpu/device/binaries/CMakeLists.txt.\n\n"
        )

    def iree_bytecode_module(
        self,
        name,
        src,
        module_name=None,
        flags=None,
        compile_tool=None,
        c_identifier=None,
        static_lib_path=None,
        deps=None,
        testonly=None,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        src_block = self._convert_string_arg_block("SRC", src)
        module_name_block = self._convert_string_arg_block(
            "MODULE_FILE_NAME", module_name
        )
        c_identifier_block = self._convert_string_arg_block(
            "C_IDENTIFIER", c_identifier
        )
        static_lib_block = self._convert_string_arg_block(
            "STATIC_LIB_PATH", static_lib_path
        )
        compile_tool_block = self._convert_target_block("COMPILE_TOOL", compile_tool)
        flags_block = self._convert_string_list_block("FLAGS", flags)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._converter.body += (
            f"iree_bytecode_module(\n"
            f"{name_block}"
            f"{src_block}"
            f"{module_name_block}"
            f"{c_identifier_block}"
            f"{compile_tool_block}"
            f"{static_lib_block}"
            f"{flags_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )

    def iree_vmasm_module(
        self,
        name,
        src,
        module_name=None,
        assemble_tool=None,
        c_identifier=None,
        deps=None,
        testonly=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        src_block = self._convert_string_arg_block("SRC", src)
        module_name_block = self._convert_string_arg_block(
            "MODULE_FILE_NAME", module_name
        )
        assemble_tool_block = self._convert_target_block("ASSEMBLE_TOOL", assemble_tool)
        c_identifier_block = self._convert_string_arg_block(
            "C_IDENTIFIER", c_identifier
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_vmasm_module(\n"
            f"{name_block}"
            f"{src_block}"
            f"{module_name_block}"
            f"{assemble_tool_block}"
            f"{c_identifier_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_spirv_asm_module(
        self,
        name,
        src,
        out=None,
        target_env=None,
        spirv_as_args=None,
        assemble_tool=None,
        c_identifier=None,
        deps=None,
        testonly=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        if target_compatible_with is None:
            target_compatible_with = _SPIRV_TOOL_TARGET_COMPATIBLE_WITH
        out_file = out or ("%s.spv" % name)
        self._target_file_paths[self._current_target_label(name)] = out_file
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        src_block = self._convert_string_arg_block("SRC", src)
        out_block = self._convert_string_arg_block("OUT", out)
        target_env_block = self._convert_string_arg_block("TARGET_ENV", target_env)
        spirv_as_args_block = self._convert_string_list_block(
            "SPIRV_AS_ARGS", spirv_as_args, sort=False
        )
        assemble_tool_block = self._convert_target_block("ASSEMBLE_TOOL", assemble_tool)
        c_identifier_block = self._convert_string_arg_block(
            "C_IDENTIFIER", c_identifier
        )
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_spirv_asm_module(\n"
            f"{name_block}"
            f"{src_block}"
            f"{out_block}"
            f"{target_env_block}"
            f"{spirv_as_args_block}"
            f"{assemble_tool_block}"
            f"{c_identifier_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_hal_executable(
        self,
        name,
        src,
        target_device,
        flags=None,
        executable_name=None,
        compile_tool=None,
        linker_tool=None,
        c_identifier=None,
        deps=None,
        testonly=None,
        **kwargs,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        src_block = self._convert_string_arg_block("SRC", self._normalize_label(src))
        target_device_block = self._convert_string_arg_block(
            "TARGET_DEVICE", target_device
        )
        executable_name_block = self._convert_string_arg_block(
            "EXECUTABLE_FILE_NAME", executable_name
        )
        c_identifier_block = self._convert_string_arg_block(
            "C_IDENTIFIER", c_identifier
        )
        compile_tool_block = self._convert_target_block("COMPILE_TOOL", compile_tool)
        flags_block = self._convert_string_list_block("FLAGS", flags)
        deps_block = self._convert_target_list_block("DEPS", deps)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._converter.body += (
            f"iree_hal_executable(\n"
            f"{name_block}"
            f"{src_block}"
            f"{target_device_block}"
            f"{executable_name_block}"
            f"{c_identifier_block}"
            f"{compile_tool_block}"
            f"{flags_block}"
            f"{deps_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )

    def iree_hal_executables(
        self,
        name,
        srcs,
        target_device,
        flags=None,
        identifier=None,
        compile_tool=None,
        testonly=None,
        **kwargs,
    ):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        target_device_block = self._convert_string_arg_block(
            "TARGET_DEVICE", target_device
        )
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        compile_tool_block = self._convert_target_block("COMPILE_TOOL", compile_tool)
        flags_block = self._convert_string_list_block("FLAGS", flags)
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._converter.body += (
            f"iree_hal_executables(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{target_device_block}"
            f"{identifier_block}"
            f"{compile_tool_block}"
            f"{flags_block}"
            f"{testonly_block}"
            f"  PUBLIC\n)\n\n"
        )

    def _iree_hal_cts_testdata(
        self,
        format_name,
        target_device,
        identifier,
        backend_name,
        format_string,
        testdata,
        flags=None,
        flag_values=None,
        cmake_format_variant_values=None,
        data=None,
        testonly=None,
        target_compatible_with=None,
        **kwargs,
    ):
        variant_placeholders = set(cmake_format_variant_values or [])
        variant_token = None
        variant_values_var = None
        if len(variant_placeholders) > 1:
            raise NotImplementedError(
                "iree_hal_cts_testdata supports one CMake format variant value"
            )
        if variant_placeholders:
            if not flag_values:
                raise ValueError(
                    "cmake_format_variant_values requires matching flag_values"
                )
            variant_placeholder = next(iter(variant_placeholders))
            variant_label = flag_values.get(variant_placeholder)
            if not variant_label:
                raise ValueError(
                    f"cmake_format_variant_values entry {variant_placeholder} "
                    "is missing from flag_values"
                )
            variant_values_var = _BUILD_SETTING_CMAKE_LIST_VARIABLES.get(variant_label)
            if not variant_values_var:
                raise NotImplementedError(
                    f"No CMake list variable mapping for {variant_label}"
                )
            variant_token = "{" + variant_placeholder + "}"

        # Resolve {PLACEHOLDER} template variables from flag_values.
        # Build settings map to CMake variables; file targets (not in the
        # mapping) have their flags stripped since CMake auto-discovers
        # platform libraries via findPlatformLibDirectory().
        if flag_values:
            file_templates = set()
            for placeholder, label in flag_values.items():
                cmake_var = _BUILD_SETTING_CMAKE_VARIABLES.get(label)
                template = "{" + placeholder + "}"
                if cmake_var is not None:
                    if placeholder not in variant_placeholders:
                        cmake_ref = "${" + cmake_var + "}"
                        format_string = format_string.replace(template, cmake_ref)
                        if flags:
                            flags = [f.replace(template, cmake_ref) for f in flags]
                else:
                    file_templates.add(template)
            if flags and file_templates:
                flags = [f for f in flags if not any(t in f for t in file_templates)]

        name_block = self._convert_string_arg_block(
            "FORMAT_NAME", format_name, quote=False
        )
        variant_token_block = self._convert_string_arg_block(
            "FORMAT_VARIANT_TOKEN", variant_token
        )
        if variant_values_var:
            format_variants_block = self._convert_string_list_block(
                "FORMAT_VARIANTS", [f"${{{variant_values_var}}}"], quote=False
            )
        else:
            format_variants_block = ""
        target_device_block = self._convert_string_arg_block(
            "TARGET_DEVICE", target_device
        )
        identifier_block = self._convert_string_arg_block("IDENTIFIER", identifier)
        backend_name_block = self._convert_string_arg_block(
            "BACKEND_NAME", backend_name
        )
        # Bracket-quote C expressions like "embedded-elf-" IREE_ARCH so CMake
        # leaves them alone. If placeholder substitution produced a CMake
        # variable reference, use a normal quoted argument so the generated
        # testdata registration contains the configured value instead of the
        # literal ${...} token.
        if "${" in format_string:
            escaped_format_string = format_string.replace("\\", "\\\\").replace(
                '"', '\\"'
            )
            format_string_block = f'  FORMAT_STRING\n    "{escaped_format_string}"\n'
        else:
            format_string_block = f"  FORMAT_STRING\n    [=[{format_string}]=]\n"
        flags_block = self._convert_string_list_block("FLAGS", flags)

        # Convert Bazel label to CMake directory path.
        # "//runtime/src/iree/hal/cts/testdata:executable_srcs"
        #   -> "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/testdata"
        testdata_dir = testdata.split(":")[0].lstrip("/")
        testdata_dir_block = (
            f'  TESTDATA_DIR\n    "${{PROJECT_SOURCE_DIR}}/{testdata_dir}"\n'
        )

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_hal_cts_testdata(\n"
            f"{name_block}"
            f"{variant_token_block}"
            f"{format_variants_block}"
            f"{target_device_block}"
            f"{identifier_block}"
            f"{backend_name_block}"
            f"{format_string_block}"
            f"{testdata_dir_block}"
            f"{flags_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def _iree_hal_cts_test_suite(
        self,
        backends_lib,
        executable_formats=None,
        testdata_libs=None,
        testdata=None,
        flag_values=None,
        name="",
        args=None,
        resource_group=None,
        tags=None,
        testonly=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if not resource_group and tags:
            for tag in tags:
                if tag.startswith("resource_group:"):
                    resource_group = tag[len("resource_group:") :]
                    break

        # Expand executable_formats into individual iree_hal_cts_testdata()
        # calls. The CMake function takes only flat TESTDATA_LIBS, avoiding
        # nested dict argument parsing.
        _testdata_libs = list(testdata_libs or [])
        if executable_formats:
            for format_name, config in executable_formats.items():
                self._iree_hal_cts_testdata(
                    format_name=format_name,
                    target_device=config["target_device"],
                    identifier=config["identifier"],
                    backend_name=config["backend_name"],
                    format_string=config["format_string"],
                    testdata=testdata,
                    flag_values=flag_values,
                    flags=config.get("flags"),
                    target_compatible_with=target_compatible_with,
                )
                _testdata_libs.append(f":testdata_{format_name}_lib")

        # Use _convert_target_list_block for BACKENDS_LIB so that local
        # targets like ":backends" preserve their "::" CMake alias form.
        # (_convert_target_block replaces "::" with "_", which is wrong
        # for local package-relative references.)
        backends_block = self._convert_target_list_block(
            "BACKENDS_LIB", [backends_lib] if backends_lib else None
        )
        testdata_libs_block = self._convert_target_list_block(
            "TESTDATA_LIBS", _testdata_libs if _testdata_libs else None
        )
        name_block = (
            self._convert_string_arg_block("NAME", name, quote=False) if name else ""
        )
        args_block = self._convert_string_list_block("ARGS", args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        resource_group_block = self._convert_string_arg_block(
            "RESOURCE_GROUP", resource_group, quote=False
        )
        testonly_block = self._convert_option_block("TESTONLY", testonly)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_hal_cts_test_suite(\n"
            f"{backends_block}"
            f"{testdata_libs_block}"
            f"{name_block}"
            f"{args_block}"
            f"{labels_block}"
            f"{resource_group_block}"
            f"{testonly_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_flatbuffer_c_library(
        self,
        name,
        srcs,
        flatcc_args=None,
        includes=None,
        deps=None,
        tags=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        flatcc_args_block = self._convert_string_list_block("FLATCC_ARGS", flatcc_args)
        includes_block = self._convert_srcs_block(includes, block_name="INCLUDES")
        deps_block = self._convert_target_list_block("DEPS", deps)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"flatbuffer_c_library(\n"
            f"{name_block}"
            f"{srcs_block}"
            f"{flatcc_args_block}"
            f"{includes_block}"
            f"{deps_block}"
            f"  PUBLIC\n)\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_execution_test_suite(
        self,
        name,
        manifests,
        tools,
        data=None,
        args=None,
        sanitizer_suppressions=None,
        tags=None,
        timeout=None,
        target_compatible_with=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        self._check_no_unhandled_kwargs("iree_execution_test_suite", kwargs)

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        manifests_block = self._convert_srcs_block(manifests, block_name="MANIFESTS")
        data_block = self._convert_srcs_block(data, block_name="DATA")
        args_block = self._convert_string_list_block("ARGS", args)
        sanitizer_suppressions_block = self._convert_sanitizer_suppressions_block(
            sanitizer_suppressions
        )
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        tool_entries = []
        for tool_name, tool_target in sorted(tools.items()):
            tool_entries.append(
                f"{tool_name}={self._convert_single_target(tool_target)}"
            )
        tools_block = self._convert_string_list_block("TOOLS", tool_entries)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_execution_test_suite(\n"
            f"{name_block}"
            f"{manifests_block}"
            f"{tools_block}"
            f"{data_block}"
            f"{args_block}"
            f"{sanitizer_suppressions_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_generated_e2e_runner_test(
        self,
        name,
        test_type,
        generator,
        generator_args=None,
        test_runner=None,
        target_backends_and_drivers=None,
        compiler_flags=None,
        runner_args=None,
        tags=None,
        timeout=None,
        target_cpu_features_variants=None,
        **kwargs,
    ):
        if self._should_skip_target(tags=tags, **kwargs):
            return
        target_backends = None
        drivers = None
        if target_backends_and_drivers is not None:
            target_backends = [it[0] for it in target_backends_and_drivers]
            drivers = [it[1] for it in target_backends_and_drivers]

        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        test_type_block = self._convert_string_arg_block(
            "TEST_TYPE", test_type, quote=False
        )
        # For now we assume that the generator target is a py_binary with a single
        # source .py file named like it.
        generator_py = f"{generator.split(':')[-1]}.py"
        generator_block = self._convert_string_arg_block(
            "GENERATOR", generator_py, quote=True
        )
        generator_args_block = self._convert_string_list_block(
            "GENERATOR_ARGS", generator_args
        )
        test_runner_block = self._convert_target_block("TEST_RUNNER", test_runner)
        target_backends_block = self._convert_string_list_block(
            "TARGET_BACKENDS", target_backends
        )
        drivers_block = self._convert_string_list_block("DRIVERS", drivers)
        compiler_flags_block = self._convert_string_list_block(
            "COMPILER_FLAGS", compiler_flags
        )
        runner_args_block = self._convert_string_list_block("RUNNER_ARGS", runner_args)
        labels_block = self._convert_string_list_block("LABELS", tags)
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)
        target_cpu_features_variants_block = self._convert_string_list_block(
            "TARGET_CPU_FEATURES_VARIANTS", target_cpu_features_variants
        )

        self._converter.body += (
            f"iree_generated_e2e_runner_test(\n"
            f"{name_block}"
            f"{test_type_block}"
            f"{generator_block}"
            f"{generator_args_block}"
            f"{test_runner_block}"
            f"{target_backends_block}"
            f"{drivers_block}"
            f"{compiler_flags_block}"
            f"{runner_args_block}"
            f"{labels_block}"
            f"{timeout_block}"
            f"{target_cpu_features_variants_block}"
            f")\n\n"
        )

    def native_test(
        self,
        name,
        src,
        args=None,
        data=None,
        env=None,
        sanitizer_suppressions=None,
        tags=None,
        timeout=None,
        target_compatible_with=None,
    ):
        if self._should_skip_target(tags=tags):
            return

        name_block = self._convert_string_arg_block("NAME", name)
        test_binary_block = self._convert_single_target_block("SRC", src)
        args_block = self._convert_string_list_block(
            "ARGS", self._convert_native_test_location_args(args)
        )
        labels_block = self._convert_string_list_block("LABELS", tags)
        sanitizer_suppressions_block = self._convert_sanitizer_suppressions_block(
            sanitizer_suppressions
        )
        timeout_block = self._convert_timeout_arg_block("TIMEOUT", timeout)

        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += (
            f"iree_native_test(\n"
            f"{name_block}"
            f"{args_block}"
            f"{test_binary_block}"
            f"{labels_block}"
            f"{sanitizer_suppressions_block}"
            f"{timeout_block}"
            f")\n\n"
        )
        self._emit_platform_guard_end(target_compatible_with)

    def iree_executable_test(self, src, **kwargs):
        self.native_test(src=src, **kwargs)

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
    ):
        if self._should_skip_target(tags=tags):
            return
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        data_block = self._convert_target_list_block("DATA", data)
        deps_block = self._convert_target_list_block("DEPS", deps)
        copts_block = self._convert_string_list_block("COPTS", copts, sort=False)
        defines_block = self._convert_string_list_block("DEFINES", defines)
        linkopts_block = self._convert_string_list_block("LINKOPTS", linkopts)
        args_block = self._convert_string_list_block("ARGS", args)
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

    def iree_cmake_extra_content(self, content, inline=False):
        if inline:
            self._converter.body += f"\n{content}\n"
        else:
            self._converter.header += f"\n{content}\n"

    def iree_genrule(self, name, srcs, outs, cmd):
        name_block = self._convert_string_arg_block("NAME", name, quote=False)
        srcs_block = self._convert_srcs_block(srcs)
        outs_block = self._convert_target_list_block("OUTS", outs)
        cmd_block = self._convert_string_arg_block("CMD", cmd, quote=True)

        self._converter.body += (
            f"iree_genrule(\n{name_block}{srcs_block}{outs_block}{cmd_block})\n\n"
        )


class Converter(object):
    """Conversion state tracking and full file template substitution."""

    def __init__(self):
        # Header appears after the license block but before `iree_add_all_subdirs`.
        self.header = ""
        # Body appears after `iree_add_all_subdirs`.
        self.body = ""

    def convert(self):
        converted_content = f"{self.header}\n\niree_add_all_subdirs()\n\n{self.body}"

        # Cleanup newline characters. This is more convenient than ensuring all
        # conversions are careful with where they insert newlines.
        converted_content = converted_content.replace("\n\n\n", "\n")
        converted_content = converted_content.rstrip() + "\n"

        return converted_content


def GetDict(obj):
    ret = {}
    for k in dir(obj):
        if not k.startswith("_"):
            ret[k] = getattr(obj, k)
    return ret


def convert_build_file(
    build_file_code,
    repo_cfg,
    build_dir,
    repo_root="",
):
    converter = Converter()
    # Allow overrides of TargetConverter and BuildFileFunctions from repo cfg.
    repo_map = getattr(repo_cfg, "REPO_MAP", {})
    target_converter = bazel_to_cmake_config.create_target_converter(
        repo_cfg,
        repo_map,
        bazel_to_cmake_targets.TargetConverter,
    )
    build_file_functions_class = bazel_to_cmake_config.select_build_file_functions(
        repo_cfg,
        build_dir,
        repo_root,
        BuildFileFunctions,
    )
    build_file_functions = build_file_functions_class(
        converter=converter,
        targets=target_converter,
        build_dir=build_dir,
        repo_root=repo_root,
    )

    exec_namespace = GetDict(build_file_functions)
    build_file_functions._exec_namespace = exec_namespace
    exec(build_file_code, exec_namespace)
    return converter.convert()
