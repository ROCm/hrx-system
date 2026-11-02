# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Native tests from resolved source arguments and data ownership."""

import re

from native_cc import arguments, command


def resolve_arguments(values):
    """Resolves file spans once, retaining each path as a runtime requirement."""
    files = []
    result = []
    for value in values or ():
        pieces = []
        while "{{" in value:
            prefix, _, value = value.partition("{{")
            path, separator, value = value.partition("}}")
            if not separator or not path:
                raise ValueError("Invalid file locator in test argument")
            if not path.startswith(("/", "$", "\\")) and not re.match(
                r"^[A-Za-z]:[/\\]", path
            ):
                path = "${PROJECT_SOURCE_DIR}/" + path
            files.append(path)
            pieces.extend((prefix, path))
        pieces.append(value)
        result.append("".join(pieces))
    return result, files


def emit(
    target,
    name,
    subject,
    *,
    args,
    environment,
    file_data,
    target_data,
    producers,
    labels,
    suppressions,
    resource_group,
    timeout,
    working_directory,
    dependencies,
):
    """Keeps native executable launchers and suppression providers late-bound."""
    package = name.rpartition("/")[0]
    all_targets = tuple(
        dict.fromkeys(
            (
                subject,
                *target_data,
                *dependencies,
                *(producers[path] for path in file_data if path in producers),
            )
        )
    )
    required = (*file_data, *("$<TARGET_FILE:" + item + ">" for item in target_data))
    # Most tests inherit exactly their executable's runtime requirements. Only
    # a composed environment needs its own native property owner; it has no
    # build action and remains late-bound to authored suppression providers.
    owner = subject
    lines = []
    if (
        suppressions
        or target_data
        or any(
            re.match(r"^(ASAN|LSAN|MSAN|TSAN|UBSAN)_OPTIONS=", value)
            for value in environment
        )
    ):
        owner = target + "_environment"
        lines.append(command("add_library", owner, "INTERFACE"))
        lines.append(
            command(
                "iree_target_sanitizer_suppressions",
                owner,
                "DEPS",
                arguments((subject, *target_data)),
                "SUPPRESSIONS",
                arguments(suppressions),
                "ENV",
                arguments(environment),
            )
        )
    lines.extend(
        [
            command(
                "add_test",
                "NAME",
                arguments((name,)),
                "COMMAND",
                arguments((subject, *args)),
            ),
            command("iree_configure_test", arguments((name,))),
            command(
                "iree_register_test_build_targets",
                arguments((name,)),
                "DEPENDENCIES",
                arguments(all_targets),
                "FILES",
                arguments(path for path in file_data if path not in producers),
                "RESOURCES",
                arguments(
                    label.split("=", 1)[1]
                    for label in labels
                    if label.startswith("runtime-resource=")
                ),
            ),
        ]
    )
    properties = {
        "WORKING_DIRECTORY": working_directory,
        "LABELS": ";".join((*labels, package)),
        "REQUIRED_FILES": ";".join(required),
        "TIMEOUT": str(timeout or 60),
    }
    if resource_group:
        properties["RESOURCE_LOCK"] = resource_group
    lines.append(
        command(
            "set_tests_properties",
            arguments((name,)),
            "PROPERTIES",
            *(key + " " + arguments((value,)) for key, value in properties.items()),
        )
    )
    lines.append(
        command(
            "set_property",
            "TEST",
            arguments((name,)),
            "APPEND PROPERTY ENVIRONMENT",
            arguments(
                (
                    *environment,
                    "$<TARGET_PROPERTY:" + owner + ",IREE_SANITIZER_ENVIRONMENT>",
                )
            ),
        )
    )
    return "".join(lines)


class BuildFileFunctions:
    def native_test(
        self,
        name,
        src,
        args=None,
        data=None,
        env=None,
        sanitizer_suppressions=None,
        tags=None,
        resource_group=None,
        timeout=None,
        target_compatible_with=None,
        visibility=None,
    ):
        if self._should_skip_target(tags=tags):
            return
        self._emit_platform_guard_begin(target_compatible_with)
        self._converter.body += "if(IREE_BUILD_TESTS)\n"
        self._converter.defer(
            lambda: self._native_test(
                name,
                self._convert_single_target(src),
                args=self._convert_test_location_args(args),
                data=sorted(self._convert_data_locations(data) or []),
                environment=self._convert_test_env(env),
                labels=sorted(tags or []),
                suppressions=[
                    value
                    for kind, label in sorted((sanitizer_suppressions or {}).items())
                    for value in (kind, self._sanitizer_suppression_name(label))
                ],
                resource_group=resource_group,
                timeout=self._timeout_map[timeout] if timeout else None,
            )
        )
        self._converter.body += "endif()\n"
        self._emit_platform_guard_end(target_compatible_with)

    def _native_test(
        self,
        name,
        subject,
        *,
        args=None,
        data=None,
        environment=None,
        labels=None,
        suppressions=(),
        resource_group=None,
        timeout=None,
        working_directory=None,
        dependencies=(),
    ):
        """Renders one final test after source producers have been collected."""
        qualified = self._convert_single_target(self._current_target_label(name))
        namespace = qualified.rpartition("::")[0]
        if subject.startswith("::"):
            subject = namespace + subject
        args, argument_files = resolve_arguments(args)
        environment, environment_files = resolve_arguments(environment)
        files, targets = [], []
        for value in dict.fromkeys(
            (*(data or ()), *argument_files, *environment_files)
        ):
            if value.startswith("$<TARGET_FILE:") and value.endswith(">"):
                targets.append(value[len("$<TARGET_FILE:") : -1])
            elif value.startswith(("$", "/", "\\")) or re.match(
                r"^[A-Za-z]:[/\\]", value
            ):
                files.append(value)
            else:
                # The location resolver has already distinguished source files
                # from target DATA, including opaque imported native providers.
                targets.append(value)
        targets = [
            namespace + target if target.startswith("::") else target
            for target in targets
        ]
        return emit(
            qualified.replace("::", "_"),
            qualified.replace("::", "/"),
            subject,
            args=args,
            environment=environment,
            file_data=files,
            target_data=targets,
            producers=self._libraries.producers if self._libraries else {},
            labels=labels or (),
            suppressions=suppressions,
            resource_group=resource_group,
            timeout=timeout,
            working_directory=working_directory or self._package_binary_dir(),
            dependencies=dependencies,
        )
