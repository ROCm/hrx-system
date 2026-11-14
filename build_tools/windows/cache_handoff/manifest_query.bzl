# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cquery formatter for configured Windows test handoff metadata."""

_ARGUMENTS_PROVIDER_SUFFIX = "%IreeRunfilesArgumentsInfo"
_ENVIRONMENT_PROVIDER_SUFFIX = "%IreeRunfilesEnvironmentInfo"
_EXECUTABLE_PROVIDER_SUFFIX = "%IreeExecutableInfo"

def _providers_with_suffix(all_providers, suffix):
    return [
        value
        for key, value in all_providers.items()
        if key.endswith(suffix)
    ]

def _runfile_path(file):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return "_main/" + file.short_path

def _add_runfile(runfiles, logical_path, physical_path):
    prior_path = runfiles.get(logical_path)
    if prior_path != None and prior_path != physical_path:
        fail("runfile %s maps to both %s and %s" % (logical_path, prior_path, physical_path))
    runfiles[logical_path] = physical_path

def _collect_runfiles(default_info, executable):
    runfiles = {}
    default_runfiles = default_info.default_runfiles
    for file in default_runfiles.files.to_list() + [executable]:
        _add_runfile(runfiles, _runfile_path(file), file.path)
    for path in default_runfiles.empty_filenames.to_list():
        _add_runfile(runfiles, "_main/" + path, "")
    for entry in default_runfiles.symlinks.to_list():
        _add_runfile(runfiles, "_main/" + entry.path, entry.target_file.path)
    for entry in default_runfiles.root_symlinks.to_list():
        _add_runfile(runfiles, entry.path, entry.target_file.path)
    repo_mapping_manifest = default_info.files_to_run.repo_mapping_manifest
    if repo_mapping_manifest:
        _add_runfile(runfiles, "_repo_mapping", repo_mapping_manifest.path)
    return [
        {
            "logical_path": logical_path,
            "physical_path": physical_path,
        }
        for logical_path, physical_path in sorted(runfiles.items())
    ]

def format(target):
    """Serializes providers needed to reproduce the configured test launch.

    Args:
      target: Configured Bazel test target.

    Returns:
      JSON object containing its executable and launch providers.
    """
    all_providers = providers(target)
    default_info = all_providers["DefaultInfo"]
    executable = default_info.files_to_run.executable
    argument_providers = _providers_with_suffix(
        all_providers,
        _ARGUMENTS_PROVIDER_SUFFIX,
    )
    environment_providers = _providers_with_suffix(
        all_providers,
        _ENVIRONMENT_PROVIDER_SUFFIX,
    )
    executable_providers = _providers_with_suffix(
        all_providers,
        _EXECUTABLE_PROVIDER_SUFFIX,
    )
    if len(executable_providers) > 1:
        fail("%s has ambiguous executable providers" % target.label)
    selected_executable = (
        executable_providers[0].src_output if executable_providers else executable
    )
    run_environment = all_providers.get("RunEnvironmentInfo")
    return json.encode({
        "argument_provider_count": len(argument_providers),
        "arguments": [
            argument
            for provider in argument_providers
            for argument in provider.arguments
        ],
        "compatible": "IncompatiblePlatformProvider" not in all_providers,
        "environment": dict(run_environment.environment) if run_environment else {},
        "executable": selected_executable.path if selected_executable else None,
        "executable_provider_count": len(executable_providers),
        "inherited_environment": sorted(run_environment.inherited_environment) if run_environment else [],
        "label": str(target.label),
        "marked_arguments": [
            argument
            for provider in argument_providers
            for argument in provider.marked_arguments
        ],
        "marked_environment": {
            name: marked_value
            for provider in environment_providers
            for name, marked_value in provider.environment.items()
        },
        "runfiles": _collect_runfiles(default_info, selected_executable) if selected_executable else [],
    })
