# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cquery formatter for configured Windows test handoff metadata."""

_ARGUMENTS_PROVIDER_SUFFIX = "%IreeRunfilesArgumentsInfo"
_ENVIRONMENT_PROVIDER_SUFFIX = "%IreeRunfilesEnvironmentInfo"

def _providers_with_suffix(all_providers, suffix):
    return [
        value
        for key, value in all_providers.items()
        if key.endswith(suffix)
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
        "executable": executable.path if executable else None,
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
    })
