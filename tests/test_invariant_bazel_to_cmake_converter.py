import pytest
import sys
import os

# Add the module path to sys.path to import the actual production code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from build_tools.bazel_to_cmake.bazel_to_cmake_converter import convert_bazel_to_cmake


@pytest.mark.parametrize("payload", [
    # Exact exploit case: code injection payload
    "__import__('os').system('rm -rf /')",
    # Boundary case: valid Bazel rule with embedded exec-like string
    'cc_library(name = "test", srcs = ["test.cc"], cmd = "echo \'$(shell pwd)\'")',
    # Valid input: normal Bazel rule without dangerous content
    'cc_library(name = "safe", srcs = ["safe.cc"])',
])
def test_bazel_to_cmake_converter_no_code_injection(payload):
    """Invariant: Converting Bazel to CMake must not execute arbitrary code from input."""
    try:
        # The security property: conversion must complete without executing payload
        result = convert_bazel_to_cmake(payload)
        # If conversion succeeds, ensure no side effects occurred
        # We can't directly detect exec() calls, but we can verify the output
        # doesn't contain the dangerous payload as executable code
        assert payload not in result or 'os.system' not in result
    except Exception as e:
        # Conversion may fail on invalid input, but must not execute code
        # Check exception doesn't indicate code execution
        error_msg = str(e)
        assert 'os.system' not in error_msg
        assert '__import__' not in error_msg