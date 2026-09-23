#!/usr/bin/env bash
set -euo pipefail

expected_path="${TEST_SRCDIR}/_main/build_tools/bazel/test/native_artifact_import_fixture.data"
[[ "${PWD}" == "${TEST_SRCDIR}/_main" ]]
[[ "${RUNFILES_DIR}" == "${TEST_SRCDIR}" ]]
[[ "${JAVA_RUNFILES}" == "${TEST_SRCDIR}" ]]
[[ "${PYTHON_RUNFILES}" == "${TEST_SRCDIR}" ]]
[[ "$#" == 1 ]]
[[ "$1" == "${expected_path}" ]]
[[ "${NATIVE_ARTIFACT_FIXTURE_PATH}" == "${expected_path}" ]]
[[ "$(cat "${expected_path}")" == "native artifact fixture" ]]
