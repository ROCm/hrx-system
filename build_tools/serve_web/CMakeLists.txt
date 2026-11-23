# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Resolve the host tools only when building the embedded dashboard. A consumer
# configuring another Loom target does not need Node or a package manager.
set(_stage "${CMAKE_CURRENT_BINARY_DIR}/tools")
foreach(_file IN ITEMS package.json pnpm-lock.yaml pnpm-workspace.yaml pack.mjs run_pack.py)
  configure_file("${_file}" "${_stage}/${_file}" COPYONLY)
endforeach()
find_program(_loom_pnpm NAMES pnpm pnpm.cmd)
if(_loom_pnpm)
  set(_install "${_loom_pnpm}")
else()
  set(_install corepack pnpm)
endif()
set(_installed "${_stage}/node_modules/.loom-installed")
add_custom_command(
  OUTPUT "${_installed}"
  COMMAND ${_install} install --frozen-lockfile --ignore-scripts
  COMMAND "${CMAKE_COMMAND}" -E touch "${_installed}"
  DEPENDS "${_stage}/package.json" "${_stage}/pnpm-lock.yaml" "${_stage}/pnpm-workspace.yaml"
  WORKING_DIRECTORY "${_stage}"
  COMMENT "Install locked dashboard build tools"
  VERBATIM
)

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/build_tools/serve_web")
iree_py_library(
  NAME pack
  MAIN "${_stage}/run_pack.py"
  SRCS "${_stage}/run_pack.py" "${_stage}/pack.mjs" "${_installed}"
)
