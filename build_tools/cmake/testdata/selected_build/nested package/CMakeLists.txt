# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach(_NAME left right)
  add_custom_target(${_NAME}_test_deps)
  add_dependencies(${_NAME}_test_deps ${_NAME})
  add_test(NAME ${_NAME} COMMAND "$<TARGET_FILE:${_NAME}>"
    "${PROJECT_SOURCE_DIR}/expected.txt"
    "${PROJECT_BINARY_DIR}/${_NAME}.ran")
  iree_configure_test(${_NAME})
  iree_register_test_build_targets(${_NAME} TARGETS ${_NAME}_test_deps)
endforeach()
