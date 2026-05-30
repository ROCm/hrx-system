# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(iree-c-embed-data embed_data.cc)
set_target_properties(iree-c-embed-data PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  FOLDER "${HRX_IDE_FOLDER}/build_tools"
  RUNTIME_OUTPUT_DIRECTORY "${IREE_BINARY_DIR}/build_tools"
)

# This executable is a host-side generator that CMake runs while building the
# tree. MSAN-instrumenting it makes the build depend on an instrumented host
# C++ runtime, which our CI images intentionally do not provide. Keep MSAN on
# the actual runtime/test binaries while leaving this build helper clean.
if(IREE_ENABLE_MSAN)
  target_compile_options(iree-c-embed-data PRIVATE "-fno-sanitize=memory")
  target_link_options(iree-c-embed-data PRIVATE "-fno-sanitize=memory")
endif()
