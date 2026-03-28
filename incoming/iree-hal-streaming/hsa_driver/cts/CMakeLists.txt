# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    backends
  SRCS
    "backends.cc"
  DEPS
    iree::async::util::proactor_pool
    iree::base::threading
    iree::hal
    iree::hal::cts::util::registry
    hsa_driver::registration
  TESTONLY
  ALWAYSLINK
  PUBLIC
)

# TODO(awoloszyn): Add executable testdata when HSA executable support is ready.
# iree_hal_cts_testdata(
#   FORMAT_NAME
#     hsa
#   TARGET_DEVICE
#     "hsa"
#   IDENTIFIER
#     "iree_cts_testdata_hsa"
#   BACKEND_NAME
#     "hsa"
#   FORMAT_STRING
#     [=["HSACO"]=]
#   TESTDATA_DIR
#     "${PROJECT_SOURCE_DIR}/runtime/src/iree/hal/cts/testdata"
#   FLAGS
#     "--iree-hsa-target-chip=gfx906" # Example chip
# )

iree_hal_cts_test_suite(
  BACKENDS_LIB
    ::backends
  # TESTDATA_LIBS
  #   ::testdata_hsa_lib # Uncomment when executable support is ready
  NAME
    "" # No prefix, as this is the primary driver
  LABELS
    "driver=hsa"
    "requires-gpu-amd"
)
