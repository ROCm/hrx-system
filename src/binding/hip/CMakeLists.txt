# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# HIP binding — libamdhip64.so drop-in replacement.

add_library(amdhip64 SHARED
    api.c
)

target_link_libraries(amdhip64 PRIVATE
    pyre_binding_common
)

target_compile_definitions(amdhip64 PRIVATE IREE_HAL_STREAMING_HIP_EXPORTS)

set_target_properties(amdhip64 PROPERTIES
    OUTPUT_NAME amdhip64
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)
