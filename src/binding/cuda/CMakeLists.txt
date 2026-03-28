# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# CUDA binding — CUDA runtime/driver API compatibility layer.

add_library(cuda SHARED
    driver.c
    runtime.c
)

target_include_directories(cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(cuda PRIVATE
    pyre_streaming
    iree_base_base
    iree_hal_hal
)

target_compile_definitions(cuda PRIVATE IREE_HAL_STREAMING_CUDA_EXPORTS)

set_target_properties(cuda PROPERTIES
    OUTPUT_NAME cuda
    VERSION 1.0.0
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)
