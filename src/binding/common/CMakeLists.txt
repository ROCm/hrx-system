# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# Common binding runtime — shared by HIP and CUDA bindings.

add_library(pyre_binding_common STATIC
    context.c
    device.c
    event.c
    graph.c
    graph_analysis.c
    graph_exec.c
    init.c
    mem_pool.c
    memory.c
    module.c
    peer.c
    registry.c
    stream.c
)

target_include_directories(pyre_binding_common PUBLIC
    ${PROJECT_SOURCE_DIR}/src/binding
    ${PROJECT_SOURCE_DIR}/src
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/libpyre>
)

target_link_libraries(pyre_binding_common PUBLIC
    pyre
    iree_async_async
    iree_async_util_proactor_pool
    iree_base_base
    iree_base_internal_arena
    iree_base_threading_threading
    iree_hal_hal
    iree_hal_utils_resource_set
    iree_io_file_handle
)

set_target_properties(pyre_binding_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
