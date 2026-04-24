# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# Common binding runtime — shared by HIP and CUDA bindings.

add_library(hrx_binding_common STATIC
    context.c
    device.c
    event.c
    fat_binary.c
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

target_include_directories(hrx_binding_common PUBLIC
    ${PROJECT_SOURCE_DIR}/src/binding
    ${PROJECT_SOURCE_DIR}/src
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/libhrx>
)

target_link_libraries(hrx_binding_common PUBLIC
    hrx
    iree_async_async
    iree_async_util_proactor_pool
    iree_base_base
    iree_base_internal_arena
    iree_base_threading_threading
    iree_hal_hal
    iree_hal_utils_resource_set
    iree_io_file_handle
)

set_target_properties(hrx_binding_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# CCOB fat binaries use zstd for compression (used by rocBLAS/hipBLAS and
# most real-world HIP shipping toolchains). If the system has it, wire it in
# so the streaming layer can decompress on-the-fly.
if(HRX_ENABLE_ZSTD)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(ZSTD QUIET libzstd)
    endif()
    if(ZSTD_FOUND)
        target_compile_definitions(hrx_binding_common PRIVATE HRX_ENABLE_ZSTD=1)
        target_include_directories(hrx_binding_common PRIVATE ${ZSTD_INCLUDE_DIRS})
        target_link_libraries(hrx_binding_common PUBLIC ${ZSTD_LIBRARIES})
        message(STATUS "hrx-runtime: CCOB zstd decompression ENABLED (libzstd ${ZSTD_VERSION})")
    else()
        message(STATUS "hrx-runtime: libzstd not found; CCOB fat binaries will fail to load "
                       "(install libzstd-dev or disable HRX_ENABLE_ZSTD)")
    endif()
endif()
