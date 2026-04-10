# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0
#
# libhrx.so — LTO'd shared library, only hrx_* symbols exported.

set(HRX_SOURCES
    status.c
    runtime.c
    device.c
    allocator.c
    semaphore.c
    stream.c
    buffer.c
    buffer_view.c
    executable.c
    fence.c
    module.c
    transfer.c
    host_allocator.c
    queue_ops.c
    value_list.c
    compiler.cpp
    ${HRX_IREE_SOURCE_DIR}/compiler/bindings/c/iree/compiler/loader/loader.cpp
)

set(HRX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_library(hrx SHARED ${HRX_SOURCES})
target_include_directories(hrx
    PUBLIC
        $<BUILD_INTERFACE:${HRX_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/hrx>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
        ${HRX_IREE_SOURCE_DIR}/compiler/bindings/c
)
target_link_libraries(hrx PRIVATE
    ${CMAKE_DL_LIBS}
    hsa-runtime64::hsa-runtime64
    hrx_hsa_registration
    iree_async_util_proactor_pool
    iree_hal_drivers_local_task_task_driver
    iree_hal_local_loaders_registration_registration
    iree_modules_hal_hal
    iree_vm_vm
)
target_compile_definitions(hrx PRIVATE HRX_BUILDING_SHARED HRX_HAS_HSA_DRIVER)
set_target_properties(hrx PROPERTIES
    OUTPUT_NAME hrx
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# Version script to hide all non-hrx_* symbols.
set_target_properties(hrx PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/hrx_exports.lds")

# LTO (interprocedural optimization).
include(CheckIPOSupported)
check_ipo_supported(RESULT _hrx_ipo_supported OUTPUT _hrx_ipo_error)
if(_hrx_ipo_supported)
  set_target_properties(hrx PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "hrx-runtime: LTO enabled for libhrx.so")
else()
  message(WARNING "hrx-runtime: LTO not supported: ${_hrx_ipo_error}")
endif()
