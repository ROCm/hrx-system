# Copyright 2026 The Pyre Authors
# SPDX-License-Identifier: Apache-2.0
#
# libpyre.so — LTO'd shared library, only pyre_* symbols exported.

set(PYRE_SOURCES
    status.c
    runtime.c
    device.c
    semaphore.c
    stream.c
    buffer.c
    queue_ops.c
)

set(PYRE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_library(pyre SHARED ${PYRE_SOURCES})
target_include_directories(pyre
    PUBLIC
        $<BUILD_INTERFACE:${PYRE_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include/pyre>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(pyre PRIVATE
    hsa-runtime64::hsa-runtime64
    pyre_hsa_registration
    iree_async_util_proactor_pool
    iree_hal_drivers_local_task_task_driver
    iree_hal_local_loaders_registration_registration
    iree_vm_vm
)
target_compile_definitions(pyre PRIVATE PYRE_BUILDING_SHARED PYRE_HAS_HSA_DRIVER)
set_target_properties(pyre PROPERTIES
    OUTPUT_NAME pyre
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE ON
)

# Version script to hide all non-pyre_* symbols.
set_target_properties(pyre PROPERTIES
    LINK_FLAGS "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/pyre_exports.lds")

# LTO (interprocedural optimization).
include(CheckIPOSupported)
check_ipo_supported(RESULT _pyre_ipo_supported OUTPUT _pyre_ipo_error)
if(_pyre_ipo_supported)
  set_target_properties(pyre PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  message(STATUS "pyre-runtime: LTO enabled for libpyre.so")
else()
  message(WARNING "pyre-runtime: LTO not supported: ${_pyre_ipo_error}")
endif()
