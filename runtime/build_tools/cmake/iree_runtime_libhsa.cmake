# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Enable statically linking against ROCR-Runtime.
# The resulting application binary will initialize the runtime at startup and
# may fail to work on any machine without a modern ROCM driver. Avoid using this
# unless it is known that the deployment is always on an up-to-date AMDGPU
# device.
option(IREE_HAL_AMDGPU_LIBHSA_STATIC
       "Statically link against the installed hsa-runtime64 package."
       OFF)

if(IREE_HAL_AMDGPU_LIBHSA_STATIC)
  if(NOT TARGET iree::third_party::hsa_runtime)
    message(FATAL_ERROR
      "IREE_HAL_AMDGPU_LIBHSA_STATIC=ON requires a ROCm/TheRock package "
      "configuration that provides hsa-runtime64::hsa-runtime64. Header-only "
      "ROCm discovery is enough for dynamic loading but cannot satisfy static "
      "libhsa linking.")
  endif()
  target_compile_options(iree_hal_driver_amdgpu_util_libhsa
    PUBLIC
      "-DIREE_HAL_AMDGPU_LIBHSA_STATIC=1"
  )
  target_link_libraries(iree_hal_driver_amdgpu_util_libhsa
    PRIVATE
      iree::third_party::hsa_runtime
  )
endif()
