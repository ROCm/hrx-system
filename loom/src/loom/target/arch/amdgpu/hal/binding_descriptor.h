// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared contract for the AMDGPU HAL buffer-descriptor pseudo.

#ifndef LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_DESCRIPTOR_H_
#define LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_DESCRIPTOR_H_

// Attribute order for low.op<amdgpu.hal.buffer_descriptor>. The descriptor
// verifier canonicalizes fields into this order; verifier and materializer code
// use ordinals so target lowering never re-parses attribute names.
typedef enum loom_amdgpu_hal_buffer_descriptor_attr_ordinal_e {
  // Byte stride encoded into the descriptor pointer high word.
  LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_CACHE_SWIZZLE_STRIDE = 0,
  // Maximum byte extent exposed through the descriptor.
  LOOM_AMDGPU_HAL_BUFFER_DESCRIPTOR_ATTR_EXTENT = 1,
} loom_amdgpu_hal_buffer_descriptor_attr_ordinal_t;

#endif  // LOOM_TARGET_ARCH_AMDGPU_HAL_BINDING_DESCRIPTOR_H_
