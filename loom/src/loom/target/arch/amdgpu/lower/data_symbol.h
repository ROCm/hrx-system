// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low helpers for addressing AMDGPU code-object data symbols.
//
// AMDGPU kernels materialize code-object data symbol addresses with a
// PC-relative sequence. Keeping that sequence behind one helper gives feedback
// channels, sanitizer site tables, ASAN shadow configuration, and future
// metadata tables one shared lowering primitive instead of feature-local
// copies.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_DATA_SYMBOL_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_DATA_SYMBOL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/attribute.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

typedef struct loom_amdgpu_data_symbol_address_t {
  // Module-local code-object data symbol to address.
  loom_symbol_ref_t symbol;
  // Byte offset within the referenced data symbol.
  uint64_t byte_offset;
} loom_amdgpu_data_symbol_address_t;

// Emits target-low IR that materializes |target| as a 64-bit scalar address.
//
// The result is a reg<amdgpu.sgpr x2> value formed from the canonical
// s_getpc_b64 + rel32 symbolic add sequence. The selected descriptor set must
// expose the AMDGPU symbolic-relocation descriptors used by native emission.
iree_status_t loom_amdgpu_build_data_symbol_address(
    loom_builder_t* builder, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_data_symbol_address_t target, loom_location_id_t location,
    loom_value_id_t* out_low_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_DATA_SYMBOL_H_
