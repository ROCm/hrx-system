// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/packet_rows.h"

#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/arch/spirv/isa.h"
#include "loom/target/arch/spirv/packet_rows_tables.inl"

const loom_spirv_packet_row_t* loom_spirv_packet_row_for_descriptor_ordinal(
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal >= IREE_ARRAYSIZE(kSpirvLogicalCorePacketRows)) {
    return NULL;
  }
  const loom_spirv_packet_row_t* row =
      &kSpirvLogicalCorePacketRows[descriptor_ordinal];
  return row->form == LOOM_SPIRV_PACKET_FORM_UNSUPPORTED ? NULL : row;
}
