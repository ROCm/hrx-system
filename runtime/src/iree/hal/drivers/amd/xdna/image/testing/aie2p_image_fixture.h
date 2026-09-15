// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_AIE2P_IMAGE_FIXTURE_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_AIE2P_IMAGE_FIXTURE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"

namespace iree::hal::amd::xdna::testing {

enum : uint32_t {
  kAie2pImageTileProgramHeaderOrdinal = 4,
  kAie2pImageArrayProgramHeaderOrdinal = 5,
  kAie2pImageControlProgramHeaderOrdinal = 6,
};

inline constexpr iree_hal_amd_xdna_elf_capabilities_t
    kAie2pImageStructuralCapabilities =
        IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS |
        IREE_HAL_AMD_XDNA_ELF_CAPABILITY_CONTROL_PROGRAMS;

// Options controlling semantic mutations of the valid synthetic image.
struct Aie2pImageFixtureOptions {
  // Capabilities declared by the ABI note and executable entry.
  iree_hal_amd_xdna_elf_capabilities_t required_capabilities =
      kAie2pImageStructuralCapabilities;
  // ARRAY program-header ordinal referenced by the executable entry.
  uint32_t entry_array_program_header_ordinal =
      kAie2pImageArrayProgramHeaderOrdinal;
  // CONTROL program-header ordinal referenced by the executable entry.
  uint32_t entry_control_program_header_ordinal =
      kAie2pImageControlProgramHeaderOrdinal;
  // TILE program-header ordinal referenced by the ARRAY record.
  uint32_t array_tile_program_header_ordinal =
      kAie2pImageTileProgramHeaderOrdinal;
  // Relocation byte offset within the CONTROL payload.
  uint32_t relocation_target_byte_offset = 40;
  // Serialized relocation field width in bytes.
  uint8_t relocation_field_byte_width = 8;
  // Signed displacement applied to the binding address.
  int64_t relocation_addend = 0;
  // Minimum legal relocated value.
  uint64_t relocation_minimum_value = 0;
  // Maximum legal relocated value.
  uint64_t relocation_maximum_value = UINT64_MAX;
  // Required relocated-value alignment.
  uint64_t relocation_required_alignment = 4;
  // Address of the first register in the relocatable CONTROL block write.
  uint32_t control_register_address = 0x1D004;
  // Whether the CONTROL record is a relocatable block write.
  bool use_relocatable_block_write = true;
};

// Mutable observations and fault injection for the synthetic target.
struct Aie2pImageTargetState {
  // Optional owner column override used by malformed-placement tests.
  uint16_t owner_column = UINT16_MAX;
  // Number of configuration-register records validated.
  mutable iree_host_size_t configuration_register_call_count = 0;
};

// Releases an owned byte sequence.
struct ByteSequenceDeleter {
  void operator()(iree_byte_sequence_t* sequence) const;
};

using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, ByteSequenceDeleter>;

// Copies |bytes| into a ref-counted byte sequence owned by the caller.
ByteSequencePtr MakeOwnedByteSequence(const std::vector<uint8_t>& bytes);

// Builds a complete valid AIE2P image with optional semantic mutations.
std::vector<uint8_t> BuildAie2pImage(
    const Aie2pImageFixtureOptions& options = {});

// Returns complete AIE2P target facts borrowing |state|.
iree_hal_amd_xdna_aie2p_target_t MakeAie2pImageTarget(
    Aie2pImageTargetState* state);

}  // namespace iree::hal::amd::xdna::testing

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_TESTING_AIE2P_IMAGE_FIXTURE_H_
