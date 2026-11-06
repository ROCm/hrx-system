// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TESTING_IMAGE_FIXTURE_H_
#define IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TESTING_IMAGE_FIXTURE_H_

#include <memory>
#include <string>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/hal/driver/amd/xdna/image/image.h"
#include "iree/hal/driver/amd/xdna/image/testing/image_builder.h"

namespace iree::hal::amd::xdna::testing {

struct ByteSequenceDeleter {
  void operator()(iree_byte_sequence_t* sequence) const;
};
using ByteSequencePtr =
    std::unique_ptr<iree_byte_sequence_t, ByteSequenceDeleter>;

// Copies bytes into a ref-counted owned source sequence.
ByteSequencePtr MakeOwnedByteSequence(const std::vector<uint8_t>& bytes);

// Native-storage fixture for the loader boundary. Payloads are deliberately
// opaque and are never submitted to hardware; execution tests use compiler
// artifacts. Defaults include shared source bytes, an undefined command gap,
// a zero-filled DMA tail, and both allocation and external address fixups.
struct ImageFixture {
  ImageFixture();

  // Encodes the current rows and exact source ranges into an ELF image.
  std::vector<uint8_t> Build() const;
  // Complete metadata with counts derived from the vectors during Build.
  iree_xdna_elf_header_record_t header = {};
  // Backing requirements in stable allocation order.
  std::vector<iree_xdna_elf_allocation_record_t> allocations;
  // Entry-relative allocation references.
  std::vector<uint32_t> uses;
  // Exported entries and their table slices.
  std::vector<iree_xdna_elf_entry_record_t> entries;
  // External logical buffer contracts.
  std::vector<iree_xdna_elf_binding_record_t> bindings;
  // Static then dynamic fixups for each entry.
  std::vector<iree_xdna_elf_relocation_record_t> relocations;
  // Establishing and continuation command ranges.
  std::vector<iree_xdna_elf_invocation_record_t> invocations;
  // Raw exported names.
  std::string names = "main";
  // Explicit loads after the metadata header, with fixed source offsets.
  std::vector<iree_hal_amd_xdna_image_program_header_t> loads;
  // Source bytes for each corresponding load; exact aliases agree.
  std::vector<std::vector<uint8_t>> payloads;
};

// Returns the matching one-column Halo execution contract.
iree_hal_amd_xdna_aie2p_target_t MakeImageTarget();

}  // namespace iree::hal::amd::xdna::testing

#endif  // IREE_HAL_DRIVER_AMD_XDNA_IMAGE_TESTING_IMAGE_FIXTURE_H_
