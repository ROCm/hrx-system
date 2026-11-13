// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/image/testing/image_fixture.h"

#include <cstring>

#include "iree/hal/driver/amd/xdna/image/aie2p/npu2.h"

namespace iree::hal::amd::xdna::testing {

void ByteSequenceDeleter::operator()(iree_byte_sequence_t* sequence) const {
  iree_byte_sequence_release(sequence);
}

ByteSequencePtr MakeOwnedByteSequence(const std::vector<uint8_t>& bytes) {
  iree_byte_span_t storage = {nullptr, bytes.size()};
  IREE_CHECK_OK(iree_allocator_clone(
      iree_allocator_system(),
      iree_make_const_byte_span(bytes.data(), bytes.size()),
      reinterpret_cast<void**>(&storage.data)));
  iree_byte_sequence_t* sequence = nullptr;
  IREE_CHECK_OK(iree_byte_sequence_create_from_span_move(
      &storage, iree_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

iree_hal_amd_xdna_aie2p_target_t MakeImageTarget() {
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &target));
  return target;
}

ImageFixture::ImageFixture() {
  const auto target = MakeImageTarget();
  header.magic = IREE_XDNA_ELF_METADATA_MAGIC;
  header.version = IREE_XDNA_ELF_METADATA_VERSION;
  header.native_encoding = IREE_XDNA_ELF_NATIVE_TRANSACTION_0_1;
  header.target_generation = IREE_XDNA_TARGET_GENERATION_AIE2P;
  header.device_profile_revision = target.identity.device_profile_revision;
  header.device_profile_id = target.identity.device_profile_id;
  header.firmware_abi_id = target.identity.firmware_abi_id;
  header.column_count = 1;
  header.row_count = 6;
  allocations.resize(2);
  allocations[0].domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_COMMAND;
  allocations[0].byte_length = 32784;
  allocations[0].alignment = 32768;
  allocations[0].first_load = 1;
  allocations[0].load_count = 2;
  allocations[1].domain = IREE_XDNA_ELF_ALLOCATION_DOMAIN_DMA;
  allocations[1].flags = IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE;
  allocations[1].byte_length = 16;
  allocations[1].alignment = 4;
  allocations[1].first_load = 3;
  allocations[1].load_count = 1;
  uses = {0, 1};
  entries.resize(1);
  entries[0].name_length = 4;
  entries[0].allocation_use_count = 2;
  entries[0].binding_count = 1;
  entries[0].static_relocation_count = 1;
  entries[0].first_dynamic_relocation = 1;
  entries[0].dynamic_relocation_count = 1;
  entries[0].invocation_count = 2;
  bindings.resize(1);
  bindings[0].kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER;
  bindings[0].address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL;
  bindings[0].access = IREE_XDNA_ELF_BINDING_ACCESS_READ;
  bindings[0].usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE;
  bindings[0].minimum_byte_length = 16;
  bindings[0].minimum_alignment = 4;
  bindings[0].maximum_byte_offset = UINT64_MAX;
  relocations.resize(2);
  relocations[0].source_ordinal = 1;
  relocations[0].kind = IREE_XDNA_ELF_RELOCATION_KIND_SHIM_ADDRESS;
  relocations[0].addend = 4;
  relocations[0].maximum_value = UINT64_C(0xFFFFFFFFFFFF);
  relocations[0].alignment = 4;
  relocations[1] = relocations[0];
  relocations[1].byte_offset = 8;
  relocations[1].source_ordinal = 0;
  invocations.resize(2);
  invocations[0].byte_length = 16;
  invocations[0].next_invocation = 1;
  invocations[1] = invocations[0];
  invocations[1].byte_offset = 32768;
  loads.push_back(MakeProgramHeader(IREE_XDNA_ELF_PROGRAM_TYPE_LOAD, 4096, 16));
  loads.push_back(loads[0]);
  loads[1].virtual_address = 32768;
  loads.push_back(MakeProgramHeader(IREE_XDNA_ELF_PROGRAM_TYPE_LOAD, 4112, 8));
  loads[2].physical_address = 1;
  loads[2].memory_size = 16;
  payloads = {std::vector<uint8_t>(16, 0xA5),
              std::vector<uint8_t>(16, 0xA5),
              {1, 2, 3, 4, 5, 6, 7, 8}};
}

std::vector<uint8_t> ImageFixture::Build() const {
  auto encoded_header = header;
  encoded_header.allocation_count = allocations.size();
  encoded_header.allocation_use_count = uses.size();
  encoded_header.entry_count = entries.size();
  encoded_header.binding_count = bindings.size();
  encoded_header.relocation_count = relocations.size();
  encoded_header.invocation_count = invocations.size();
  encoded_header.string_byte_length = names.size();
  std::vector<uint8_t> metadata(
      IREE_XDNA_ELF_HEADER_RECORD_SIZE +
      allocations.size() * IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE +
      uses.size() * 4 + entries.size() * IREE_XDNA_ELF_ENTRY_RECORD_SIZE +
      bindings.size() * IREE_XDNA_ELF_BINDING_RECORD_SIZE +
      relocations.size() * IREE_XDNA_ELF_RELOCATION_RECORD_SIZE +
      invocations.size() * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE + names.size());
  uint8_t* cursor = metadata.data();
  iree_xdna_elf_encode_header(&encoded_header, cursor);
  cursor += IREE_XDNA_ELF_HEADER_RECORD_SIZE;
  for (const auto& row : allocations) {
    iree_xdna_elf_encode_allocation(&row, cursor);
    cursor += IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE;
  }
  for (uint32_t use : uses) {
    iree_unaligned_store_le_u32(cursor, use);
    cursor += 4;
  }
  for (const auto& row : entries) {
    iree_xdna_elf_encode_entry(&row, cursor);
    cursor += IREE_XDNA_ELF_ENTRY_RECORD_SIZE;
  }
  for (const auto& row : bindings) {
    iree_xdna_elf_encode_binding(&row, cursor);
    cursor += IREE_XDNA_ELF_BINDING_RECORD_SIZE;
  }
  for (const auto& row : relocations) {
    iree_xdna_elf_encode_relocation(&row, cursor);
    cursor += IREE_XDNA_ELF_RELOCATION_RECORD_SIZE;
  }
  for (const auto& row : invocations) {
    iree_xdna_elf_encode_invocation(&row, cursor);
    cursor += IREE_XDNA_ELF_INVOCATION_RECORD_SIZE;
  }
  std::memcpy(cursor, names.data(), names.size());
  ImageBuilder builder;
  const auto metadata_header = MakeProgramHeader(
      IREE_XDNA_ELF_PROGRAM_TYPE_METADATA, 512, metadata.size());
  builder.AddProgram(metadata_header, std::move(metadata));
  for (size_t i = 0; i < loads.size(); ++i) {
    builder.AddProgram(loads[i], payloads[i]);
  }
  return builder.Build();
}

}  // namespace iree::hal::amd::xdna::testing
