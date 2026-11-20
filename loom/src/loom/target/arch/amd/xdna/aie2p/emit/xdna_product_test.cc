// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/xdna_product.h"

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "iree/hal/driver/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/driver/amd/xdna/image/image.h"
#include "iree/io/vec_stream.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class XdnaProductTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
            IREE_IO_STREAM_MODE_SEEKABLE,
        4096, iree_allocator_system(), &stream_));
  }

  void TearDown() override {
    iree_hal_amd_xdna_image_free(image_);
    iree_io_stream_release(stream_);
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void ExpectAdmissionIssue(const loom_aie2p_xdna_product_t* product,
                            loom_aie2p_xdna_product_issue_kind_t expected_kind,
                            uint32_t expected_entry_ordinal,
                            uint64_t expected_actual, uint64_t expected_minimum,
                            uint64_t expected_maximum) {
    loom_aie2p_xdna_product_issue_t issue = {};
    loom_aie2p_xdna_product_admission_t* admission = nullptr;
    bool admitted = true;
    IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(product, &arena_, &admitted,
                                                 &admission, &issue));
    EXPECT_FALSE(admitted);
    EXPECT_EQ(admission, nullptr);
    EXPECT_EQ(issue.kind, expected_kind);
    EXPECT_EQ(issue.entry_ordinal, expected_entry_ordinal);
    EXPECT_EQ(issue.actual, expected_actual);
    EXPECT_EQ(issue.minimum, expected_minimum);
    EXPECT_EQ(issue.maximum, expected_maximum);
  }

  void InitializeSourceKnownProduct(loom_aie2p_array_program_t* out_program,
                                    loom_aie2p_xdna_entry_t* out_entry,
                                    loom_aie2p_xdna_product_t* out_product) {
    const loom_xdna_device_profile_t* profile =
        loom_xdna_device_profile_lookup(IREE_SV("amd.xdna.strix_halo.17f0_11"));
    IREE_ASSERT(profile != nullptr);
    *out_program = (loom_aie2p_array_program_t){0};
    *out_entry = loom_aie2p_xdna_entry_t{
        .name = IREE_SV("entry"),
        .column_count = 1,
        .bindings = nullptr,
        .binding_count = 0,
        .array_program = out_program,
        .tiles = nullptr,
        .tile_count = 0,
    };
    *out_product = loom_aie2p_xdna_product_t{
        .device_profile = profile,
        .entries = out_entry,
        .entry_count = 1,
    };
  }

  void LoadOnNpu2(const loom_aie2p_xdna_product_t& product, uint16_t columns) {
    loom_aie2p_xdna_product_issue_t issue = {};
    loom_aie2p_xdna_product_admission_t* admission = nullptr;
    bool admitted = false;
    IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(&product, &arena_, &admitted,
                                                 &admission, &issue));
    ASSERT_TRUE(admitted);
    loom_aie2p_xdna_product_image_t* native = nullptr;
    bool finalized = false;
    IREE_ASSERT_OK(loom_aie2p_xdna_product_finalize(admission, &finalized,
                                                    &native, &issue));
    ASSERT_TRUE(finalized) << issue.kind;
    IREE_ASSERT_OK(loom_aie2p_xdna_product_write(native, stream_));
    iree_arena_reset(&arena_);
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        IREE_SV("amd.xdna.strix_halo.17f0_11"), columns, &target));
    iree_byte_sequence_t* sequence = nullptr;
    IREE_ASSERT_OK(iree_io_vec_stream_move_contents(stream_, &sequence));
    iree_status_t status = iree_hal_amd_xdna_image_allocate(
        sequence, &target, iree_allocator_system(), &image_);
    iree_byte_sequence_release(sequence);
    IREE_ASSERT_OK(status);
  }

  // Image admitted by the runtime, independent of the native emitter's arena.
  iree_hal_amd_xdna_image_t* image_ = nullptr;
  // Backing blocks for temporary native emission data.
  iree_arena_block_pool_t pool_;
  // Scratch storage used only while constructing the product.
  iree_arena_allocator_t arena_;
  // Growable native ELF output.
  iree_io_stream_t* stream_ = nullptr;
};

TEST_F(XdnaProductTest, RejectsSourceKnownEntryCount) {
  loom_aie2p_array_program_t program;
  loom_aie2p_xdna_entry_t entry;
  loom_aie2p_xdna_product_t product;
  InitializeSourceKnownProduct(&program, &entry, &product);
  constexpr iree_host_size_t kMaximumEntryCount =
      IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT / 2u;
  std::vector<loom_aie2p_xdna_entry_t> entries(kMaximumEntryCount + 1u, entry);
  product.entries = entries.data();
  product.entry_count = entries.size();
  ExpectAdmissionIssue(&product, LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_COUNT,
                       UINT32_MAX, entries.size(), 0, kMaximumEntryCount);
}

TEST_F(XdnaProductTest, RejectsSourceKnownEntryNameLength) {
  loom_aie2p_array_program_t program;
  loom_aie2p_xdna_entry_t entry;
  loom_aie2p_xdna_product_t product;
  InitializeSourceKnownProduct(&program, &entry, &product);
  const std::string name(IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH + 1u, 'x');
  entry.name = iree_make_string_view(name.data(), name.size());
  ExpectAdmissionIssue(&product,
                       LOOM_AIE2P_XDNA_PRODUCT_ISSUE_ENTRY_NAME_BYTE_LENGTH, 0,
                       name.size(), 0, IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH);
}

TEST_F(XdnaProductTest, RejectsSourceKnownRelocationCount) {
  loom_aie2p_array_program_t program;
  loom_aie2p_xdna_entry_t entry;
  loom_aie2p_xdna_product_t product;
  InitializeSourceKnownProduct(&program, &entry, &product);
  constexpr iree_host_size_t kSourceRelocationCount =
      IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT / 2u + 1u;
  std::vector<loom_aie2p_program_relocation_t> relocations(
      kSourceRelocationCount);
  program.relocations = relocations.data();
  program.relocation_count = relocations.size();
  ExpectAdmissionIssue(
      &product, LOOM_AIE2P_XDNA_PRODUCT_ISSUE_RELOCATION_RECORD_COUNT, 0,
      2u * kSourceRelocationCount, 0, IREE_XDNA_ELF_MAX_TABLE_RECORD_COUNT);
}

TEST_F(XdnaProductTest, RejectsSourceKnownMetadataSize) {
  loom_aie2p_array_program_t program;
  loom_aie2p_xdna_entry_t entry;
  loom_aie2p_xdna_product_t product;
  InitializeSourceKnownProduct(&program, &entry, &product);
  const std::string name(IREE_XDNA_ELF_MAX_ENTRY_NAME_LENGTH, 'x');
  entry.name = iree_make_string_view(name.data(), name.size());
  constexpr iree_host_size_t kEntryCount = 4096;
  std::vector<loom_aie2p_xdna_entry_t> entries(kEntryCount, entry);
  product.entries = entries.data();
  product.entry_count = entries.size();
  const uint64_t expected_metadata_size =
      IREE_XDNA_ELF_HEADER_RECORD_SIZE +
      kEntryCount * (IREE_XDNA_ELF_ALLOCATION_RECORD_SIZE + sizeof(uint32_t) +
                     IREE_XDNA_ELF_ENTRY_RECORD_SIZE +
                     2u * IREE_XDNA_ELF_INVOCATION_RECORD_SIZE + name.size());
  ASSERT_GT(expected_metadata_size, IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE);
  ExpectAdmissionIssue(
      &product, LOOM_AIE2P_XDNA_PRODUCT_ISSUE_METADATA_BYTE_LENGTH, UINT32_MAX,
      expected_metadata_size, 0, IREE_XDNA_ELF_MAX_METADATA_TABLE_SIZE);
}

TEST_F(XdnaProductTest, PreservesRequirementsIndependentOfWorkerTopology) {
  // Service-only entries initialize a lock without loading a compute program.
  // Their partition width and external contracts remain required even though
  // no worker or payload-channel graph exists at this boundary.
  loom_aie2p_program_record_t commands[2] = {};
  commands[0].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32;
  commands[0].value.register_write32 = {0x0221F000, 1};
  commands[1].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32;
  commands[1].value.register_write32 = {0x0E21F000, 1};
  const loom_aie2p_array_program_t programs[] = {
      {nullptr, 0, &commands[0], 1, nullptr, 0, {true, 0, 1}},
      {nullptr, 0, &commands[1], 1, nullptr, 0, {}},
  };
  const iree_xdna_elf_binding_record_t bindings[] = {
      {
          .kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER,
          .address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
          .access = IREE_XDNA_ELF_BINDING_ACCESS_READ,
          .usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
          .minimum_byte_length = 4096 + 256,
          .minimum_alignment = 256,
          .minimum_byte_offset = 256,
          .maximum_byte_offset = 1024,
      },
      {
          .kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER,
          .address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_HOST,
          .access = IREE_XDNA_ELF_BINDING_ACCESS_READ |
                    IREE_XDNA_ELF_BINDING_ACCESS_WRITE,
          .usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE |
                   IREE_XDNA_ELF_BINDING_USAGE_COHERENT,
          .minimum_byte_length = 65536 + 64,
          .minimum_alignment = 64,
          .minimum_byte_offset = 0,
          .maximum_byte_offset = UINT64_MAX,
      },
  };
  const loom_aie2p_xdna_entry_t entries[] = {
      {IREE_SV("ingress"), 2, &bindings[0], 1, &programs[0], nullptr, 0},
      {IREE_SV("egress"), 8, &bindings[1], 1, &programs[1], nullptr, 0},
  };
  const auto profile_key = IREE_SV("amd.xdna.strix_halo.17f0_11");
  const loom_aie2p_xdna_product_t product = {
      loom_xdna_device_profile_lookup(profile_key), entries,
      IREE_ARRAYSIZE(entries)};
  ASSERT_NO_FATAL_FAILURE(LoadOnNpu2(product, 8));
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  const auto& header = tables->header;
  EXPECT_EQ(header.column_count, 8u);
  ASSERT_EQ(header.allocation_count, 2u);
  ASSERT_EQ(header.allocation_use_count, 2u);
  ASSERT_EQ(header.entry_count, 2u);
  ASSERT_EQ(header.binding_count, 2u);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(entries); ++i) {
    const auto& entry = tables->entries[i];
    ASSERT_EQ(entry.first_binding, i);
    EXPECT_EQ(entry.binding_count, 1u);
    const auto& binding = tables->bindings[entry.first_binding];
    EXPECT_EQ(binding.kind, bindings[i].kind);
    EXPECT_EQ(binding.address_space, bindings[i].address_space);
    EXPECT_EQ(binding.access, bindings[i].access);
    EXPECT_EQ(binding.usage, bindings[i].usage);
    EXPECT_EQ(binding.minimum_byte_length, bindings[i].minimum_byte_length);
    EXPECT_EQ(binding.minimum_alignment, bindings[i].minimum_alignment);
    EXPECT_EQ(binding.minimum_byte_offset, bindings[i].minimum_byte_offset);
    EXPECT_EQ(binding.maximum_byte_offset, bindings[i].maximum_byte_offset);
    for (uint32_t j = 0; j < entry.invocation_count; ++j) {
      const auto& invocation = tables->invocations[entry.first_invocation + j];
      // Both forms preserve ordinary command lengths; only the first entry
      // declares a measured interval, including its final record boundary.
      EXPECT_EQ(invocation.byte_length, 40u);
      EXPECT_EQ(invocation.measurement.begin, i == 0 ? 16u : 0u);
      EXPECT_EQ(invocation.measurement.end, i == 0 ? 40u : 0u);
    }
  }
}

class SyntheticXdnaProduct {
 public:
  SyntheticXdnaProduct(iree_host_size_t entry_count,
                       iree_host_size_t tile_count) {
    profile_ =
        loom_xdna_device_profile_lookup(IREE_SV("amd.xdna.strix_halo.17f0_11"));
    IREE_ASSERT(profile_ != nullptr);
    family_ = loom_xdna_device_profile_array_family(profile_);
    IREE_ASSERT(family_ != nullptr);
    const loom_xdna_tile_facts_t* compute = nullptr;
    for (uint8_t i = 0; i < family_->tile_count; ++i) {
      if (family_->tiles[i].kind == LOOM_XDNA_TILE_KIND_COMPUTE) {
        compute = &family_->tiles[i];
      }
    }
    IREE_ASSERT(compute != nullptr);
    IREE_ASSERT_LE(tile_count, (iree_host_size_t)family_->column_count *
                                   compute->row_count);

    sections_[0] = loom_native_section_t{
        .name = IREE_SV(".text.kernel"),
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                  LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
        .address = 0,
        .alignment = 16,
        .contents = iree_make_const_byte_span(code_.data(), code_.size()),
    };
    sections_[1] = loom_native_section_t{
        .name = IREE_SV(".storage.first"),
        .storage = LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
        .access =
            LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_WRITE,
        .address = 0x70000,
        .alignment = 64,
        .contents = iree_const_byte_span_empty(),
        .reservation_length = 64,
    };
    sections_[2] = loom_native_section_t{
        .name = IREE_SV(".storage.second"),
        .storage = LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
        .access =
            LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_WRITE,
        .address = 0x70100,
        .alignment = 64,
        .contents = iree_const_byte_span_empty(),
        .reservation_length = 64,
    };
    placements_[0].memory_space = LOOM_XDNA_MEMORY_SPACE_PROGRAM;
    placements_[1].memory_space = LOOM_XDNA_MEMORY_SPACE_DATA;
    placements_[2].memory_space = LOOM_XDNA_MEMORY_SPACE_DATA;
    linked_tile_ = loom_aie2p_linked_tile_t{
        .assembly =
            {
                .sections = sections_.data(),
                .section_count = sections_.size(),
            },
        .section_placements = placements_.data(),
        .section_placement_count = placements_.size(),
        .symbol_layouts = nullptr,
        .symbol_layout_count = 0,
        .entry_section_index = 0,
        .entry_address = 0,
    };
    entry_symbol_ = loom_native_object_symbol_t{
        .name = IREE_SV("kernel"),
        .section_contribution_index = 0,
        .section_offset = 0,
        .size = code_.size(),
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
        .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
    };
    program_object_.contribution.object.symbols = &entry_symbol_;
    program_object_.contribution.object.symbol_count = 1;
    program_object_.contribution.realization.entry_symbol_index = 0;

    records_.resize(tile_count);
    for (iree_host_size_t i = 0; i < tile_count; ++i) {
      records_[i].type = LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD;
      records_[i].value.tile_program_load.tile_program_index = (uint32_t)i;
    }
    array_program_.array_records = records_.data();
    array_program_.array_record_count = records_.size();

    tiles_.resize(entry_count * tile_count);
    names_.reserve(entry_count);
    entries_.resize(entry_count);
    for (iree_host_size_t i = 0; i < entry_count; ++i) {
      names_.emplace_back("entry_" + std::to_string(i));
      for (iree_host_size_t j = 0; j < tile_count; ++j) {
        tiles_[i * tile_count + j] = loom_aie2p_xdna_tile_t{
            .coordinate = {(uint16_t)(j / compute->row_count),
                           (uint16_t)(compute->first_row +
                                      j % compute->row_count)},
            .program_name = entry_symbol_.name,
            .program = &program_object_,
            .linked_tile = &linked_tile_,
        };
      }
      entries_[i] = loom_aie2p_xdna_entry_t{
          .name = iree_make_cstring_view(names_[i].c_str()),
          .column_count = (uint16_t)((tile_count + compute->row_count - 1u) /
                                     compute->row_count),
          .bindings = nullptr,
          .binding_count = 0,
          .array_program = &array_program_,
          .tiles = tiles_.data() + i * tile_count,
          .tile_count = tile_count,
      };
    }
    product_ = loom_aie2p_xdna_product_t{
        .device_profile = profile_,
        .entries = entries_.data(),
        .entry_count = entries_.size(),
    };
  }

  const loom_aie2p_xdna_product_t* product() const { return &product_; }

 private:
  const loom_xdna_device_profile_t* profile_ = nullptr;
  const loom_xdna_array_family_t* family_ = nullptr;
  std::array<uint8_t, 4> code_ = {0x44, 0x20, 0xc1, 0x20};
  std::array<loom_native_section_t, 3> sections_ = {};
  std::array<loom_aie2p_linked_section_placement_t, 3> placements_ = {};
  loom_aie2p_linked_tile_t linked_tile_ = {};
  loom_native_object_symbol_t entry_symbol_ = {};
  loom_aie2p_program_object_t program_object_ = {};
  std::vector<loom_aie2p_program_record_t> records_;
  loom_aie2p_array_program_t array_program_ = {};
  std::vector<loom_aie2p_xdna_tile_t> tiles_;
  std::vector<std::string> names_;
  std::vector<loom_aie2p_xdna_entry_t> entries_;
  loom_aie2p_xdna_product_t product_ = {};
};

TEST_F(XdnaProductTest, StaticLoaderAddressesPrecedeBothBindingRelocations) {
  SyntheticXdnaProduct fixture(4, 1);
  auto product = *fixture.product();
  std::vector<loom_aie2p_xdna_entry_t> entries(product.entries,
                                               product.entries + 4);
  product.entries = entries.data();
  auto regular = *entries[0].array_program;
  const uint32_t descriptor[8] = {1, 0, 0, 0, 0, 0, 0, 1};
  const loom_aie2p_program_record_t shim = {
      .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
      .value = {.register_block_write32 = {0x1d000u, descriptor, 8}},
  };
  const loom_aie2p_program_relocation_t call_address = {
      .target_word_index = 1,
      .maximum_value = (UINT64_C(1) << 48) - 4,
      .required_alignment = 4,
  };
  const iree_xdna_elf_binding_record_t binding = {
      .kind = IREE_XDNA_ELF_BINDING_KIND_BUFFER,
      .address_space = IREE_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL,
      .access = IREE_XDNA_ELF_BINDING_ACCESS_READ,
      .usage = IREE_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE,
      .minimum_byte_length = 4,
      .minimum_alignment = 4,
      .maximum_byte_offset = UINT64_MAX,
  };
  regular.control_records = &shim;
  regular.control_record_count = 1;
  regular.relocations = &call_address;
  regular.relocation_count = 1;
  auto bulk = regular;
  const loom_aie2p_program_record_t initialize[] = {
      regular.array_records[0],
      shim,
  };
  const std::array<uint8_t, 8> data = {1, 2, 3, 4, 5, 6, 7, 8};
  const loom_aie2p_program_upload_patch_t patch = {1, 4, 4};
  const loom_aie2p_program_upload_t upload = {
      .records = initialize,
      .record_count = IREE_ARRAYSIZE(initialize),
      .contents = iree_make_const_byte_span(data.data(), data.size()),
      .patches = &patch,
      .patch_count = 1,
  };
  bulk.upload = &upload;
  for (size_t i = 0; i < entries.size(); ++i) {
    entries[i].array_program = i % 2 ? &bulk : &regular;
    entries[i].bindings = &binding;
    entries[i].binding_count = 1;
  }
  ASSERT_NO_FATAL_FAILURE(LoadOnNpu2(product, 1));
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  ASSERT_EQ(tables->header.allocation_count, 6u);
  ASSERT_EQ(tables->header.relocation_count, 10u);
  uint32_t first_use = 0;
  for (uint32_t i = 0; i < entries.size(); ++i) {
    const auto& entry = tables->entries[i];
    const bool uploads = (i % 2) != 0;
    ASSERT_EQ(entry.first_allocation_use, first_use);
    ASSERT_EQ(entry.allocation_use_count, uploads ? 2u : 1u);
    EXPECT_EQ(entry.static_relocation_count, uploads ? 1u : 0u);
    ASSERT_EQ(entry.dynamic_relocation_count, 2u);
    const auto& cold = tables->invocations[entry.first_invocation];
    const auto& warm = tables->invocations[entry.first_invocation + 1];
    // Header, block-write header, and four bytes of spliced tile code.
    constexpr uint32_t kTileEnd = 16 + 16 + 4;
    constexpr uint32_t kDescriptorSize = 16 + sizeof(descriptor);
    constexpr uint32_t kAddress = 16 + sizeof(uint32_t);
    EXPECT_EQ(cold.byte_length, kTileEnd + kDescriptorSize * (uploads ? 2 : 1));
    EXPECT_EQ(warm.byte_length, 16u + kDescriptorSize);
    EXPECT_EQ(tables->relocations[entry.first_dynamic_relocation].byte_offset,
              kTileEnd + (uploads ? kDescriptorSize : 0) + kAddress);
    EXPECT_EQ(
        tables->relocations[entry.first_dynamic_relocation + 1].byte_offset,
        warm.byte_offset + 16 + kAddress);
    if (uploads) {
      const auto& fixup = tables->relocations[entry.first_static_relocation];
      EXPECT_EQ(fixup.destination_use, 0u);
      EXPECT_EQ(fixup.source_ordinal, 1u);
      EXPECT_EQ(fixup.byte_offset, kTileEnd + kAddress);
      EXPECT_EQ(fixup.addend, 4);
      EXPECT_EQ(fixup.maximum_value, call_address.maximum_value);
      EXPECT_EQ(fixup.alignment, 4u);
      const uint32_t source = tables->allocation_uses[first_use + 1];
      const auto& allocation = tables->allocations[source];
      EXPECT_EQ(allocation.domain, IREE_XDNA_ELF_ALLOCATION_DOMAIN_DMA);
      EXPECT_EQ(allocation.flags, IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE);
      EXPECT_EQ(allocation.alignment, 64u);
      ASSERT_EQ(allocation.byte_length, data.size());
      std::array<uint8_t, 8> loaded = {};
      IREE_ASSERT_OK(iree_hal_amd_xdna_image_load_allocation(image_, source,
                                                             loaded.data()));
      EXPECT_EQ(loaded, data);
    }
    first_use += entry.allocation_use_count;
  }
}

TEST_F(XdnaProductTest, StreamedEntryPreservesOtherSectionAndWorkerLoads) {
  SyntheticXdnaProduct fixture(1, 2);
  auto product = *fixture.product();
  auto entry = product.entries[0];
  auto program = *entry.array_program;
  auto linked = *entry.tiles[0].linked_tile;
  std::vector<loom_native_section_t> sections(
      linked.assembly.sections,
      linked.assembly.sections + linked.assembly.section_count);
  const uint8_t table[] = {0xa1, 0xb2, 0xc3, 0xd4};
  sections[1].storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS;
  sections[1].contents = iree_make_const_byte_span(table, sizeof(table));
  sections[1].reservation_length = 0;
  linked.assembly.sections = sections.data();
  loom_aie2p_xdna_tile_t tiles[] = {entry.tiles[0], entry.tiles[1]};
  for (auto& tile : tiles) {
    tile.linked_tile = &linked;
  }
  entry.tiles = tiles;
  entry.array_program = &program;
  product.entries = &entry;
  const uint8_t source[] = {0x44, 0x20, 0xc1, 0x20};
  const uint32_t words[8] = {1, 0, 0, 0, 0, 0, 0, 1};
  const loom_aie2p_program_record_t records[] = {
      program.array_records[0],
      program.array_records[1],
      {
          .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
          .value = {.register_block_write32 = {0x1d000u, words, 8}},
      },
  };
  const bool streamed[] = {true, false};
  const loom_aie2p_program_upload_patch_t patch = {2, 0, sizeof(source)};
  const loom_aie2p_program_upload_t upload = {
      .records = records,
      .record_count = IREE_ARRAYSIZE(records),
      .contents = iree_make_const_byte_span(source, sizeof(source)),
      .patches = &patch,
      .patch_count = 1,
      .worker_entries = streamed,
  };
  program.upload = &upload;
  ASSERT_NO_FATAL_FAILURE(LoadOnNpu2(product, 1));
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  const auto& allocation = tables->allocations[0];
  std::vector<uint8_t> commands(allocation.byte_length);
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_image_load_allocation(image_, 0, commands.data()));
  // Each four-byte section is a 16-byte block header followed by its contents.
  const size_t first = 16;
  const size_t second = first + 20;
  const size_t third = second + 20;
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + 8), 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + first + 16),
            iree_unaligned_load_le_u32(table));
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + second + 16),
            iree_unaligned_load_le_u32(source));
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + third + 16),
            iree_unaligned_load_le_u32(table));
  const auto* family =
      loom_xdna_device_profile_array_family(product.device_profile);
  const uint32_t first_row = static_cast<uint32_t>(tiles[0].coordinate.row)
                             << family->row_shift;
  const uint32_t second_row = static_cast<uint32_t>(tiles[1].coordinate.row)
                              << family->row_shift;
  const auto* facts = loom_xdna_array_tile_facts(family, tiles[0].coordinate);
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + first + 8),
            first_row | facts->memory.local_base);
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + second + 8),
            second_row | facts->memory.program_load_base);
  EXPECT_EQ(iree_unaligned_load_le_u32(commands.data() + third + 8),
            second_row | facts->memory.local_base);
  EXPECT_EQ(tables->relocations[0].byte_offset, third + 20 + 16 + 4);
}

TEST_F(XdnaProductTest, InlineLoaderFitsDirectoryAtProgramHeaderLimit) {
  // Each inline entry has three segments. Its DMA alternative adds a trailing
  // descriptor fragment and a data segment. The same native code is shared by
  // all entries in both layouts.
  constexpr uint32_t kEntries =
      (IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT - 1) / 3;
  SyntheticXdnaProduct fixture(kEntries, 1);
  auto product = *fixture.product();
  std::vector<loom_aie2p_xdna_entry_t> entries(product.entries,
                                               product.entries + kEntries);
  product.entries = entries.data();
  auto program = *entries[0].array_program;
  const uint32_t words[8] = {1, 0, 0, 0, 0, 0, 0, 1};
  const loom_aie2p_program_record_t records[] = {
      program.array_records[0],
      {
          .type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32,
          .value = {.register_block_write32 = {0x1d000u, words, 8}},
      },
  };
  const uint8_t bytes[4] = {};
  const loom_aie2p_program_upload_patch_t patch = {1, 0, sizeof(bytes)};
  const loom_aie2p_program_upload_t upload = {
      .records = records,
      .record_count = IREE_ARRAYSIZE(records),
      .contents = iree_make_const_byte_span(bytes, sizeof(bytes)),
      .patches = &patch,
      .patch_count = 1,
  };
  program.upload = &upload;
  for (auto& entry : entries) {
    entry.array_program = &program;
  }
  // The joint directory limit also restores code that one entry would have
  // installed from DMA. The fallback owns a complete runnable cold program.
  const bool worker_entry[] = {true};
  auto worker_upload = upload;
  worker_upload.worker_entries = worker_entry;
  auto worker_program = program;
  worker_program.upload = &worker_upload;
  entries[0].array_program = &worker_program;
  ASSERT_NO_FATAL_FAILURE(LoadOnNpu2(product, 1));
  const auto* tables = iree_hal_amd_xdna_image_tables(image_);
  EXPECT_EQ(tables->header.entry_count, kEntries);
  EXPECT_EQ(tables->header.allocation_count, kEntries);
  EXPECT_EQ(tables->header.relocation_count, 0u);
  const auto& cold = tables->invocations[tables->entries[0].first_invocation];
  EXPECT_EQ(cold.byte_length, 16u + 16u + 4u);
  std::vector<uint8_t> contents(tables->allocations[0].byte_length);
  IREE_ASSERT_OK(
      iree_hal_amd_xdna_image_load_allocation(image_, 0, contents.data()));
  const auto& code =
      entries[0].tiles[0].linked_tile->assembly.sections[0].contents;
  EXPECT_EQ(iree_unaligned_load_le_u32(contents.data() + 32),
            iree_unaligned_load_le_u32(code.data));
}

TEST_F(XdnaProductTest, LoadsInitializedTileSectionsBeforeActivation) {
  const loom_xdna_device_profile_t* profile =
      loom_xdna_device_profile_lookup(IREE_SV("amd.xdna.strix_halo.17f0_11"));
  ASSERT_NE(profile, nullptr);
  const loom_xdna_array_family_t* family =
      loom_xdna_device_profile_array_family(profile);
  ASSERT_NE(family, nullptr);

  uint32_t program_load_base = 0;
  for (uint8_t i = 0; i < family->tile_count; ++i) {
    if (family->tiles[i].kind == LOOM_XDNA_TILE_KIND_COMPUTE) {
      program_load_base = family->tiles[i].memory.program_load_base;
    }
  }
  ASSERT_NE(program_load_base, 0u);

  constexpr loom_xdna_tile_coordinate_t kCoordinate = {1, 2};
  constexpr uint32_t kDataOwnerOffset = 0x4020;
  const std::array<uint8_t, 6> code = {
      0x44, 0x20, 0xc1, 0x20, 0x40, 0x07,
  };
  const std::array<uint8_t, 5> table = {
      0x11, 0x22, 0x33, 0x44, 0x55,
  };
  loom_native_section_t linked_sections[] = {
      {
          .name = IREE_SV(".text.kernel"),
          .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                    LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
          .address = 0,
          .alignment = 16,
          .contents = iree_make_const_byte_span(code.data(), code.size()),
      },
      {
          .name = IREE_SV(".rodata.table"),
          .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
          .access = LOOM_NATIVE_SECTION_ACCESS_READ,
          .address = 0x74020,
          .alignment = 32,
          .contents = iree_make_const_byte_span(table.data(), table.size()),
      },
      {
          .name = IREE_SV(".storage.kernel.scratch"),
          .storage = LOOM_NATIVE_SECTION_STORAGE_RESERVATION,
          .access = LOOM_NATIVE_SECTION_ACCESS_READ |
                    LOOM_NATIVE_SECTION_ACCESS_WRITE,
          .address = 0x70000,
          .alignment = 64,
          .contents = iree_const_byte_span_empty(),
          .reservation_length = 64,
      },
  };
  const loom_aie2p_linked_section_placement_t linked_placements[] = {
      {
          .memory_space = LOOM_XDNA_MEMORY_SPACE_PROGRAM,
          .owner_offset = 0,
      },
      {
          .memory_space = LOOM_XDNA_MEMORY_SPACE_DATA,
          .owner_offset = kDataOwnerOffset,
      },
      {
          .memory_space = LOOM_XDNA_MEMORY_SPACE_DATA,
          .owner_offset = 0,
      },
  };
  const loom_aie2p_linked_tile_t linked_tile = {
      .assembly =
          {
              .sections = linked_sections,
              .section_count = IREE_ARRAYSIZE(linked_sections),
          },
      .section_placements = linked_placements,
      .section_placement_count = IREE_ARRAYSIZE(linked_placements),
      .symbol_layouts = nullptr,
      .symbol_layout_count = 0,
      .entry_section_index = 0,
      .entry_address = 0,
  };
  const loom_native_object_symbol_t entry_symbol = {
      .name = IREE_SV("kernel"),
      .section_contribution_index = 0,
      .section_offset = 0,
      .size = code.size(),
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  loom_aie2p_program_object_t tile_program = {};
  tile_program.contribution.object.symbols = &entry_symbol;
  tile_program.contribution.object.symbol_count = 1;
  tile_program.contribution.realization.entry_symbol_index = 0;

  const loom_aie2p_xdna_tile_t tile = {
      .coordinate = kCoordinate,
      .program_name = entry_symbol.name,
      .program = &tile_program,
      .linked_tile = &linked_tile,
  };
  constexpr uint32_t kActivationAddress = 0x02232000;
  constexpr uint32_t kActivationValue = 1;
  constexpr uint32_t kCompletionAddress = 0x0221F0F0;
  loom_aie2p_program_record_t records[4] = {};
  records[0].type = LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD;
  records[0].value.tile_program_load.tile_program_index = 0;
  records[1].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32;
  records[1].value.register_write32.address = kActivationAddress;
  records[1].value.register_write32.value = kActivationValue;
  records[2].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32;
  records[2].value.register_mask_wait32 = {kCompletionAddress, 0x3F, 1};
  records[3].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_WRITE32;
  records[3].value.register_write32 = {kActivationAddress, 0};
  loom_aie2p_array_program_t array_program = {
      .array_records = records,
      .array_record_count = IREE_ARRAYSIZE(records),
  };

  const loom_aie2p_xdna_entry_t entry = {
      .name = IREE_SV("entry"),
      .column_count = kCoordinate.column + 1,
      .bindings = nullptr,
      .binding_count = 0,
      .array_program = &array_program,
      .tiles = &tile,
      .tile_count = 1,
  };
  const loom_aie2p_xdna_product_t product = {
      .device_profile = profile,
      .entries = &entry,
      .entry_count = 1,
  };

  loom_aie2p_xdna_product_issue_t issue = {};
  loom_aie2p_xdna_product_admission_t* admission = nullptr;
  bool admitted = false;
  IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(&product, &arena_, &admitted,
                                               &admission, &issue));
  ASSERT_TRUE(admitted);
  loom_aie2p_xdna_product_image_t* image = nullptr;
  bool finalized = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_product_finalize(admission, &finalized, &image, &issue));
  ASSERT_TRUE(finalized);
  IREE_ASSERT_OK(loom_aie2p_xdna_product_write(image, stream_));

  std::vector<uint8_t> file_bytes(iree_io_stream_length(stream_));
  IREE_ASSERT_OK(iree_io_stream_seek(stream_, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(iree_io_stream_read(stream_, file_bytes.size(),
                                     file_bytes.data(), nullptr));
  ASSERT_GE(file_bytes.size(), 52u);
  const uint32_t program_header_offset =
      iree_unaligned_load_le_u32(file_bytes.data() + 28);
  const uint16_t program_header_size =
      iree_unaligned_load_le_u16(file_bytes.data() + 42);
  const uint16_t program_header_count =
      iree_unaligned_load_le_u16(file_bytes.data() + 44);
  ASSERT_EQ(program_header_size, 32u);
  ASSERT_LE((uint64_t)program_header_offset +
                (uint64_t)program_header_count * program_header_size,
            file_bytes.size());

  const uint8_t* metadata_header = file_bytes.data() + program_header_offset;
  ASSERT_EQ(iree_unaligned_load_le_u32(metadata_header),
            IREE_XDNA_ELF_PROGRAM_TYPE_METADATA);
  const uint32_t metadata_offset =
      iree_unaligned_load_le_u32(metadata_header + 4);
  const uint32_t metadata_size =
      iree_unaligned_load_le_u32(metadata_header + 16);
  ASSERT_LE((uint64_t)metadata_offset + metadata_size, file_bytes.size());
  const uint8_t* metadata = file_bytes.data() + metadata_offset;
  const iree_xdna_elf_header_record_t product_header =
      iree_xdna_elf_decode_header(metadata);
  ASSERT_EQ(product_header.allocation_count, 1u);
  const iree_xdna_elf_allocation_record_t allocation =
      iree_xdna_elf_decode_allocation(metadata +
                                      IREE_XDNA_ELF_HEADER_RECORD_SIZE);

  std::vector<uint8_t> transaction(allocation.byte_length, 0);
  std::vector<uint32_t> load_offsets;
  for (uint32_t i = 0; i < allocation.load_count; ++i) {
    const uint32_t header_index = allocation.first_load + i;
    ASSERT_LT(header_index, program_header_count);
    const uint8_t* header = file_bytes.data() + program_header_offset +
                            (uint64_t)header_index * program_header_size;
    ASSERT_EQ(iree_unaligned_load_le_u32(header),
              IREE_XDNA_ELF_PROGRAM_TYPE_LOAD);
    const uint32_t file_offset = iree_unaligned_load_le_u32(header + 4);
    const uint32_t load_offset = iree_unaligned_load_le_u32(header + 8);
    const uint32_t allocation_ordinal = iree_unaligned_load_le_u32(header + 12);
    const uint32_t file_size = iree_unaligned_load_le_u32(header + 16);
    const uint32_t memory_size = iree_unaligned_load_le_u32(header + 20);
    ASSERT_EQ(allocation_ordinal, 0u);
    ASSERT_LE(file_size, memory_size);
    ASSERT_LE((uint64_t)file_offset + file_size, file_bytes.size());
    ASSERT_LE((uint64_t)load_offset + memory_size, transaction.size());
    std::memcpy(transaction.data() + load_offset,
                file_bytes.data() + file_offset, file_size);
    load_offsets.push_back(load_offset);
  }

  ASSERT_GE(load_offsets.size(), 5u);
  EXPECT_EQ(load_offsets[0], 0u);
  EXPECT_EQ(load_offsets[1], 32u);
  EXPECT_EQ(load_offsets[2], 40u);
  EXPECT_EQ(load_offsets[3], 56u);
  EXPECT_EQ(load_offsets[4], 64u);
  EXPECT_EQ(iree_unaligned_load_le_u32(transaction.data() + 8), 5u);

  const uint32_t coordinate_address =
      ((uint32_t)kCoordinate.column << family->column_shift) |
      ((uint32_t)kCoordinate.row << family->row_shift);
  const uint8_t* code_load = transaction.data() + 16;
  EXPECT_EQ(code_load[0], 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(code_load + 8),
            coordinate_address | program_load_base);
  EXPECT_EQ(iree_unaligned_load_le_u32(code_load + 12), 24u);
  EXPECT_EQ(0, std::memcmp(transaction.data() + 32, code.data(), code.size()));
  EXPECT_EQ(transaction[38], 0u);
  EXPECT_EQ(transaction[39], 0u);

  const uint8_t* data_load = transaction.data() + 40;
  EXPECT_EQ(data_load[0], 1u);
  EXPECT_EQ(iree_unaligned_load_le_u32(data_load + 8),
            coordinate_address | kDataOwnerOffset);
  EXPECT_EQ(iree_unaligned_load_le_u32(data_load + 12), 24u);
  EXPECT_EQ(0,
            std::memcmp(transaction.data() + 56, table.data(), table.size()));
  EXPECT_EQ(transaction[61], 0u);
  EXPECT_EQ(transaction[62], 0u);
  EXPECT_EQ(transaction[63], 0u);

  const uint8_t* activation = transaction.data() + 64;
  EXPECT_EQ(iree_unaligned_load_le_u32(activation + 8), kActivationAddress);
  EXPECT_EQ(iree_unaligned_load_le_u32(activation + 16), kActivationValue);
  EXPECT_EQ(iree_unaligned_load_le_u32(activation + 20), 24u);

  // Firmware transaction 0.1's masked poll is a complete 32-byte record,
  // including its reserved tail word. The stop write follows the poll.
  const std::array<uint8_t, 32> completion = {
      4, 0, 0, 0, 0,    0, 0, 0, 0xF0, 0xF0, 0x21, 2, 0, 0, 0, 0,
      1, 0, 0, 0, 0x3F, 0, 0, 0, 32,   0,    0,    0, 0, 0, 0, 0,
  };
  ASSERT_GE(transaction.size(), 144u);
  EXPECT_EQ(iree_unaligned_load_le_u32(transaction.data() + 12), 144u);
  EXPECT_EQ(0, std::memcmp(transaction.data() + 88, completion.data(),
                           completion.size()));
  const uint8_t* stop = transaction.data() + 120;
  EXPECT_EQ(stop[0], 0u);
  EXPECT_EQ(iree_unaligned_load_le_u32(stop + 8), kActivationAddress);
  EXPECT_EQ(iree_unaligned_load_le_u32(stop + 16), 0u);
  EXPECT_EQ(iree_unaligned_load_le_u32(stop + 20), 24u);
}

TEST(Aie2pXdnaProductTest, DeduplicatesSharedSectionsBeforeDirectoryAdmission) {
  constexpr iree_host_size_t kEntryCount = 100;
  constexpr iree_host_size_t kTileCount = 19;
  static_assert(kEntryCount * kTileCount * 3 >
                IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT);
  SyntheticXdnaProduct fixture(kEntryCount, kTileCount);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_aie2p_xdna_product_issue_t issue = {};
  loom_aie2p_xdna_product_admission_t* admission = nullptr;
  bool admitted = false;
  IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(fixture.product(), &arena,
                                               &admitted, &admission, &issue));
  ASSERT_TRUE(admitted);
  loom_aie2p_xdna_product_image_t* image = nullptr;
  bool finalized = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_product_finalize(admission, &finalized, &image, &issue));
  ASSERT_TRUE(finalized);

  iree_io_stream_t* stream = nullptr;
  IREE_ASSERT_OK(iree_io_vec_stream_create(
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_WRITABLE |
          IREE_IO_STREAM_MODE_SEEKABLE,
      4096, iree_allocator_system(), &stream));
  IREE_ASSERT_OK(loom_aie2p_xdna_product_write(image, stream));
  std::array<uint8_t, 52> header = {};
  IREE_ASSERT_OK(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
  IREE_ASSERT_OK(
      iree_io_stream_read(stream, header.size(), header.data(), nullptr));
  EXPECT_EQ(iree_unaligned_load_le_u16(header.data() + 44), 3901u);
  EXPECT_EQ(iree_unaligned_load_le_u16(header.data() + 48), 2008u);

  iree_io_stream_release(stream);
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST_F(XdnaProductTest, RejectsExactSectionHeaderOverflow) {
  const loom_xdna_device_profile_t* profile =
      loom_xdna_device_profile_lookup(IREE_SV("amd.xdna.strix_halo.17f0_11"));
  ASSERT_NE(profile, nullptr);

  // Five fixed sections and two command fragments leave room for 4,089
  // resident sections. Access-free sections remain distinct by contract, so
  // one more proves that admission uses the exact post-intern count.
  constexpr iree_host_size_t kLinkedSectionCount = 4090;
  constexpr uint64_t kExpectedSectionCount = 5u + 2u + kLinkedSectionCount;
  static_assert(kExpectedSectionCount ==
                IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT + 1u);
  const std::array<uint8_t, 4> code = {0x44, 0x20, 0xc1, 0x20};
  std::vector<loom_native_section_t> linked_sections(kLinkedSectionCount);
  std::vector<loom_aie2p_linked_section_placement_t> linked_placements(
      kLinkedSectionCount);
  linked_sections[0] = loom_native_section_t{
      .name = IREE_SV(".text.kernel"),
      .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
      .access =
          LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE,
      .address = 0,
      .alignment = 16,
      .contents = iree_make_const_byte_span(code.data(), code.size()),
  };
  linked_placements[0].memory_space = LOOM_XDNA_MEMORY_SPACE_PROGRAM;
  for (iree_host_size_t i = 1; i < kLinkedSectionCount; ++i) {
    linked_sections[i] = loom_native_section_t{
        .name = IREE_SV(".discard"),
        .storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS,
        .access = LOOM_NATIVE_SECTION_ACCESS_NONE,
        .address = (uint32_t)i,
        .alignment = 1,
        .contents = iree_const_byte_span_empty(),
    };
    linked_placements[i].memory_space = LOOM_XDNA_MEMORY_SPACE_PROGRAM;
  }
  const loom_aie2p_linked_tile_t linked_tile = {
      .assembly =
          {
              .sections = linked_sections.data(),
              .section_count = linked_sections.size(),
          },
      .section_placements = linked_placements.data(),
      .section_placement_count = linked_placements.size(),
      .symbol_layouts = nullptr,
      .symbol_layout_count = 0,
      .entry_section_index = 0,
      .entry_address = 0,
  };
  const loom_native_object_symbol_t entry_symbol = {
      .name = IREE_SV("kernel"),
      .section_contribution_index = 0,
      .section_offset = 0,
      .size = code.size(),
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };
  loom_aie2p_program_object_t tile_program = {};
  tile_program.contribution.object.symbols = &entry_symbol;
  tile_program.contribution.object.symbol_count = 1;
  tile_program.contribution.realization.entry_symbol_index = 0;
  const loom_aie2p_xdna_tile_t tile = {
      .coordinate = {0, 2},
      .program_name = entry_symbol.name,
      .program = &tile_program,
      .linked_tile = &linked_tile,
  };
  loom_aie2p_program_record_t record = {
      .type = LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD,
  };
  record.value.tile_program_load.tile_program_index = 0;
  const loom_aie2p_array_program_t program = {
      .array_records = &record,
      .array_record_count = 1,
      .control_records = nullptr,
      .control_record_count = 0,
      .relocations = nullptr,
      .relocation_count = 0,
  };
  const loom_aie2p_xdna_entry_t entry = {
      .name = IREE_SV("entry"),
      .column_count = 1,
      .bindings = nullptr,
      .binding_count = 0,
      .array_program = &program,
      .tiles = &tile,
      .tile_count = 1,
  };
  const loom_aie2p_xdna_product_t product = {
      .device_profile = profile,
      .entries = &entry,
      .entry_count = 1,
  };

  loom_aie2p_xdna_product_issue_t issue = {};
  loom_aie2p_xdna_product_admission_t* admission = nullptr;
  bool admitted = false;
  IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(&product, &arena_, &admitted,
                                               &admission, &issue));
  ASSERT_TRUE(admitted);
  loom_aie2p_xdna_product_image_t* image = nullptr;
  bool finalized = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_product_finalize(admission, &finalized, &image, &issue));
  EXPECT_FALSE(finalized);
  EXPECT_EQ(image, nullptr);
  EXPECT_EQ(issue.kind, LOOM_AIE2P_XDNA_PRODUCT_ISSUE_SECTION_HEADER_COUNT);
  EXPECT_EQ(issue.entry_ordinal, UINT32_MAX);
  EXPECT_EQ(issue.actual, kExpectedSectionCount);
  EXPECT_EQ(issue.maximum, IREE_XDNA_ELF_MAX_SECTION_HEADER_COUNT);
}

TEST(Aie2pXdnaProductTest, RejectsExactProgramHeaderOverflow) {
  constexpr iree_host_size_t kEntryCount = 65;
  constexpr iree_host_size_t kTileCount = 32;
  static_assert(1 + kEntryCount * (2 * kTileCount + 1) == 4226);
  SyntheticXdnaProduct fixture(kEntryCount, kTileCount);
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);

  loom_aie2p_xdna_product_issue_t issue = {};
  loom_aie2p_xdna_product_admission_t* admission = nullptr;
  bool admitted = false;
  IREE_ASSERT_OK(loom_aie2p_xdna_product_admit(fixture.product(), &arena,
                                               &admitted, &admission, &issue));
  ASSERT_TRUE(admitted);
  loom_aie2p_xdna_product_image_t* image = nullptr;
  bool finalized = false;
  IREE_ASSERT_OK(
      loom_aie2p_xdna_product_finalize(admission, &finalized, &image, &issue));
  EXPECT_FALSE(finalized);
  EXPECT_EQ(image, nullptr);
  EXPECT_EQ(issue.kind, LOOM_AIE2P_XDNA_PRODUCT_ISSUE_PROGRAM_HEADER_COUNT);
  EXPECT_EQ(issue.actual, 4226u);
  EXPECT_EQ(issue.maximum, IREE_XDNA_ELF_MAX_PROGRAM_HEADER_COUNT);

  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);
}

}  // namespace
}  // namespace loom
