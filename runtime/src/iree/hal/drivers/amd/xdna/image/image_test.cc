// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/image.h"

#include <functional>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/testing/image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ImageFixture;
using iree::hal::amd::xdna::testing::MakeImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

static Status Admit(const std::vector<uint8_t>& bytes) {
  auto source = MakeOwnedByteSequence(bytes);
  const auto target = MakeImageTarget();
  iree_hal_amd_xdna_image_t* image = nullptr;
  Status status(iree_hal_amd_xdna_image_create(
      source.get(), &target, iree_allocator_system(), &image));
  iree_hal_amd_xdna_image_destroy(image);
  return status;
}

TEST(ImageTest, RetainsSourceAndIndexedRequirements) {
  const ImageFixture fixture;
  auto source = MakeOwnedByteSequence(fixture.Build());
  const auto target = MakeImageTarget();
  iree_hal_amd_xdna_image_t* image = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_xdna_image_create(
      source.get(), &target, iree_allocator_system(), &image));
  source.reset();
  const auto* tables = iree_hal_amd_xdna_image_tables(image);
  EXPECT_EQ(tables->header.allocation_count, 2u);
  EXPECT_EQ(tables->header.entry_count, 1u);
  uint32_t ordinal = UINT32_MAX;
  IREE_EXPECT_OK(
      iree_hal_amd_xdna_image_find_entry(image, IREE_SV("main"), &ordinal));
  EXPECT_EQ(ordinal, 0u);
  IREE_EXPECT_STATUS_IS(
      StatusCode::kNotFound,
      iree_hal_amd_xdna_image_find_entry(image, IREE_SV("absent"), &ordinal));
  const auto entry = iree_hal_amd_xdna_image_tables_entry(tables, 0);
  EXPECT_EQ(entry.allocation_use_count, 2u);
  EXPECT_EQ(iree_hal_amd_xdna_image_tables_allocation_use(tables, 1), 1u);
  const auto invocation =
      iree_hal_amd_xdna_image_tables_invocation(tables, entry.first_invocation);
  EXPECT_EQ(invocation.next_invocation, 1u);
  const auto* directory = iree_hal_amd_xdna_image_directory(image);
  const auto* load =
      iree_hal_amd_xdna_image_directory_program_header(directory, 3);
  std::vector<uint8_t> bytes(load->file_range.length);
  IREE_EXPECT_OK(iree_hal_amd_xdna_image_directory_read_source_range(
      directory, load->file_range,
      iree_make_byte_span(bytes.data(), bytes.size())));
  EXPECT_EQ(bytes, fixture.payloads[2]);
  iree_hal_amd_xdna_image_destroy(image);
}

TEST(ImageTest, AdmitsOnlyExactUnreferencedBindingSlots) {
  ImageFixture fixture;
  fixture.bindings.insert(fixture.bindings.begin(),
                          iree_xdna_elf_binding_record_t{});
  fixture.entries[0].binding_count = 2;
  fixture.relocations[1].source_ordinal = 1;
  IREE_EXPECT_OK(Admit(fixture.Build()));

  fixture.relocations[1].source_ordinal = 0;
  EXPECT_EQ(Admit(fixture.Build()).code(), StatusCode::kInvalidArgument);

  fixture.relocations[1].source_ordinal = 1;
  fixture.bindings[0].minimum_alignment = 1;
  EXPECT_EQ(Admit(fixture.Build()).code(), StatusCode::kInvalidArgument);
}

TEST(ImageTest, RejectsIncompatibleExecutionContracts) {
  const std::function<void(ImageFixture&)> mutations[] = {
      [](auto& f) { ++f.header.device_profile_id; },
      [](auto& f) { ++f.header.device_profile_revision; },
      [](auto& f) { ++f.header.firmware_abi_id; },
      [](auto& f) { ++f.header.target_generation; },
      [](auto& f) { ++f.header.native_encoding; },
      [](auto& f) { ++f.header.column_count; },
      [](auto& f) { ++f.header.row_count; },
  };
  for (size_t i = 0; i < std::size(mutations); ++i) {
    SCOPED_TRACE(i);
    ImageFixture fixture;
    mutations[i](fixture);
    EXPECT_EQ(Admit(fixture.Build()).code(), StatusCode::kFailedPrecondition);
  }
}

TEST(ImageTest, RejectsMalformedBackingAndReferences) {
  const std::function<void(ImageFixture&)> mutations[] = {
      [](auto& f) { f.header.magic = 0; },
      [](auto& f) { ++f.header.version; },
      [](auto& f) { f.allocations[0].domain = 0; },
      [](auto& f) { f.allocations[0].flags = 3; },
      [](auto& f) { f.allocations[0].byte_length = 32783; },
      [](auto& f) { f.allocations[0].alignment = 3; },
      [](auto& f) { f.allocations[0].first_load = 2; },
      [](auto& f) { f.allocations[0].load_count = UINT32_MAX; },
      [](auto& f) { f.allocations[1].load_count = 0; },
      [](auto& f) { f.loads[0].type = 0; },
      [](auto& f) { f.loads[0].flags |= IREE_XDNA_ELF_PROGRAM_FLAG_WRITE; },
      [](auto& f) { f.loads[0].physical_address = 1; },
      [](auto& f) { f.loads[1].virtual_address = 12; },
      [](auto& f) { f.uses[0] = 2; },
      [](auto& f) { f.uses[1] = 0; },
      [](auto& f) { f.entries[0].allocation_use_count = 3; },
      [](auto& f) { f.entries[0].first_binding = 1; },
      [](auto& f) { f.entries[0].binding_count = UINT32_MAX; },
      [](auto& f) { f.entries[0].first_static_relocation = 1; },
      [](auto& f) { f.entries[0].static_relocation_count = 3; },
      [](auto& f) { f.entries[0].first_dynamic_relocation = 0; },
      [](auto& f) { f.entries[0].dynamic_relocation_count = UINT32_MAX; },
      [](auto& f) { f.entries[0].name_offset = UINT32_MAX; },
      [](auto& f) { f.entries[0].name_length = 0; },
      [](auto& f) { f.entries[0].first_invocation = 1; },
      [](auto& f) { f.entries[0].invocation_count = 0; },
      [](auto& f) { f.invocations[0].allocation_use = 2; },
      [](auto& f) { f.invocations[0].allocation_use = 1; },
      [](auto& f) { f.invocations[0].next_invocation = 2; },
      [](auto& f) { f.invocations[0].byte_offset = 4; },
      [](auto& f) { f.invocations[0].byte_length = 0; },
      [](auto& f) { f.invocations[0].byte_length = 18; },
      [](auto& f) { f.invocations[0].byte_length = 32784; },
      [](auto& f) { f.invocations[1].byte_length = 20; },
      [](auto& f) { f.bindings[0].kind = 0; },
      [](auto& f) { f.bindings[0].address_space = 0; },
      [](auto& f) { f.bindings[0].access = 0; },
      [](auto& f) { f.bindings[0].access = 4; },
      [](auto& f) { f.bindings[0].usage = 16; },
      [](auto& f) { f.bindings[0].minimum_alignment = 0; },
      [](auto& f) {
        f.bindings[0].minimum_byte_offset = 1;
        f.bindings[0].maximum_byte_offset = 0;
      },
      [](auto& f) { f.relocations[0].destination_use = 2; },
      [](auto& f) { f.relocations[0].source_ordinal = 2; },
      [](auto& f) { f.relocations[0].source_ordinal = 0; },
      [](auto& f) { f.relocations[1].source_ordinal = 1; },
      [](auto& f) { f.relocations[0].kind = 0; },
      [](auto& f) { f.relocations[0].byte_offset = 2; },
      [](auto& f) { f.relocations[0].byte_offset = 16; },
      [](auto& f) { f.relocations[0].byte_offset = UINT32_MAX - 3; },
      [](auto& f) { f.relocations[0].minimum_value = UINT64_MAX; },
      [](auto& f) { f.relocations[0].maximum_value = UINT64_MAX; },
      [](auto& f) { f.relocations[0].alignment = 2; },
      [](auto& f) { f.relocations[0].alignment = 12; },
      [](auto& f) { f.relocations[1].byte_offset = 0; },
      [](auto& f) {
        f.allocations[0].flags = IREE_XDNA_ELF_ALLOCATION_FLAG_IMMUTABLE;
      },
  };
  for (size_t i = 0; i < std::size(mutations); ++i) {
    SCOPED_TRACE(i);
    ImageFixture fixture;
    mutations[i](fixture);
    EXPECT_EQ(Admit(fixture.Build()).code(), StatusCode::kInvalidArgument);
  }
}

TEST(ImageTest, RejectsTruncatedOrExcessiveMetadataBeforeRowAccess) {
  auto fixture = ImageFixture();
  for (uint32_t count : {65536u, UINT32_MAX}) {
    auto bytes = fixture.Build();
    auto header = fixture.header;
    header.allocation_count = count;
    iree_xdna_elf_encode_header(&header, bytes.data() + 512);
    EXPECT_EQ(Admit(bytes).code(), StatusCode::kOutOfRange);
  }
  auto bytes = fixture.Build();
  auto header = fixture.header;
  header.allocation_count = 1024;
  iree_xdna_elf_encode_header(&header, bytes.data() + 512);
  EXPECT_EQ(Admit(bytes).code(), StatusCode::kInvalidArgument);
}

static iree_status_t RejectAllocation(void* self,
                                      iree_allocator_command_t command,
                                      const void* params,
                                      void** inout_pointer) {
  (void)self;
  (void)command;
  (void)params;
  (void)inout_pointer;
  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                          "allocation rejected");
}

TEST(ImageTest, DoesNotPublishImageOnAllocationFailure) {
  auto source = MakeOwnedByteSequence(ImageFixture().Build());
  const auto target = MakeImageTarget();
  iree_hal_amd_xdna_image_t* image = nullptr;
  const iree_allocator_t allocator = {nullptr, RejectAllocation};
  IREE_EXPECT_STATUS_IS(
      StatusCode::kResourceExhausted,
      iree_hal_amd_xdna_image_create(source.get(), &target, allocator, &image));
  EXPECT_EQ(image, nullptr);
}

}  // namespace
