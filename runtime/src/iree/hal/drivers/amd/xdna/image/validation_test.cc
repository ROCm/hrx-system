// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/image/validation.h"

#include <memory>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::Aie2pImageFixtureOptions;
using iree::hal::amd::xdna::testing::Aie2pImageTargetState;
using iree::hal::amd::xdna::testing::BuildAie2pImage;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::kAie2pImageStructuralCapabilities;
using iree::hal::amd::xdna::testing::kAie2pImageTileProgramHeaderOrdinal;
using iree::hal::amd::xdna::testing::MakeAie2pImageTarget;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;
using testing::HasSubstr;

struct ValidationDeleter {
  void operator()(iree_hal_amd_xdna_image_validation_t* validation) const {
    iree_hal_amd_xdna_image_validation_destroy(validation);
  }
};

using ValidationPtr =
    std::unique_ptr<iree_hal_amd_xdna_image_validation_t, ValidationDeleter>;

static Status ValidateImage(
    const std::vector<uint8_t>& bytes,
    const iree_hal_amd_xdna_aie2p_target_t* aie2p_target,
    ValidationPtr* out_validation = nullptr) {
  ByteSequencePtr sequence = MakeOwnedByteSequence(bytes);
  iree_hal_amd_xdna_image_directory_t* directory = nullptr;
  iree_hal_amd_xdna_image_tables_t* tables = nullptr;
  iree_hal_amd_xdna_image_programs_t* programs = nullptr;
  iree_hal_amd_xdna_image_validation_t* validation = nullptr;
  iree_hal_amd_xdna_image_target_t image_target;
  iree_status_t status = iree_hal_amd_xdna_aie2p_target_initialize_image_target(
      aie2p_target, &image_target);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_directory_create(
        sequence.get(), iree_allocator_system(), &directory);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_tables_create(
        directory, iree_allocator_system(), &tables);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_programs_create(
        directory, image_target.program_record_validator,
        iree_allocator_system(), &programs);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_image_validation_create(
        directory, tables, programs, &image_target, iree_allocator_system(),
        &validation);
  }
  iree_hal_amd_xdna_image_programs_destroy(programs);
  iree_hal_amd_xdna_image_tables_destroy(tables);
  iree_hal_amd_xdna_image_directory_destroy(directory);

  if (iree_status_is_ok(status)) {
    if (out_validation != nullptr) {
      out_validation->reset(validation);
    } else {
      iree_hal_amd_xdna_image_validation_destroy(validation);
    }
  } else {
    iree_hal_amd_xdna_image_validation_destroy(validation);
  }
  return Status(std::move(status));
}

static void ExpectStatus(const Status& status, StatusCode expected_code,
                         const char* expected_message_substring) {
  EXPECT_EQ(status.code(), expected_code);
  EXPECT_THAT(status.ToString(), HasSubstr(expected_message_substring));
}

TEST(ImageValidationTest, QualifiesCompleteImage) {
  Aie2pImageTargetState state;
  const iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&state);
  ValidationPtr validation;
  IREE_ASSERT_OK(ValidateImage(BuildAie2pImage(), &target, &validation));

  EXPECT_EQ(iree_hal_amd_xdna_image_validation_structural_capabilities(
                validation.get()),
            kAie2pImageStructuralCapabilities);
  ASSERT_EQ(
      iree_hal_amd_xdna_image_validation_tile_placement_count(validation.get()),
      1u);
  const iree_hal_amd_xdna_image_tile_placement_t* placement =
      iree_hal_amd_xdna_image_validation_tile_placement(validation.get(), 0);
  ASSERT_NE(placement, nullptr);
  EXPECT_EQ(placement->program_header_ordinal,
            kAie2pImageTileProgramHeaderOrdinal);
  EXPECT_EQ(placement->owner_column, 1u);
  EXPECT_EQ(placement->owner_row, 2u);
  EXPECT_EQ(placement->memory_space,
            IREE_HAL_AMD_XDNA_ELF_TILE_MEMORY_SPACE_PROGRAM);
  EXPECT_EQ(placement->owner_offset, 0u);
  EXPECT_EQ(placement->byte_length, 32u);
  EXPECT_EQ(state.configuration_register_call_count, 1u);
}

TEST(ImageValidationTest, RejectsTargetAndCapabilityMismatches) {
  Aie2pImageTargetState state;
  iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&state);
  target.identity.policy_id ^= 1;
  ExpectStatus(ValidateImage(BuildAie2pImage(), &target),
               StatusCode::kFailedPrecondition, "does not match the target");

  target = MakeAie2pImageTarget(&state);
  target.supported_capabilities =
      IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS;
  ExpectStatus(ValidateImage(BuildAie2pImage(), &target),
               StatusCode::kUnimplemented, "unsupported by the target");

  target = MakeAie2pImageTarget(&state);
  Aie2pImageFixtureOptions options;
  options.required_capabilities =
      IREE_HAL_AMD_XDNA_ELF_CAPABILITY_RUNTIME_RELOCATIONS;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kInvalidArgument,
               "capability note disagrees with its program directory");
}

TEST(ImageValidationTest, RejectsDanglingProgramRelationships) {
  Aie2pImageTargetState state;
  const iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&state);
  Aie2pImageFixtureOptions options;
  options.array_tile_program_header_ordinal = 3;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kInvalidArgument,
               "invalid or duplicate TILE reference");

  options = {};
  options.entry_array_program_header_ordinal =
      kAie2pImageTileProgramHeaderOrdinal;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kInvalidArgument,
               "references an invalid ARRAY realization");

  options = {};
  options.entry_control_program_header_ordinal = UINT32_MAX;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kInvalidArgument,
               "CONTROL program is not referenced by an entry");
}

TEST(ImageValidationTest, RejectsInvalidRelocationTargetsAndValues) {
  Aie2pImageTargetState state;
  const iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&state);
  Aie2pImageFixtureOptions options;
  options.relocation_target_byte_offset =
      IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kPermissionDenied,
               "targets immutable program framing");

  options = {};
  options.relocation_minimum_value = 1;
  options.relocation_maximum_value = 2;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kInvalidArgument,
               "no value satisfying its range and alignment");

  options = {};
  options.use_relocatable_block_write = false;
  options.relocation_target_byte_offset =
      IREE_HAL_AMD_XDNA_ELF_CONTROL_HEADER_SIZE + 12;
  options.relocation_field_byte_width = 4;
  options.relocation_maximum_value = UINT32_MAX;
  ExpectStatus(ValidateImage(BuildAie2pImage(options), &target),
               StatusCode::kPermissionDenied,
               "does not target block-write word data");
}

TEST(ImageValidationTest, RejectsInvalidCanonicalTilePlacement) {
  Aie2pImageTargetState state;
  state.owner_column = 9;
  const iree_hal_amd_xdna_aie2p_target_t target = MakeAie2pImageTarget(&state);
  ExpectStatus(ValidateImage(BuildAie2pImage(), &target),
               StatusCode::kInvalidArgument,
               "resolver returned an invalid placement");
}

}  // namespace
