// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/xdna/executable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "iree/hal/drivers/amd/xdna/image/aie2p/npu2.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32.h"
#include "iree/hal/drivers/amd/xdna/image/testdata/mul_i32_npu4.h"
#include "iree/hal/drivers/amd/xdna/image/testing/aie2p_image_fixture.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::hal::amd::xdna::testing::ByteSequencePtr;
using iree::hal::amd::xdna::testing::MakeOwnedByteSequence;

struct ExecutableDeleter {
  void operator()(iree_hal_executable_t* executable) const {
    iree_hal_executable_release(executable);
  }
};

using ExecutablePtr = std::unique_ptr<iree_hal_executable_t, ExecutableDeleter>;

static ByteSequencePtr LoadMulI32Image() {
  EXPECT_EQ(iree_hal_amd_xdna_test_mul_i32_size(), 1u);
  if (iree_hal_amd_xdna_test_mul_i32_size() != 1u) return {};
  const iree_file_toc_t* file = iree_hal_amd_xdna_test_mul_i32_create();
  const auto* begin = reinterpret_cast<const uint8_t*>(file->data);
  return MakeOwnedByteSequence(std::vector<uint8_t>(begin, begin + file->size));
}

class XdnaExecutableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_family_spec_.name = IREE_SV("test");
    queue_family_spec_.physical_device_affinity = 1;
    queue_family_spec_.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH;
    iree_hal_queue_family_initialize(/*ordinal=*/7, &queue_family_spec_,
                                     &queue_family_);
  }

  ExecutablePtr LoadCanonical() {
    ByteSequencePtr sequence = LoadMulI32Image();
    iree_hal_amd_xdna_aie2p_target_t target;
    IREE_CHECK_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
        IREE_SV("amd.xdna.strix_halo.17f0_11"),
        /*context_column_count=*/1, &target));
    iree_hal_executable_t* executable = nullptr;
    IREE_CHECK_OK(iree_hal_amd_xdna_executable_create(
        &queue_family_, sequence.get(), &target, iree_allocator_system(),
        &executable));
    return ExecutablePtr(executable);
  }

  // Executable-only family metadata; this fixture provisions no HAL queues.
  iree_hal_queue_family_spec_t queue_family_spec_ = {};
  // HAL queue family borrowed by every executable created by this fixture.
  iree_hal_queue_family_t queue_family_ = {};
};

TEST_F(XdnaExecutableTest, LoadsCanonicalImageAndReflection) {
  ExecutablePtr executable = LoadCanonical();

  EXPECT_EQ(iree_hal_executable_queue_family(executable.get()), &queue_family_);
  EXPECT_TRUE(iree_hal_amd_xdna_executable_isa(executable.get()));
  EXPECT_EQ(iree_hal_executable_function_count(executable.get()), 1u);

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable.get(), IREE_SV("mul_i32"), &function));
  EXPECT_EQ(function.value, 0u);

  iree_hal_executable_function_info_t function_info;
  IREE_ASSERT_OK(iree_hal_executable_function_info(executable.get(), function,
                                                   &function_info));
  EXPECT_TRUE(iree_string_view_equal(function_info.name, IREE_SV("mul_i32")));
  EXPECT_EQ(function_info.binding_count, 3u);
  EXPECT_EQ(function_info.parameter_count, 3u);
  EXPECT_EQ(function_info.maximum_workgroup_invocations, 1u);
  EXPECT_EQ(function_info.workgroup_size[0], 1u);
  EXPECT_EQ(function_info.workgroup_size[1], 1u);
  EXPECT_EQ(function_info.workgroup_size[2], 1u);

  std::array<iree_hal_executable_function_parameter_t, 3> parameters;
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable.get(), function, parameters.size(), parameters.data()));
  for (iree_host_size_t i = 0; i < parameters.size(); ++i) {
    EXPECT_EQ(parameters[i].type,
              IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING);
    EXPECT_EQ(parameters[i].offset, i);
  }

  iree_hal_amd_xdna_executable_entry_t entry;
  IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_entry(executable.get(),
                                                          function, &entry));
  EXPECT_EQ(entry.binding_count, 3u);
  EXPECT_EQ(entry.native.relocation_count, 3u);
  EXPECT_EQ(iree_unaligned_load_le_u32(entry.array.data + 8), 50u);
  ASSERT_GE(entry.native.control.data_length, 16u);
  EXPECT_EQ(iree_unaligned_load_le_u32(entry.native.control.data + 8), 8u);
  EXPECT_EQ(iree_unaligned_load_le_u32(entry.native.control.data + 12),
            entry.native.control.data_length);

  constexpr uint64_t kExpectedBindingByteLengths[] = {64, 64, 64};
  for (iree_host_size_t i = 0; i < entry.binding_count; ++i) {
    iree_hal_amd_xdna_elf_binding_record_t binding;
    IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_binding(
        executable.get(), function, i, &binding));
    EXPECT_EQ(binding.binding_ordinal, i);
    EXPECT_EQ(binding.entry_ordinal, 0u);
    EXPECT_EQ(binding.kind, IREE_HAL_AMD_XDNA_ELF_BINDING_KIND_BUFFER);
    EXPECT_EQ(binding.address_space,
              IREE_HAL_AMD_XDNA_ELF_BINDING_ADDRESS_SPACE_GLOBAL);
    EXPECT_EQ(binding.access, i == 2
                                  ? IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_WRITE
                                  : IREE_HAL_AMD_XDNA_ELF_BINDING_ACCESS_READ);
    EXPECT_EQ(binding.usage,
              IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_DEVICE_VISIBLE |
                  IREE_HAL_AMD_XDNA_ELF_BINDING_USAGE_COHERENT);
    EXPECT_EQ(binding.minimum_byte_length, kExpectedBindingByteLengths[i]);
    EXPECT_EQ(binding.minimum_alignment, 4u);
  }
}

TEST_F(XdnaExecutableTest, CanonicalImagesRequireTheirExactDeviceProfile) {
  const iree_file_toc_t* images[] = {
      iree_hal_amd_xdna_test_mul_i32_npu4_create(),
      iree_hal_amd_xdna_test_mul_i32_create(),
  };
  const iree_string_view_t target_ids[] = {
      IREE_SVL("amd.xdna.strix.17f0_10"),
      IREE_SVL("amd.xdna.strix_halo.17f0_11"),
  };
  for (size_t image_ordinal = 0; image_ordinal < 2; ++image_ordinal) {
    const auto* begin =
        reinterpret_cast<const uint8_t*>(images[image_ordinal]->data);
    ByteSequencePtr sequence = MakeOwnedByteSequence(
        std::vector<uint8_t>(begin, begin + images[image_ordinal]->size));
    for (size_t target_ordinal = 0; target_ordinal < 2; ++target_ordinal) {
      SCOPED_TRACE(::testing::Message() << "image=" << image_ordinal
                                        << " target=" << target_ordinal);
      iree_hal_amd_xdna_aie2p_target_t target;
      IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
          target_ids[target_ordinal], 1, &target));
      auto* executable = reinterpret_cast<iree_hal_executable_t*>(uintptr_t{1});
      auto* sentinel = executable;
      Status status(iree_hal_amd_xdna_executable_create(
          &queue_family_, sequence.get(), &target, iree_allocator_system(),
          &executable));
      if (image_ordinal != target_ordinal) {
        EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
        EXPECT_EQ(executable, sentinel);
        continue;
      }
      IREE_ASSERT_OK(status);
      ExecutablePtr owned_executable(executable);
      iree_hal_executable_function_t function;
      IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
          executable, IREE_SV("mul_i32"), &function));
      iree_hal_amd_xdna_executable_entry_t entry;
      IREE_ASSERT_OK(iree_hal_amd_xdna_executable_query_entry(
          executable, function, &entry));
      EXPECT_EQ(entry.binding_count, 3u);
      EXPECT_EQ(entry.native.relocation_count, 3u);
      EXPECT_GT(entry.array.data_length, 0u);
      EXPECT_GT(entry.native.control.data_length, 0u);
    }
  }
}

TEST_F(XdnaExecutableTest, RejectsMismatchedTargetWithoutPublishing) {
  ByteSequencePtr sequence = LoadMulI32Image();
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &target));
  ++target.identity.policy_id;
  auto* executable = reinterpret_cast<iree_hal_executable_t*>(uintptr_t{1});
  auto* sentinel = executable;
  IREE_EXPECT_STATUS_IS(StatusCode::kFailedPrecondition,
                        iree_hal_amd_xdna_executable_create(
                            &queue_family_, sequence.get(), &target,
                            iree_allocator_system(), &executable));
  EXPECT_EQ(executable, sentinel);
}

TEST_F(XdnaExecutableTest, AllocationFailureDoesNotPublish) {
  ByteSequencePtr sequence = LoadMulI32Image();
  iree_hal_amd_xdna_aie2p_target_t target;
  IREE_ASSERT_OK(iree_hal_amd_xdna_aie2p_npu2_target_initialize(
      IREE_SV("amd.xdna.strix_halo.17f0_11"), 1, &target));
  auto* executable = reinterpret_cast<iree_hal_executable_t*>(uintptr_t{1});
  auto* sentinel = executable;
  IREE_EXPECT_STATUS_IS(StatusCode::kInvalidArgument,
                        iree_hal_amd_xdna_executable_create(
                            &queue_family_, sequence.get(), &target,
                            iree_allocator_null(), &executable));
  EXPECT_EQ(executable, sentinel);
}

}  // namespace
