// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/vm/launch_config.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using ByteSequencePtr =
    HandlePtr<loomc_byte_sequence_t, loomc_byte_sequence_release>;

ByteSequencePtr CreateSequence(loomc_byte_span_t contents) {
  loomc_byte_sequence_t* sequence = nullptr;
  LOOMC_EXPECT_OK(loomc_byte_sequence_create_copy(
      contents, loomc_allocator_system(), &sequence));
  return ByteSequencePtr(sequence);
}

TEST(VmLaunchConfigProgramTest, RejectsNullArtifactAndClearsOutput) {
  loomc_vm_launch_config_program_t* program =
      reinterpret_cast<loomc_vm_launch_config_program_t*>(UINTPTR_MAX);
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_vm_launch_config_program_load(
          /*artifact=*/nullptr, loomc_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
}

TEST(VmLaunchConfigProgramTest, RejectsUnsupportedArtifactFormat) {
  uint8_t contents[] = {0};
  ByteSequencePtr sequence =
      CreateSequence(loomc_make_byte_span(contents, sizeof(contents)));
  const loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      /*.identifier=*/loomc_make_cstring_view("launch_config.loombc"),
      /*.contents=*/sequence.get(),
  };
  loomc_vm_launch_config_program_t* program =
      reinterpret_cast<loomc_vm_launch_config_program_t*>(UINTPTR_MAX);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_UNIMPLEMENTED,
                         loomc_vm_launch_config_program_load(
                             &artifact, loomc_allocator_system(), &program));
  EXPECT_EQ(program, nullptr);
}

TEST(VmLaunchConfigProgramTest, RejectsInvalidAllocator) {
  ByteSequencePtr sequence = CreateSequence(loomc_byte_span_empty());
  const loomc_artifact_t artifact = {
      /*.kind=*/LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG,
      /*.format=*/loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_VM_BYTECODE),
      /*.identifier=*/loomc_make_cstring_view("launch_config.vm"),
      /*.contents=*/sequence.get(),
  };
  loomc_vm_launch_config_program_t* program =
      reinterpret_cast<loomc_vm_launch_config_program_t*>(UINTPTR_MAX);
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_vm_launch_config_program_load(
                             &artifact, /*allocator=*/{}, &program));
  EXPECT_EQ(program, nullptr);
}

TEST(VmLaunchConfigProgramTest, RejectsLookupWithoutProgramAndClearsToken) {
  loomc_vm_launch_config_function_t function = {/*.value=*/0};
  LOOMC_EXPECT_STATUS_IS(
      LOOMC_STATUS_INVALID_ARGUMENT,
      loomc_vm_launch_config_program_lookup_function(
          /*program=*/nullptr, loomc_make_cstring_view("missing"), &function));
  EXPECT_FALSE(loomc_vm_launch_config_function_is_valid(function));
}

TEST(VmLaunchConfigProgramTest,
     RejectsOrdinalLookupWithoutProgramAndClearsToken) {
  loomc_vm_launch_config_function_t function = {/*.value=*/0};
  EXPECT_FALSE(loomc_vm_launch_config_program_function_at(
      /*program=*/nullptr, /*ordinal=*/0, &function));
  EXPECT_FALSE(loomc_vm_launch_config_function_is_valid(function));
}

TEST(VmLaunchConfigProgramTest, FailedInvocationLeavesOutputUntouched) {
  loomc_launch_config_t config = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      /*.structure_size=*/sizeof(config),
      /*.next=*/nullptr,
      /*.workgroup_count=*/{777, 778, 779},
  };
  LOOMC_EXPECT_STATUS_IS(LOOMC_STATUS_INVALID_ARGUMENT,
                         loomc_vm_launch_config_program_invoke(
                             /*program=*/nullptr, /*function=*/{0},
                             /*workload_argument_bits=*/nullptr,
                             /*workload_argument_count=*/0, &config));
  EXPECT_EQ(config.workgroup_count.x, 777u);
  EXPECT_EQ(config.workgroup_count.y, 778u);
  EXPECT_EQ(config.workgroup_count.z, 779u);
}

}  // namespace
