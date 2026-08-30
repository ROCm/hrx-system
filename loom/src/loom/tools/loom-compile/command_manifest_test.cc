// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-compile/command_manifest.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_status_t CreateByteSequence(iree_host_size_t length,
                                        iree_byte_sequence_t** out_sequence) {
  iree_byte_span_t span = iree_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      iree_allocator_system(), length, (void**)&span.data));
  span.data_length = length;
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &span, iree_allocator_system(), out_sequence);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), span.data);
  }
  return status;
}

TEST(CommandManifestTest, FormatsCanonicalFilenames) {
  char storage[64];
  iree_string_view_t filename = iree_string_view_empty();
  IREE_EXPECT_OK(loom_compile_command_manifest_format_program_filename(
      42, sizeof(storage), storage, &filename));
  EXPECT_TRUE(iree_string_view_equal(filename, IREE_SV("program-42.loomcmd")));

  IREE_EXPECT_OK(loom_compile_command_manifest_format_kernel_request_filename(
      7, sizeof(storage), storage, &filename));
  EXPECT_TRUE(iree_string_view_equal(filename, IREE_SV("kernel-7.loombc")));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_compile_command_manifest_format_program_filename(
                            42, 4, storage, &filename));
  EXPECT_TRUE(iree_string_view_is_empty(filename));
}

TEST(CommandManifestTest, WritesArtifactSetSchema) {
  iree_byte_sequence_t* program_data[2] = {NULL, NULL};
  IREE_ASSERT_OK(CreateByteSequence(7, &program_data[0]));
  IREE_ASSERT_OK(CreateByteSequence(11, &program_data[1]));
  const uint32_t first_requirements[] = {0};
  const uint32_t second_requirements[] = {1, 0};
  loom_cmd_program_artifact_t programs[] = {
      {
          /*.symbol=*/IREE_SV("alpha"),
          /*.data=*/program_data[0],
          /*.entry_requirement_indices=*/first_requirements,
          /*.entry_requirement_count=*/IREE_ARRAYSIZE(first_requirements),
      },
      {
          /*.symbol=*/IREE_SV("beta"),
          /*.data=*/program_data[1],
          /*.entry_requirement_indices=*/second_requirements,
          /*.entry_requirement_count=*/IREE_ARRAYSIZE(second_requirements),
      },
  };
  loom_cmd_program_artifact_entry_t entries[] = {
      {/*.symbol=*/IREE_SV("entry_a")},
      {/*.symbol=*/IREE_SV("entry_b")},
  };
  const loom_cmd_program_artifact_set_t artifact_set = {
      /*.programs=*/{/*.values=*/programs,
                     /*.count=*/IREE_ARRAYSIZE(programs)},
      /*.entries=*/
      {/*.values=*/entries,
       /*.count=*/IREE_ARRAYSIZE(entries)},
  };
  const uint32_t source_requirement_indices[] = {1};

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_OK(loom_compile_command_manifest_write(
      &artifact_set, source_requirement_indices,
      IREE_ARRAYSIZE(source_requirement_indices), &stream));
  EXPECT_TRUE(iree_string_view_equal(
      iree_string_builder_view(&builder),
      IREE_SV("{\"schema_version\":2,\"format\":\"loom-command-set\","
              "\"programs\":[{\"symbol\":\"alpha\",\"artifact\":"
              "\"program-0.loomcmd\",\"byte_length\":7,"
              "\"entry_requirements\":[0]},{\"symbol\":\"beta\","
              "\"artifact\":\"program-1.loomcmd\",\"byte_length\":11,"
              "\"entry_requirements\":[1,0]}],\"entries\":[{\"symbol\":"
              "\"entry_a\"},{\"symbol\":\"entry_b\",\"source_request\":"
              "\"kernel-1.loombc\"}]}")));
  iree_string_builder_deinitialize(&builder);
  iree_byte_sequence_release(program_data[0]);
  iree_byte_sequence_release(program_data[1]);
}

}  // namespace
}  // namespace loom
