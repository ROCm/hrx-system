// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/diagnostics/wait_hazard.h"

#include <cstring>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/json_sink.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static void ExpectObjectValueEquals(iree_string_view_t object,
                                    iree_string_view_t key,
                                    iree_string_view_t expected) {
  EXPECT_TRUE(iree_string_view_equal(LookupObject(object, key), expected));
}

struct TextSinkState {
  loom_output_stream_t stream;
};

static iree_status_t FormatTextSink(void* user_data,
                                    const loom_diagnostic_t* diagnostic) {
  TextSinkState* state = static_cast<TextSinkState*>(user_data);
  return loom_diagnostic_format(diagnostic, &state->stream);
}

static loom_amdgpu_wait_hazard_t MakeWaitHazard(
    iree_string_view_t explanation) {
  return loom_amdgpu_wait_hazard_t{
      /*.diagnostic_code=*/IREE_SVL("wait_counter"),
      /*.has_kernel=*/true,
      /*.kernel_name=*/IREE_SVL("main"),
      /*.kernel_entry_offset=*/32,
      /*.counter_name=*/IREE_SVL("xcnt"),
      /*.access_kind=*/IREE_SVL("def"),
      /*.register_class=*/IREE_SVL("vgpr"),
      /*.register_index=*/16,
      /*.register_width=*/8,
      /*.section_name=*/IREE_SVL(".text"),
      /*.consumer_section_offset=*/144,
      /*.consumer_file_offset=*/400,
      /*.consumer_instruction=*/
      IREE_SVL("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23]"),
      /*.producer_section_offset=*/96,
      /*.producer_file_offset=*/352,
      /*.producer_instruction=*/
      IREE_SVL("buffer_load_dwordx4 v[16:19], off, s[0:3], 0"),
      /*.required_count=*/0,
      /*.explanation=*/explanation,
  };
}

TEST(AmdgpuWaitHazardTest, EmitsEveryStructuredFieldAsJson) {
  char explanation[] = "missing s_waitcnt_depctr";
  const loom_amdgpu_wait_hazard_t hazard =
      MakeWaitHazard(iree_make_cstring_view(explanation));

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_json_sink_options_t json_options = {
      /*.stream=*/&stream,
  };
  const loom_diagnostic_sink_t sink = {
      /*.fn=*/loom_diagnostic_json_sink,
      /*.user_data=*/&json_options,
  };
  IREE_ASSERT_OK(loom_amdgpu_wait_hazard_emit(&hazard, &sink));
  std::memset(explanation, 'x', sizeof(explanation) - 1);

  const iree_string_view_t root =
      ParseJsonDocument(iree_string_builder_view(&builder));
  ExpectObjectValueEquals(root, IREE_SV("error_id"), IREE_SV("ERR_AMDGPU_045"));
  ExpectObjectValueEquals(root, IREE_SV("emitter"), IREE_SV("pass"));
  const iree_string_view_t params = LookupObject(root, IREE_SV("params"));
  ExpectObjectValueEquals(params, IREE_SV("diagnostic_code"),
                          IREE_SV("wait_counter"));
  ExpectObjectValueEquals(params, IREE_SV("has_kernel"), IREE_SV("true"));
  ExpectObjectValueEquals(params, IREE_SV("kernel_name"), IREE_SV("main"));
  ExpectObjectValueEquals(params, IREE_SV("kernel_entry_offset"),
                          IREE_SV("32"));
  ExpectObjectValueEquals(params, IREE_SV("counter_name"), IREE_SV("xcnt"));
  ExpectObjectValueEquals(params, IREE_SV("access_kind"), IREE_SV("def"));
  ExpectObjectValueEquals(params, IREE_SV("register_class"), IREE_SV("vgpr"));
  ExpectObjectValueEquals(params, IREE_SV("register_index"), IREE_SV("16"));
  ExpectObjectValueEquals(params, IREE_SV("register_width"), IREE_SV("8"));
  ExpectObjectValueEquals(params, IREE_SV("section_name"), IREE_SV(".text"));
  ExpectObjectValueEquals(params, IREE_SV("consumer_section_offset"),
                          IREE_SV("144"));
  ExpectObjectValueEquals(params, IREE_SV("consumer_file_offset"),
                          IREE_SV("400"));
  ExpectObjectValueEquals(
      params, IREE_SV("consumer_instruction"),
      IREE_SV("v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], v[16:23]"));
  ExpectObjectValueEquals(params, IREE_SV("producer_section_offset"),
                          IREE_SV("96"));
  ExpectObjectValueEquals(params, IREE_SV("producer_file_offset"),
                          IREE_SV("352"));
  ExpectObjectValueEquals(
      params, IREE_SV("producer_instruction"),
      IREE_SV("buffer_load_dwordx4 v[16:19], off, s[0:3], 0"));
  ExpectObjectValueEquals(params, IREE_SV("required_count"), IREE_SV("0"));
  ExpectObjectValueEquals(params, IREE_SV("explanation"),
                          IREE_SV("missing s_waitcnt_depctr"));
  iree_string_builder_deinitialize(&builder);
}

TEST(AmdgpuWaitHazardTest, FormatsEveryStructuredFieldAsText) {
  const loom_amdgpu_wait_hazard_t hazard =
      MakeWaitHazard(IREE_SV("missing s_waitcnt_depctr"));
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  TextSinkState state;
  loom_output_stream_for_builder(&builder, &state.stream);
  const loom_diagnostic_sink_t sink = {
      /*.fn=*/FormatTextSink,
      /*.user_data=*/&state,
  };
  IREE_ASSERT_OK(loom_amdgpu_wait_hazard_emit(&hazard, &sink));
  const iree_string_view_t text = iree_string_builder_view(&builder);
  EXPECT_NE(
      iree_string_view_find(
          text,
          IREE_SV(
              "AMDGPU final-artifact wait hazard 'wait_counter' for kernel "
              "'main' (present=true, entry .text+32) at .text+144 (file+400): "
              "instruction 'v_wmma_f32_16x16x16_bf16 v[0:7], v[8:15], "
              "v[16:23]' has def conflict on vgpr[16:8] with producer at "
              "section+96 (file+352) instruction 'buffer_load_dwordx4 "
              "v[16:19], off, s[0:3], 0'; counter xcnt requires count 0: "
              "missing s_waitcnt_depctr"),
          0),
      IREE_STRING_VIEW_NPOS);
  iree_string_builder_deinitialize(&builder);
}

TEST(AmdgpuWaitHazardTest, RejectsMissingHazard) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_wait_hazard_emit(nullptr, nullptr));
}

}  // namespace
}  // namespace loom
