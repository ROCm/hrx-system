// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/reporting/format.h"

namespace loom {
namespace {

TEST(ConfigBindingsReportTest, RecordingAndMergingOwnStrings) {
  loom_target_compile_report_t source;
  loom_target_compile_report_initialize(iree_allocator_system(), &source);
  source.requested_detail_flags =
      LOOM_TARGET_COMPILE_REPORT_DETAIL_CONFIG_BINDING_ROWS;
  std::string key = "motif.left_depth";
  std::string value = "4";
  const loom_target_compile_report_config_binding_row_t row = {
      iree_make_string_view(key.data(), key.size()),
      iree_make_string_view(value.data(), value.size()),
  };
  IREE_ASSERT_OK(
      loom_target_compile_report_record_config_binding_row(&source, &row));
  key.assign(key.size(), 'x');
  value.assign(value.size(), '9');
  loom_target_compile_report_t merged;
  loom_target_compile_report_initialize(iree_allocator_system(), &merged);
  IREE_ASSERT_OK(
      loom_target_compile_report_record_entry_report(&merged, &source));
  loom_target_compile_report_deinitialize(&source);

  EXPECT_EQ(merged.config_binding_rows.count, 1u);
  for (auto mode : {LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_SUMMARY,
                    LOOM_TARGET_COMPILE_REPORT_FORMAT_MODE_DETAILS}) {
    const loom_target_compile_report_format_options_t options = {mode};
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_ASSERT_OK(
        loom_target_compile_report_format_json(&merged, &options, &stream));
    const std::string text(iree_string_builder_buffer(&builder),
                           iree_string_builder_size(&builder));
    EXPECT_NE(text.find("\"key\":\"motif.left_depth\",\"value\":\"4\""),
              std::string::npos)
        << text;
    iree_string_builder_deinitialize(&builder);
  }
  loom_target_compile_report_deinitialize(&merged);
}

}  // namespace
}  // namespace loom
