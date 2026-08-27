// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/printf_format.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static void AppendSlot(uint64_t value, std::vector<uint8_t>* arguments) {
  const iree_host_size_t old_size = arguments->size();
  arguments->resize(old_size + sizeof(value));
  memcpy(arguments->data() + old_size, &value, sizeof(value));
}

static void AppendDouble(double value, std::vector<uint8_t>* arguments) {
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  AppendSlot(bits, arguments);
}

static void AppendString(const char* value, std::vector<uint8_t>* arguments) {
  const iree_host_size_t length = strlen(value) + 1;
  const iree_host_size_t occupied_length = (length + 7u) & ~7u;
  const iree_host_size_t old_size = arguments->size();
  arguments->resize(old_size + occupied_length);
  memcpy(arguments->data() + old_size, value, length);
}

static iree_status_t Format(iree_string_view_t format,
                            const std::vector<uint8_t>& arguments,
                            std::string* output, int* out_count) {
  iree_string_builder_t builder;
  iree_string_builder_t scratch_builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_string_builder_initialize(iree_allocator_system(), &scratch_builder);
  iree_status_t status = iree_hip_printf_format(
      &builder, &scratch_builder, format, arguments.data(), arguments.size());
  if (iree_status_is_ok(status)) {
    const iree_string_view_t text = iree_string_builder_view(&builder);
    if (text.size == 0) {
      output->clear();
    } else {
      output->assign(text.data, text.size);
    }
    *out_count = (int)text.size;
  } else {
    EXPECT_EQ(0u, iree_string_builder_size(&builder));
    *out_count = -1;
  }
  iree_string_builder_deinitialize(&scratch_builder);
  iree_string_builder_deinitialize(&builder);
  return status;
}

TEST(PrintfFormatTest, FormatsDeviceSlotsWithHostVarargTypes) {
  std::vector<uint8_t> arguments;
  AppendSlot(42, &arguments);
  AppendSlot(UINT64_C(0x1234), &arguments);
  AppendSlot(3, &arguments);
  AppendString("abcdef", &arguments);
  AppendDouble(1.25, &arguments);
  AppendSlot(1, &arguments);  // Deliberately invalid if %n is dereferenced.

  std::string output;
  int output_count = 0;
  IREE_ASSERT_OK(
      Format(IREE_SV("value=%d hex=%#llx string=%.*s float=%.2f %% %nend"),
             arguments, &output, &output_count));
  EXPECT_EQ("value=42 hex=0x1234 string=abc float=1.25 % end", output);
  EXPECT_EQ(output.size(), output_count);
}

TEST(PrintfFormatTest, FormatsIntegerLengthModifiersWithExactVarargTypes) {
  std::vector<uint8_t> arguments;
  AppendSlot(UINT64_C(0xFF), &arguments);
  AppendSlot(UINT64_C(0xFFFF), &arguments);
  AppendSlot(UINT64_MAX - 1, &arguments);
  AppendSlot(UINT64_MAX, &arguments);
  AppendSlot(UINT64_MAX - 2, &arguments);
  AppendSlot(4, &arguments);
  AppendSlot(UINT64_MAX - 4, &arguments);
  AppendSlot(6, &arguments);

  std::string output;
  int output_count = 0;
  IREE_ASSERT_OK(Format(IREE_SV("%hhd %hu %ld %llu %jd %zu %td %tu"), arguments,
                        &output, &output_count));
  EXPECT_EQ("-1 65535 -2 18446744073709551615 -3 4 -5 6", output);
  EXPECT_EQ(output.size(), output_count);
}

TEST(PrintfFormatTest, RejectsUnsupportedOrMalformedGrammar) {
  static const iree_string_view_t cases[] = {
      IREE_SV("%"),
      IREE_SV("%Lf"),
      IREE_SV("%ls"),
      IREE_SV("%1$u"),
  };
  std::vector<uint8_t> arguments;
  AppendSlot(1, &arguments);
  for (iree_string_view_t format : cases) {
    std::string output;
    int output_count = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Format(format, arguments, &output, &output_count));
  }
}

TEST(PrintfFormatTest, RejectsTruncatedArguments) {
  std::string output;
  int output_count = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      Format(IREE_SV("%d"), std::vector<uint8_t>(7), &output, &output_count));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Format(IREE_SV("%s"), std::vector<uint8_t>{'a', '\0'},
                               &output, &output_count));
}

TEST(PrintfFormatTest, FormatsLargeFields) {
  std::vector<uint8_t> static_arguments;
  AppendSlot(1, &static_arguments);
  std::string output;
  int output_count = 0;
  IREE_ASSERT_OK(
      Format(IREE_SV("%1048577u"), static_arguments, &output, &output_count));
  EXPECT_EQ(1048577u, output.size());
  EXPECT_EQ('1', output.back());
  EXPECT_EQ(output.size(), output_count);

  std::vector<uint8_t> width_arguments;
  AppendSlot(1048577, &width_arguments);
  AppendSlot(1, &width_arguments);
  IREE_ASSERT_OK(
      Format(IREE_SV("%*d"), width_arguments, &output, &output_count));
  EXPECT_EQ(1048577u, output.size());
  EXPECT_EQ('1', output.back());
  EXPECT_EQ(output.size(), output_count);

  std::vector<uint8_t> precision_arguments;
  AppendSlot(1048577, &precision_arguments);
  AppendString("value", &precision_arguments);
  IREE_ASSERT_OK(
      Format(IREE_SV("%.*s"), precision_arguments, &output, &output_count));
  EXPECT_EQ("value", output);
  EXPECT_EQ(output.size(), output_count);
}

TEST(PrintfFormatTest, FormatsLongConversionSpecifier) {
  std::string format = "%";
  format.append(256, '0');
  format.append("1u");
  std::vector<uint8_t> arguments;
  AppendSlot(7, &arguments);

  std::string output;
  int output_count = 0;
  IREE_ASSERT_OK(Format(iree_make_string_view(format.data(), format.size()),
                        arguments, &output, &output_count));
  EXPECT_EQ("7", output);
  EXPECT_EQ(output.size(), output_count);
}

TEST(PrintfFormatTest, FormatsLargeEncodedInput) {
  iree_string_builder_t builder;
  iree_string_builder_t scratch_builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_string_builder_initialize(iree_allocator_system(), &scratch_builder);
  const std::string format(1024 * 1024 + 1, 'x');
  IREE_ASSERT_OK(iree_hip_printf_format(
      &builder, &scratch_builder,
      iree_make_string_view(format.data(), format.size()),
      /*arguments=*/nullptr, /*argument_length=*/0));
  EXPECT_EQ(format, std::string(iree_string_builder_buffer(&builder),
                                iree_string_builder_size(&builder)));
  iree_string_builder_deinitialize(&scratch_builder);
  iree_string_builder_deinitialize(&builder);
}

TEST(PrintfFormatTest, FormatsLargeAggregateOutput) {
  iree_string_builder_t builder;
  iree_string_builder_t scratch_builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_string_builder_initialize(iree_allocator_system(), &scratch_builder);
  std::vector<uint8_t> arguments;
  AppendSlot(1, &arguments);
  AppendSlot(2, &arguments);

  IREE_ASSERT_OK(iree_hip_printf_format(&builder, &scratch_builder,
                                        IREE_SV("%600000d%600000d"),
                                        arguments.data(), arguments.size()));
  EXPECT_EQ(1200000u, iree_string_builder_size(&builder));
  EXPECT_EQ('1', iree_string_builder_buffer(&builder)[599999]);
  EXPECT_EQ('2', iree_string_builder_buffer(&builder)[1199999]);
  iree_string_builder_deinitialize(&scratch_builder);
  iree_string_builder_deinitialize(&builder);
}

TEST(PrintfFormatTest, RejectsNullNonEmptyArgumentSpan) {
  iree_string_builder_t builder;
  iree_string_builder_t scratch_builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_string_builder_initialize(iree_allocator_system(), &scratch_builder);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_printf_format(&builder, &scratch_builder, IREE_SV("%d"),
                             /*arguments=*/nullptr, /*argument_length=*/8));
  EXPECT_EQ(0u, iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&scratch_builder);
  iree_string_builder_deinitialize(&builder);
}

TEST(PrintfFormatTest, ClearsPartialTextAfterLateFormatFailure) {
  iree_string_builder_t builder;
  iree_string_builder_t scratch_builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  iree_string_builder_initialize(iree_allocator_system(), &scratch_builder);
  std::vector<uint8_t> arguments;
  AppendSlot(42, &arguments);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hip_printf_format(&builder, &scratch_builder,
                             IREE_SV("value=%d trailing=%"), arguments.data(),
                             arguments.size()));
  EXPECT_EQ(0u, iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&scratch_builder);
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
