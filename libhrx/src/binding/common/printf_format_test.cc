// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/printf_format.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(PrintfFormatTest, ParsesSupportedGrammar) {
  struct ParseCase {
    const char* spec;
    char conversion;
    uint8_t star_count;
  };
  static const ParseCase cases[] = {
      {/*.spec=*/"%d", /*.conversion=*/'d', /*.star_count=*/0},
      {/*.spec=*/"%#08llx", /*.conversion=*/'x', /*.star_count=*/0},
      {/*.spec=*/"%-*.*s", /*.conversion=*/'s', /*.star_count=*/2},
      {/*.spec=*/"%.*f", /*.conversion=*/'f', /*.star_count=*/1},
      {/*.spec=*/"%zn", /*.conversion=*/'n', /*.star_count=*/0},
  };
  for (const ParseCase& parse_case : cases) {
    SCOPED_TRACE(parse_case.spec);
    iree_hal_streaming_printf_spec_t parsed = {};
    ASSERT_TRUE(iree_hal_streaming_printf_parse_spec(
        parse_case.spec, strlen(parse_case.spec), &parsed));
    EXPECT_EQ(parse_case.conversion, parsed.conversion);
    EXPECT_EQ(parse_case.star_count, parsed.star_count);
  }
}

TEST(PrintfFormatTest, RejectsUnsupportedOrMalformedGrammar) {
  static const char* cases[] = {
      "d",           // Missing conversion introducer.
      "%",           // Missing conversion.
      "%Ld",         // Long double-sized argument slots are unsupported.
      "%ls",         // Wide strings are unsupported.
      "%1$u",        // Positional arguments are not part of the device ABI.
      "%1048577u",   // Literal width exceeds the output bound.
      "%.1048577u",  // Literal precision exceeds the output bound.
  };
  for (const char* parse_case : cases) {
    SCOPED_TRACE(parse_case);
    iree_hal_streaming_printf_spec_t parsed = {};
    EXPECT_FALSE(iree_hal_streaming_printf_parse_spec(
        parse_case, strlen(parse_case), &parsed));
  }
}

TEST(PrintfFormatTest, ParsesMetadataRecord) {
  uint64_t hash = 0;
  iree_string_view_t format = iree_string_view_empty();
  IREE_ASSERT_OK(iree_hal_streaming_printf_parse_metadata_record(
      IREE_SV("0:0:0123456789abcdef,value=%d"), &hash, &format));
  EXPECT_EQ(UINT64_C(0x0123456789ABCDEF), hash);
  EXPECT_TRUE(iree_string_view_equal(format, IREE_SV("value=%d")));
}

TEST(PrintfFormatTest, RejectsMalformedMetadataRecords) {
  static const iree_string_view_t cases[] = {
      IREE_SV("1:0:1,value"),
      IREE_SV("0:0:,value"),
      IREE_SV("0:0:not-hex,value"),
      IREE_SV("0:0:0123456789abcdef0,value"),
  };
  for (iree_string_view_t parse_case : cases) {
    uint64_t hash = 1;
    iree_string_view_t format = IREE_SV("unchanged");
    iree_status_t status = iree_hal_streaming_printf_parse_metadata_record(
        parse_case, &hash, &format);
    EXPECT_FALSE(iree_status_is_ok(status));
    iree_status_ignore(status);
    EXPECT_EQ(0u, hash);
    EXPECT_TRUE(iree_string_view_is_empty(format));
  }
}

static void AppendSlot(uint64_t value, std::vector<uint8_t>* arguments) {
  const iree_host_size_t old_size = arguments->size();
  arguments->resize(old_size + sizeof(value));
  memcpy(arguments->data() + old_size, &value, sizeof(value));
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
  FILE* stream = tmpfile();
  if (!stream) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "temporary output stream unavailable");
  }
  iree_status_t status = iree_hal_streaming_printf_format(
      stream, format, arguments.data(), arguments.size(), out_count);
  if (iree_status_is_ok(status)) {
    fflush(stream);
    const long output_length = ftell(stream);
    if (output_length < 0 || fseek(stream, 0, SEEK_SET) != 0) {
      status = iree_make_status(IREE_STATUS_UNKNOWN,
                                "failed to read temporary output stream");
    } else {
      output->resize((size_t)output_length);
      if (fread(output->data(), 1, output->size(), stream) != output->size()) {
        status = iree_make_status(IREE_STATUS_UNKNOWN,
                                  "failed to read formatted output");
      }
    }
  }
  fclose(stream);
  return status;
}

TEST(PrintfFormatTest, FormatsDeviceSlots) {
  std::vector<uint8_t> arguments;
  AppendSlot(42, &arguments);
  AppendSlot(3, &arguments);
  AppendString("abcdef", &arguments);
  AppendSlot(1, &arguments);  // Deliberately invalid if %n is dereferenced.

  std::string output;
  int output_count = 0;
  IREE_ASSERT_OK(Format(IREE_SV("value=%d string=%.*s %% %nend"), arguments,
                        &output, &output_count));
  EXPECT_EQ(output, "value=42 string=abc % end");
  EXPECT_EQ(output_count, output.size());
}

TEST(PrintfFormatTest, RejectsTruncatedArgumentSlot) {
  const std::vector<uint8_t> arguments(7);
  std::string output;
  int output_count = 0;
  iree_status_t status =
      Format(IREE_SV("%d"), arguments, &output, &output_count);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(PrintfFormatTest, RejectsTruncatedStringPadding) {
  const std::vector<uint8_t> arguments = {'a', '\0'};
  std::string output;
  int output_count = 0;
  iree_status_t status =
      Format(IREE_SV("%s"), arguments, &output, &output_count);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(0, output_count);
}

TEST(PrintfFormatTest, RejectsOversizedLiteralWidth) {
  std::vector<uint8_t> arguments;
  AppendSlot(1, &arguments);
  std::string output;
  int output_count = 0;
  iree_status_t status =
      Format(IREE_SV("%1048577d"), arguments, &output, &output_count);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(PrintfFormatTest, RejectsOversizedDynamicPrecision) {
  std::vector<uint8_t> arguments;
  AppendSlot(IREE_HAL_STREAMING_PRINTF_MAX_FIELD_LENGTH + 1, &arguments);
  AppendString("value", &arguments);
  std::string output;
  int output_count = 0;
  iree_status_t status =
      Format(IREE_SV("%.*s"), arguments, &output, &output_count);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
