// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for string parsing utilities: integer/float parsing, device size
// parsing with units, bitfield parsing, pattern matching, and hex byte parsing.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <charconv>

#include "iree/base/api.h"

// Sample bitfield mapping table for fuzzing iree_bitfield_parse.
// Uses realistic flag names similar to actual IREE usage.
static const iree_bitfield_string_mapping_t kTestBitfieldMappings[] = {
    {0x7, IREE_SVL("ALL")},        // Combined flag (A|B|C).
    {0x1, IREE_SVL("READ")},       // Bit 0.
    {0x2, IREE_SVL("WRITE")},      // Bit 1.
    {0x4, IREE_SVL("EXECUTE")},    // Bit 2.
    {0x8, IREE_SVL("DISCARD")},    // Bit 3.
    {0x10, IREE_SVL("MAPPABLE")},  // Bit 4.
    {0x20, IREE_SVL("COHERENT")},  // Bit 5.
};

template <typename Float>
static void iree_string_view_fuzz_check_hex_float_reference(
    const char* text, size_t text_length, const char* reference_text,
    size_t reference_text_length, Float actual, uint64_t sign_mask,
    uint64_t infinity_bits) {
  Float reference = 0.0;
  const std::from_chars_result result =
      std::from_chars(reference_text, reference_text + reference_text_length,
                      reference, std::chars_format::hex);
  if (result.ptr != reference_text + reference_text_length) {
    fprintf(stderr, "hex float reference consumed %td of %zu bytes for %.*s\n",
            result.ptr - reference_text, reference_text_length,
            (int)text_length, text);
    abort();
  }

  uint64_t actual_bits = 0;
  memcpy(&actual_bits, &actual, sizeof(actual));
  if (result.ec == std::errc()) {
    uint64_t reference_bits = 0;
    memcpy(&reference_bits, &reference, sizeof(reference));
    if (actual_bits != reference_bits) {
      fprintf(stderr,
              "hex float %zu-bit mismatch for %.*s: actual=0x%016llx "
              "reference=0x%016llx\n",
              sizeof(Float) * 8, (int)text_length, text,
              (unsigned long long)actual_bits,
              (unsigned long long)reference_bits);
      abort();
    }
    return;
  }

  // std::from_chars leaves the output unchanged when the exact value is out
  // of range. Valid hexadecimal inputs can only round to signed zero or
  // infinity in that case.
  if (result.ec == std::errc::result_out_of_range) {
    const bool expected_negative = reference_text[0] == '-';
    const bool actual_negative = (actual_bits & sign_mask) != 0;
    const uint64_t magnitude_bits = actual_bits & ~sign_mask;
    if (actual_negative == expected_negative &&
        (magnitude_bits == 0 || magnitude_bits == infinity_bits)) {
      return;
    }
  }

  fprintf(stderr, "hex float reference rejected %.*s: error=%d\n",
          (int)text_length, text, (int)result.ec);
  abort();
}

static void iree_string_view_fuzz_structured_hex_float(const uint8_t* data,
                                                       size_t size) {
  if (size < 4) {
    return;
  }

  static const char kHexDigits[] = "0123456789abcdef";
  const size_t significand_byte_count = size - 3 < 32 ? size - 3 : 32;
  const size_t significand_nibble_count = significand_byte_count * 2;
  const size_t point_position = data[0] % (significand_nibble_count + 1);

  char text[96];
  size_t text_length = 0;
  if ((data[0] & 0x80) != 0) {
    text[text_length++] = '-';
  }
  text[text_length++] = '0';
  text[text_length++] = (data[0] & 0x40) != 0 ? 'X' : 'x';
  for (size_t i = 0; i <= significand_nibble_count; ++i) {
    if (i == point_position) {
      text[text_length++] = '.';
    }
    if (i == significand_nibble_count) {
      break;
    }
    const uint8_t byte = data[3 + i / 2];
    const uint8_t nibble =
        (i & 1) == 0 ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
    text[text_length++] = kHexDigits[nibble];
  }
  text[text_length++] = (data[0] & 0x20) != 0 ? 'P' : 'p';
  const int16_t exponent_bits = (int16_t)((uint16_t)data[1] << 8 | data[2]);
  const int exponent_length = snprintf(
      text + text_length, sizeof(text) - text_length, "%d", (int)exponent_bits);
  if (exponent_length <= 0 ||
      (size_t)exponent_length >= sizeof(text) - text_length) {
    return;
  }
  text_length += (size_t)exponent_length;

  const iree_string_view_t input =
      iree_make_string_view(text, (iree_host_size_t)text_length);
  float actual_f32 = 0.0f;
  double actual_f64 = 0.0;
  if (!iree_string_view_atof(input, &actual_f32) ||
      !iree_string_view_atod(input, &actual_f64)) {
    abort();
  }

  // std::from_chars intentionally omits the 0x prefix for hexadecimal input.
  // Keep the sign, remove the prefix, and use its locale-independent parser as
  // an oracle instead of inheriting platform libc subnormal-rounding defects.
  char reference_text[sizeof(text)];
  const size_t sign_length = text[0] == '-' ? 1 : 0;
  memcpy(reference_text, text, sign_length);
  memcpy(reference_text + sign_length, text + sign_length + 2,
         text_length - sign_length - 2);
  const size_t reference_text_length = text_length - 2;
  iree_string_view_fuzz_check_hex_float_reference(
      text, text_length, reference_text, reference_text_length, actual_f32,
      UINT64_C(0x80000000), UINT64_C(0x7F800000));
  iree_string_view_fuzz_check_hex_float_reference(
      text, text_length, reference_text, reference_text_length, actual_f64,
      UINT64_C(0x8000000000000000), UINT64_C(0x7FF0000000000000));
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  //===--------------------------------------------------------------------===//
  // Integer parsing (signed and unsigned, various bases)
  //===--------------------------------------------------------------------===//

  {
    int32_t value_i32 = 0;
    (void)iree_string_view_atoi_int32(input, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 10, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 16, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 2, &value_i32);
  }

  {
    uint32_t value_u32 = 0;
    (void)iree_string_view_atoi_uint32(input, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 10, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 16, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 2, &value_u32);
  }

  {
    int64_t value_i64 = 0;
    (void)iree_string_view_atoi_int64(input, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 10, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 16, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 2, &value_i64);
  }

  {
    uint64_t value_u64 = 0;
    (void)iree_string_view_atoi_uint64(input, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 10, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 16, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 2, &value_u64);
  }

  //===--------------------------------------------------------------------===//
  // Floating point parsing
  //===--------------------------------------------------------------------===//

  iree_string_view_fuzz_structured_hex_float(data, size);

  {
    float value_f32 = 0.0f;
    (void)iree_string_view_atof(input, &value_f32);
  }

  {
    double value_f64 = 0.0;
    (void)iree_string_view_atod(input, &value_f64);
  }

  //===--------------------------------------------------------------------===//
  // Device size parsing with units (e.g., "1kb", "2mib", "3gb")
  //===--------------------------------------------------------------------===//

  {
    iree_device_size_t device_size = 0;
    iree_status_t status =
        iree_string_view_parse_device_size(input, &device_size);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Bitfield parsing
  //===--------------------------------------------------------------------===//

  {
    uint32_t bitfield_value = 0;
    iree_status_t status =
        iree_bitfield_parse(input, IREE_ARRAYSIZE(kTestBitfieldMappings),
                            kTestBitfieldMappings, &bitfield_value);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Pattern matching (wildcard patterns with * and ?)
  //===--------------------------------------------------------------------===//

  // Use the first half as value and second half as pattern.
  if (size >= 2) {
    size_t mid = size / 2;
    iree_string_view_t value =
        iree_make_string_view(reinterpret_cast<const char*>(data), mid);
    iree_string_view_t pattern = iree_make_string_view(
        reinterpret_cast<const char*>(data + mid), size - mid);
    (void)iree_string_view_match_pattern(value, pattern);
  }

  // Also test pattern matching with specific patterns that stress recursion.
  (void)iree_string_view_match_pattern(input, IREE_SV("*"));
  (void)iree_string_view_match_pattern(input, IREE_SV("?*?"));
  (void)iree_string_view_match_pattern(input, IREE_SV("***"));

  //===--------------------------------------------------------------------===//
  // Hex byte parsing
  //===--------------------------------------------------------------------===//

  // Parse up to 64 bytes of hex data.
  {
    uint8_t hex_buffer[64] = {0};
    (void)iree_string_view_parse_hex_bytes(input, sizeof(hex_buffer),
                                           hex_buffer);
  }

  // Try parsing various sizes to test boundary conditions.
  for (size_t parse_size = 1; parse_size <= 8; ++parse_size) {
    uint8_t small_buffer[8] = {0};
    (void)iree_string_view_parse_hex_bytes(input, parse_size, small_buffer);
  }

  //===--------------------------------------------------------------------===//
  // String view operations that process the data
  //===--------------------------------------------------------------------===//

  (void)iree_string_view_trim(input);

  // Split operations with various split characters.
  {
    iree_string_view_t lhs, rhs;
    (void)iree_string_view_split(input, '|', &lhs, &rhs);
    (void)iree_string_view_split(input, '=', &lhs, &rhs);
    (void)iree_string_view_split(input, ',', &lhs, &rhs);
    (void)iree_string_view_split(input, ':', &lhs, &rhs);
  }

  // Find operations.
  if (size > 0) {
    char search_char = static_cast<char>(data[0]);
    (void)iree_string_view_find_char(input, search_char, 0);

    if (size > 1) {
      iree_string_view_t search_set =
          iree_make_string_view(reinterpret_cast<const char*>(data), size / 2);
      (void)iree_string_view_find_first_of(input, search_set, 0);
      (void)iree_string_view_find_last_of(input, search_set, SIZE_MAX);
    }
  }

  // Comparison operations.
  if (size >= 2) {
    size_t mid = size / 2;
    iree_string_view_t left =
        iree_make_string_view(reinterpret_cast<const char*>(data), mid);
    iree_string_view_t right = iree_make_string_view(
        reinterpret_cast<const char*>(data + mid), size - mid);
    (void)iree_string_view_equal(left, right);
    (void)iree_string_view_equal_case(left, right);
    (void)iree_string_view_compare(left, right);
    (void)iree_string_view_starts_with(left, right);
    (void)iree_string_view_ends_with(left, right);
  }

  return 0;
}
