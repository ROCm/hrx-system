// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/string_view.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"

static_assert(FLT_RADIX == 2 && sizeof(float) == sizeof(uint32_t) &&
                  FLT_MANT_DIG == 24 && FLT_MIN_EXP == -125 &&
                  FLT_MAX_EXP == 128,
              "iree_string_view_atof requires IEEE-754 binary32");
static_assert(FLT_RADIX == 2 && sizeof(double) == sizeof(uint64_t) &&
                  DBL_MANT_DIG == 53 && DBL_MIN_EXP == -1021 &&
                  DBL_MAX_EXP == 1024,
              "iree_string_view_atod requires IEEE-754 binary64");

static inline size_t iree_min_host_size(iree_host_size_t a,
                                        iree_host_size_t b) {
  return a < b ? a : b;
}

// Here to ensure that we don't pull in locale-specific code:
static inline bool iree_isupper(char c) { return (unsigned)c - 'A' < 26; }
static inline char iree_tolower(char c) {
  return iree_isupper(c) ? (c | 32) : c;
}

IREE_API_EXPORT bool iree_string_view_equal(iree_string_view_t lhs,
                                            iree_string_view_t rhs) {
  if (lhs.size != rhs.size) return false;
  if (lhs.size == 0) return true;  // Both empty - equal without memcmp.
  return memcmp(lhs.data, rhs.data, lhs.size) == 0;
}

IREE_API_EXPORT bool iree_string_view_equal_case(iree_string_view_t lhs,
                                                 iree_string_view_t rhs) {
  if (lhs.size != rhs.size) return false;
  for (iree_host_size_t i = 0; i < lhs.size; ++i) {
    if (iree_tolower(lhs.data[i]) != iree_tolower(rhs.data[i])) return false;
  }
  return true;
}

IREE_API_EXPORT int iree_string_view_compare(iree_string_view_t lhs,
                                             iree_string_view_t rhs) {
  iree_host_size_t min_size = iree_min_host_size(lhs.size, rhs.size);
  if (min_size == 0) {
    // Both empty, or one is a prefix of the other starting from empty.
    if (lhs.size == rhs.size) return 0;
    return lhs.size < rhs.size ? -1 : 1;
  }
  int cmp = strncmp(lhs.data, rhs.data, min_size);
  if (cmp != 0) {
    return cmp;
  } else if (lhs.size == rhs.size) {
    return 0;
  }
  return lhs.size < rhs.size ? -1 : 1;
}

IREE_API_EXPORT iree_host_size_t iree_string_view_find_char(
    iree_string_view_t value, char c, iree_host_size_t pos) {
  if (iree_string_view_is_empty(value) || pos >= value.size) {
    return IREE_STRING_VIEW_NPOS;
  }
  const char* result =
      (const char*)(memchr(value.data + pos, c, value.size - pos));
  return result != NULL ? result - value.data : IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_host_size_t iree_string_view_find(
    iree_string_view_t value, iree_string_view_t needle, iree_host_size_t pos) {
  if (needle.size == 0) return pos <= value.size ? pos : IREE_STRING_VIEW_NPOS;
  if (needle.size > value.size) return IREE_STRING_VIEW_NPOS;
  // Safe: needle.size <= value.size, so subtraction cannot underflow.
  if (pos > value.size - needle.size) return IREE_STRING_VIEW_NPOS;
  // Single character: delegate to memchr.
  if (needle.size == 1) {
    return iree_string_view_find_char(value, needle.data[0], pos);
  }
  // Brute-force scan. Fine for the string sizes we deal with.
  iree_host_size_t limit = value.size - needle.size;
  for (iree_host_size_t i = pos; i <= limit; ++i) {
    if (memcmp(value.data + i, needle.data, needle.size) == 0) {
      return i;
    }
  }
  return IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_host_size_t iree_string_view_find_first_of(
    iree_string_view_t value, iree_string_view_t s, iree_host_size_t pos) {
  if (iree_string_view_is_empty(value) || iree_string_view_is_empty(s)) {
    return IREE_STRING_VIEW_NPOS;
  }
  if (s.size == 1) {
    // Avoid the cost of the lookup table for a single-character search.
    return iree_string_view_find_char(value, s.data[0], pos);
  }
  bool lookup_table[UCHAR_MAX + 1] = {0};
  for (iree_host_size_t i = 0; i < s.size; ++i) {
    lookup_table[(uint8_t)s.data[i]] = true;
  }
  for (iree_host_size_t i = pos; i < value.size; ++i) {
    if (lookup_table[(uint8_t)value.data[i]]) {
      return i;
    }
  }
  return IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT iree_host_size_t iree_string_view_find_last_of(
    iree_string_view_t value, iree_string_view_t s, iree_host_size_t pos) {
  if (iree_string_view_is_empty(value) || iree_string_view_is_empty(s)) {
    return IREE_STRING_VIEW_NPOS;
  }
  bool lookup_table[UCHAR_MAX + 1] = {0};
  for (iree_host_size_t i = 0; i < s.size; ++i) {
    lookup_table[(uint8_t)s.data[i]] = true;
  }
  pos = iree_min(pos, value.size - 1) + 1;
  iree_host_size_t i = pos;
  while (i != 0) {
    --i;
    if (lookup_table[(uint8_t)value.data[i]]) {
      return i;
    }
  }
  return IREE_STRING_VIEW_NPOS;
}

IREE_API_EXPORT bool iree_string_view_starts_with(iree_string_view_t value,
                                                  iree_string_view_t prefix) {
  if (!value.data || !prefix.data || !prefix.size || prefix.size > value.size) {
    return false;
  }
  return strncmp(value.data, prefix.data, prefix.size) == 0;
}

IREE_API_EXPORT bool iree_string_view_ends_with(iree_string_view_t value,
                                                iree_string_view_t suffix) {
  if (!value.data || !suffix.data || !suffix.size || suffix.size > value.size) {
    return false;
  }
  return strncmp(value.data + value.size - suffix.size, suffix.data,
                 suffix.size) == 0;
}

IREE_API_EXPORT iree_string_view_t
iree_string_view_remove_prefix(iree_string_view_t value, iree_host_size_t n) {
  if (n >= value.size) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(value.data + n, value.size - n);
}

IREE_API_EXPORT iree_string_view_t
iree_string_view_remove_suffix(iree_string_view_t value, iree_host_size_t n) {
  if (n >= value.size) {
    return iree_string_view_empty();
  }
  return iree_make_string_view(value.data, value.size - n);
}

IREE_API_EXPORT iree_string_view_t iree_string_view_strip_prefix(
    iree_string_view_t value, iree_string_view_t prefix) {
  if (iree_string_view_starts_with(value, prefix)) {
    return iree_string_view_remove_prefix(value, prefix.size);
  }
  return value;
}

IREE_API_EXPORT iree_string_view_t iree_string_view_strip_suffix(
    iree_string_view_t value, iree_string_view_t suffix) {
  if (iree_string_view_ends_with(value, suffix)) {
    return iree_string_view_remove_suffix(value, suffix.size);
  }
  return value;
}

IREE_API_EXPORT bool iree_string_view_consume_prefix(
    iree_string_view_t* value, iree_string_view_t prefix) {
  if (iree_string_view_starts_with(*value, prefix)) {
    *value = iree_string_view_remove_prefix(*value, prefix.size);
    return true;
  }
  return false;
}

IREE_API_EXPORT bool iree_string_view_consume_suffix(
    iree_string_view_t* value, iree_string_view_t suffix) {
  if (iree_string_view_ends_with(*value, suffix)) {
    *value = iree_string_view_remove_suffix(*value, suffix.size);
    return true;
  }
  return false;
}

IREE_API_EXPORT iree_string_view_t
iree_string_view_trim(iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) return value;
  iree_host_size_t start = 0;
  while (start < value.size && isspace((unsigned char)value.data[start])) {
    ++start;
  }
  iree_host_size_t end = value.size;
  while (end > start && isspace((unsigned char)value.data[end - 1])) {
    --end;
  }
  return iree_make_string_view(value.data + start, end - start);
}

IREE_API_EXPORT iree_string_view_t iree_string_view_substr(
    iree_string_view_t value, iree_host_size_t pos, iree_host_size_t n) {
  pos = iree_min_host_size(pos, value.size);
  n = iree_min_host_size(n, value.size - pos);
  return iree_make_string_view(value.data + pos, n);
}

IREE_API_EXPORT intptr_t iree_string_view_split(iree_string_view_t value,
                                                char split_char,
                                                iree_string_view_t* out_lhs,
                                                iree_string_view_t* out_rhs) {
  if (out_lhs) *out_lhs = iree_string_view_empty();
  if (out_rhs) *out_rhs = iree_string_view_empty();
  if (!value.data || !value.size) {
    return -1;
  }
  const void* first_ptr = memchr(value.data, split_char, value.size);
  if (!first_ptr) {
    if (out_lhs) *out_lhs = value;
    return -1;
  }
  intptr_t offset = (intptr_t)((const char*)(first_ptr)-value.data);
  if (out_lhs) {
    out_lhs->data = value.data;
    out_lhs->size = offset;
  }
  if (out_rhs) {
    out_rhs->data = value.data + offset + 1;
    out_rhs->size = value.size - offset - 1;
  }
  return offset;
}

IREE_API_EXPORT void iree_string_view_replace_char(iree_string_view_t value,
                                                   char old_char,
                                                   char new_char) {
  char* p = (char*)value.data;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    if (p[i] == old_char) p[i] = new_char;
  }
}

IREE_API_EXPORT bool iree_string_view_match_pattern(
    iree_string_view_t value, iree_string_view_t pattern) {
  iree_host_size_t value_position = 0;
  iree_host_size_t pattern_position = 0;
  iree_host_size_t star_pattern_position = IREE_STRING_VIEW_NPOS;
  iree_host_size_t star_value_position = 0;
  while (value_position < value.size) {
    if (pattern_position < pattern.size &&
        pattern.data[pattern_position] == '*') {
      // A later star subsumes backtracking to any earlier star. Remember where
      // this star begins matching so a later mismatch can extend it by one
      // byte and retry the remaining pattern.
      star_pattern_position = pattern_position++;
      star_value_position = value_position;
      continue;
    }
    if (pattern_position < pattern.size &&
        (pattern.data[pattern_position] == '?' ||
         pattern.data[pattern_position] == value.data[value_position])) {
      ++pattern_position;
      ++value_position;
      continue;
    }
    if (star_pattern_position == IREE_STRING_VIEW_NPOS) return false;
    pattern_position = star_pattern_position + 1;
    value_position = ++star_value_position;
  }

  // Only stars can match the empty suffix after consuming the value.
  while (pattern_position < pattern.size &&
         pattern.data[pattern_position] == '*') {
    ++pattern_position;
  }
  return pattern_position == pattern.size;
}

IREE_API_EXPORT void iree_string_view_to_cstring(
    iree_string_view_t value, char* buffer, iree_host_size_t buffer_length) {
  if (!buffer_length) return;
  // Truncate and ensure there's space for the NUL terminator.
  iree_host_size_t length = iree_min(value.size, buffer_length - 1);
  // Copy string contents up to the truncated length.
  // Guard against NULL data pointer (empty string view).
  if (length > 0) {
    memcpy(buffer, value.data, length);
  }
  // Add NUL terminator.
  buffer[length] = 0;
}

IREE_API_EXPORT iree_host_size_t iree_string_view_append_to_buffer(
    iree_string_view_t source_value, iree_string_view_t* target_value,
    char* buffer) {
  // Do not copy zero-sized values to avoid passing NULLs to memcpy. The source
  // and destination pointers are required to be valid and non-NULL by the C
  // standard.
  if (source_value.size > 0) {
    memcpy(buffer, source_value.data, source_value.size);
  }
  target_value->data = buffer;
  target_value->size = source_value.size;
  return source_value.size;
}

// NOTE: these implementations aren't great due to the enforced memcpy we
// perform. These _should_ never be on a hot path, though, so this keeps our
// code size small.

IREE_API_EXPORT bool iree_string_view_atoi_int32_base(iree_string_view_t value,
                                                      int base,
                                                      int32_t* out_value) {
  // Copy to scratch memory with a NUL terminator.
  char temp[16] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  long parsed_value = strtol(temp, &end, base);
  if (temp == end) return false;
  if ((parsed_value == LONG_MIN || parsed_value == LONG_MAX) &&
      errno == ERANGE) {
    return false;
  }
  *out_value = (int32_t)parsed_value;
  return parsed_value != 0 || errno == 0;
}

IREE_API_EXPORT bool iree_string_view_atoi_int32(iree_string_view_t value,
                                                 int32_t* out_value) {
  return iree_string_view_atoi_int32_base(value, /*base=*/0, out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_uint32_base(iree_string_view_t value,
                                                       int base,
                                                       uint32_t* out_value) {
  // Copy to scratch memory with a NUL terminator.
  char temp[16] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  unsigned long parsed_value = strtoul(temp, &end, base);
  if (temp == end) return false;
  if (parsed_value == ULONG_MAX && errno == ERANGE) return false;
  *out_value = (uint32_t)parsed_value;
  return parsed_value != 0 || errno == 0;
}

IREE_API_EXPORT bool iree_string_view_atoi_uint32(iree_string_view_t value,
                                                  uint32_t* out_value) {
  return iree_string_view_atoi_uint32_base(value, /*base=*/0, out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_int64_base(iree_string_view_t value,
                                                      int base,
                                                      int64_t* out_value) {
  // Copy to scratch memory with a NUL terminator.
  char temp[32] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  long long parsed_value = strtoll(temp, &end, base);
  if (temp == end) return false;
  if ((parsed_value == LLONG_MIN || parsed_value == LLONG_MAX) &&
      errno == ERANGE) {
    return false;
  }
  *out_value = (int64_t)parsed_value;
  return parsed_value != 0 || errno == 0;
}

IREE_API_EXPORT bool iree_string_view_atoi_int64(iree_string_view_t value,
                                                 int64_t* out_value) {
  return iree_string_view_atoi_int64_base(value, /*base=*/0, out_value);
}

IREE_API_EXPORT bool iree_string_view_atoi_uint64_base(iree_string_view_t value,
                                                       int base,
                                                       uint64_t* out_value) {
  // Copy to scratch memory with a NUL terminator.
  char temp[32] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  unsigned long long parsed_value = strtoull(temp, &end, base);
  if (temp == end) return false;
  if (parsed_value == ULLONG_MAX && errno == ERANGE) return false;
  *out_value = (uint64_t)parsed_value;
  return parsed_value != 0 || errno == 0;
}

IREE_API_EXPORT bool iree_string_view_atoi_uint64(iree_string_view_t value,
                                                  uint64_t* out_value) {
  return iree_string_view_atoi_uint64_base(value, /*base=*/0, out_value);
}

typedef struct iree_hex_float_t {
  // Sign bit parsed from the source spelling.
  bool is_negative;
  // First up to 64 bits of the integer significand, most significant first.
  uint64_t prefix_bits;
  // Number of valid bits in |prefix_bits|.
  uint32_t prefix_bit_count;
  // Bit width of the full integer significand after leading zero removal.
  int64_t total_bit_count;
  // Power of two applied to the integer significand.
  int64_t binary_exponent;
  // Whether a set bit exists after |prefix_bits|.
  bool has_trailing_one;
} iree_hex_float_t;

static int64_t iree_saturating_add_int64(int64_t lhs, int64_t rhs) {
  if (rhs > 0 && lhs > INT64_MAX - rhs) return INT64_MAX;
  if (rhs < 0 && lhs < INT64_MIN - rhs) return INT64_MIN;
  return lhs + rhs;
}

static int64_t iree_saturating_sub_int64(int64_t lhs, int64_t rhs) {
  if (rhs == INT64_MIN) {
    return lhs >= 0 ? INT64_MAX : INT64_MAX + lhs + 1;
  }
  return iree_saturating_add_int64(lhs, -rhs);
}

static int iree_hex_digit_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  c = iree_tolower(c);
  return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}

static void iree_hex_float_append_significand_bits(iree_hex_float_t* parsed,
                                                   uint32_t bits,
                                                   uint32_t bit_count) {
  parsed->total_bit_count =
      iree_saturating_add_int64(parsed->total_bit_count, (int64_t)bit_count);
  if (parsed->prefix_bit_count == 64) {
    parsed->has_trailing_one |= bits != 0;
    return;
  }
  const uint32_t available_bit_count = 64 - parsed->prefix_bit_count;
  const uint32_t appended_bit_count =
      bit_count < available_bit_count ? bit_count : available_bit_count;
  parsed->prefix_bits = (parsed->prefix_bits << appended_bit_count) |
                        (bits >> (bit_count - appended_bit_count));
  parsed->prefix_bit_count += appended_bit_count;
  if (appended_bit_count < bit_count) {
    const uint32_t remaining_bit_count = bit_count - appended_bit_count;
    parsed->has_trailing_one |=
        (bits & ((UINT32_C(1) << remaining_bit_count) - 1)) != 0;
  }
}

static bool iree_string_view_parse_hex_float(iree_string_view_t value,
                                             iree_hex_float_t* out_parsed) {
  memset(out_parsed, 0, sizeof(*out_parsed));
  if (iree_string_view_is_empty(value) || value.size > INT64_MAX / 4) {
    return false;
  }

  iree_host_size_t position = 0;
  if (value.data[position] == '+' || value.data[position] == '-') {
    out_parsed->is_negative = value.data[position] == '-';
    if (++position == value.size) return false;
  }
  if (position + 2 > value.size || value.data[position] != '0' ||
      iree_tolower(value.data[position + 1]) != 'x') {
    return false;
  }
  position += 2;

  bool saw_digit = false;
  bool saw_nonzero_digit = false;
  bool saw_point = false;
  int64_t fractional_nibble_count = 0;
  while (position < value.size) {
    const char c = value.data[position];
    if (c == '.' && !saw_point) {
      saw_point = true;
      ++position;
      continue;
    }
    const int digit = iree_hex_digit_value(c);
    if (digit < 0) break;
    saw_digit = true;
    if (saw_point) ++fractional_nibble_count;
    if (!saw_nonzero_digit) {
      if (digit == 0) {
        ++position;
        continue;
      }
      saw_nonzero_digit = true;
      uint32_t first_bit_count = 4;
      while ((digit & (1 << (first_bit_count - 1))) == 0) {
        --first_bit_count;
      }
      iree_hex_float_append_significand_bits(out_parsed, (uint32_t)digit,
                                             first_bit_count);
    } else {
      iree_hex_float_append_significand_bits(out_parsed, (uint32_t)digit, 4);
    }
    ++position;
  }
  if (!saw_digit || position == value.size ||
      (value.data[position] != 'p' && value.data[position] != 'P')) {
    return false;
  }
  ++position;

  bool exponent_is_negative = false;
  if (position < value.size &&
      (value.data[position] == '+' || value.data[position] == '-')) {
    exponent_is_negative = value.data[position] == '-';
    ++position;
  }
  if (position == value.size || value.data[position] < '0' ||
      value.data[position] > '9') {
    return false;
  }
  int64_t explicit_exponent = 0;
  while (position < value.size && value.data[position] >= '0' &&
         value.data[position] <= '9') {
    const int64_t digit = value.data[position] - '0';
    if (explicit_exponent > (INT64_MAX - digit) / 10) {
      explicit_exponent = INT64_MAX;
    } else {
      explicit_exponent = explicit_exponent * 10 + digit;
    }
    ++position;
  }
  if (position != value.size) return false;
  if (exponent_is_negative) {
    explicit_exponent =
        explicit_exponent == INT64_MAX ? INT64_MIN : -explicit_exponent;
  }
  out_parsed->binary_exponent =
      iree_saturating_sub_int64(explicit_exponent, fractional_nibble_count * 4);
  return true;
}

static uint64_t iree_hex_float_prefix(const iree_hex_float_t* parsed,
                                      uint32_t bit_count) {
  if (bit_count == 0) return 0;
  return parsed->prefix_bits >> (parsed->prefix_bit_count - bit_count);
}

static bool iree_hex_float_bit(const iree_hex_float_t* parsed,
                               uint32_t bit_position) {
  return ((parsed->prefix_bits >>
           (parsed->prefix_bit_count - bit_position - 1)) &
          1) != 0;
}

static bool iree_hex_float_has_one_after(const iree_hex_float_t* parsed,
                                         uint32_t bit_position) {
  const uint32_t remaining_prefix_bit_count =
      parsed->prefix_bit_count - bit_position - 1;
  if (remaining_prefix_bit_count != 0) {
    const uint64_t mask = (UINT64_C(1) << remaining_prefix_bit_count) - 1;
    if ((parsed->prefix_bits & mask) != 0) return true;
  }
  return parsed->has_trailing_one;
}

// Rounds the integer significand divided by 2^|right_shift| to nearest even.
// The caller guarantees that the rounded result fits in 54 bits.
static uint64_t iree_hex_float_round_right(const iree_hex_float_t* parsed,
                                           int64_t right_shift) {
  if (right_shift <= 0) {
    const uint32_t left_shift = (uint32_t)-right_shift;
    return parsed->prefix_bits << left_shift;
  }
  if (right_shift > parsed->total_bit_count) return 0;

  const uint32_t quotient_bit_count =
      (uint32_t)(parsed->total_bit_count - right_shift);
  uint64_t quotient = iree_hex_float_prefix(parsed, quotient_bit_count);
  const bool round_bit = iree_hex_float_bit(parsed, quotient_bit_count);
  const bool sticky = iree_hex_float_has_one_after(parsed, quotient_bit_count);
  if (round_bit && (sticky || (quotient & 1) != 0)) ++quotient;
  return quotient;
}

static uint64_t iree_hex_float_to_ieee_bits(const iree_hex_float_t* parsed,
                                            uint32_t precision,
                                            int32_t minimum_exponent,
                                            int32_t maximum_exponent,
                                            int32_t exponent_bias,
                                            uint32_t sign_bit_position) {
  const uint64_t sign =
      parsed->is_negative ? UINT64_C(1) << sign_bit_position : 0;
  if (parsed->total_bit_count == 0) return sign;

  int64_t exponent = iree_saturating_add_int64(parsed->binary_exponent,
                                               parsed->total_bit_count - 1);
  const uint64_t infinity_exponent =
      (uint64_t)(maximum_exponent + exponent_bias + 1);
  if (exponent > maximum_exponent) {
    return sign | (infinity_exponent << (precision - 1));
  }

  if (exponent >= minimum_exponent) {
    uint64_t significand =
        iree_hex_float_round_right(parsed, parsed->total_bit_count - precision);
    if (significand == (UINT64_C(1) << precision)) {
      significand >>= 1;
      if (++exponent > maximum_exponent) {
        return sign | (infinity_exponent << (precision - 1));
      }
    }
    const uint64_t fraction_mask = (UINT64_C(1) << (precision - 1)) - 1;
    return sign | ((uint64_t)(exponent + exponent_bias) << (precision - 1)) |
           (significand & fraction_mask);
  }

  const int64_t subnormal_exponent =
      (int64_t)minimum_exponent - ((int64_t)precision - 1);
  const int64_t right_shift =
      iree_saturating_sub_int64(subnormal_exponent, parsed->binary_exponent);
  const uint64_t fraction = iree_hex_float_round_right(parsed, right_shift);
  if (fraction == (UINT64_C(1) << (precision - 1))) {
    return sign | (UINT64_C(1) << (precision - 1));
  }
  return sign | fraction;
}

static bool iree_string_view_is_hex_float(iree_string_view_t value) {
  iree_host_size_t position = 0;
  if (value.size != 0 &&
      (value.data[position] == '+' || value.data[position] == '-')) {
    ++position;
  }
  return position + 2 <= value.size && value.data[position] == '0' &&
         iree_tolower(value.data[position + 1]) == 'x';
}

IREE_API_EXPORT bool iree_string_view_atof(iree_string_view_t value,
                                           float* out_value) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_hex_float(value)) {
    iree_hex_float_t parsed;
    if (!iree_string_view_parse_hex_float(value, &parsed)) return false;
    const uint32_t bits = (uint32_t)iree_hex_float_to_ieee_bits(
        &parsed, FLT_MANT_DIG, FLT_MIN_EXP - 1, FLT_MAX_EXP - 1,
        2 - FLT_MIN_EXP, 31);
    float parsed_value = 0.0f;
    memcpy(&parsed_value, &bits, sizeof(parsed_value));
    *out_value = parsed_value;
    return true;
  }

  // Copy to scratch memory with a NUL terminator.
  char temp[64] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  float parsed_value = strtof(temp, &end);
  if (temp == end || end != temp + value.size) return false;
  *out_value = parsed_value;
  return true;
}

IREE_API_EXPORT bool iree_string_view_atod(iree_string_view_t value,
                                           double* out_value) {
  value = iree_string_view_trim(value);
  if (iree_string_view_is_hex_float(value)) {
    iree_hex_float_t parsed;
    if (!iree_string_view_parse_hex_float(value, &parsed)) return false;
    const uint64_t bits =
        iree_hex_float_to_ieee_bits(&parsed, DBL_MANT_DIG, DBL_MIN_EXP - 1,
                                    DBL_MAX_EXP - 1, 2 - DBL_MIN_EXP, 63);
    double parsed_value = 0.0;
    memcpy(&parsed_value, &bits, sizeof(parsed_value));
    *out_value = parsed_value;
    return true;
  }

  // Copy to scratch memory with a NUL terminator.
  char temp[64] = {0};
  if (iree_string_view_is_empty(value) || value.size >= IREE_ARRAYSIZE(temp)) {
    return false;
  }
  memcpy(temp, value.data, value.size);

  // Attempt to parse.
  errno = 0;
  char* end = NULL;
  double parsed_value = strtod(temp, &end);
  if (temp == end || end != temp + value.size) return false;
  *out_value = parsed_value;
  return true;
}

static bool iree_string_view_parse_hex_nibble(char c, uint8_t* out_b) {
  if ('0' <= c && c <= '9') {
    *out_b = (uint8_t)(c - '0');
    return true;
  } else if ('a' <= c && c <= 'f') {
    *out_b = (uint8_t)(c - 'a' + 10);
    return true;
  } else if ('A' <= c && c <= 'F') {
    *out_b = (uint8_t)(c - 'A' + 10);
    return true;
  }
  return false;
}

IREE_API_EXPORT bool iree_string_view_parse_hex_bytes(
    iree_string_view_t value, iree_host_size_t buffer_length,
    uint8_t* out_buffer) {
  // Strip leading/trailing whitespace.
  value = iree_string_view_trim(value);

  // Try to consume all bytes.
  for (iree_host_size_t i = 0; i < buffer_length; ++i) {
    // Strip interior whitespace/-'s.
    if (i > 0 && value.size) {
      char c = value.data[0];
      if (c == ' ' || c == '-') {
        value = iree_string_view_remove_prefix(value, 1);
      }
    }

    // Ensure there are two nibbles.
    if (value.size < 2) return false;

    // Hex nibbles to byte; fail if invalid characters.
    uint8_t b0 = 0, b1 = 0;
    if (!iree_string_view_parse_hex_nibble(value.data[0], &b0) ||
        !iree_string_view_parse_hex_nibble(value.data[1], &b1)) {
      return false;
    }
    out_buffer[i] = (b0 << 4) + b1;

    // Eat the nibbles.
    value = iree_string_view_remove_prefix(value, 2);
  }

  // Should have consumed all characters.
  return iree_string_view_is_empty(value);
}

IREE_API_EXPORT iree_status_t iree_string_view_parse_device_size(
    iree_string_view_t value, iree_device_size_t* out_size) {
  // TODO(benvanik): probably worth to-lowering here on the size. Having copies
  // of all the string view utils for just this case is code size overkill. For
  // now only accept lazy lowercase.
  iree_device_size_t scale = 1;
  if (iree_string_view_consume_suffix(&value, IREE_SV("kb"))) {
    scale = 1000;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("kib"))) {
    scale = 1024;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("mb"))) {
    scale = 1000 * 1000;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("mib"))) {
    scale = 1024 * 1024;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("gb"))) {
    scale = 1000 * 1000 * 1000;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("gib"))) {
    scale = 1024 * 1024 * 1024;
  } else if (iree_string_view_consume_suffix(&value, IREE_SV("b"))) {
    scale = 1;
  }
  uint64_t size = 0;
  if (!iree_string_view_atoi_uint64(value, &size)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "size must be an integer, got '%.*s'",
                            (int)value.size, value.data);
  }
  size *= scale;
  if (size > IREE_DEVICE_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "size parsed (%" PRIu64
        ") is out of the representable range of iree_device_size_t (%" PRIu64
        ")",
        size, (uint64_t)IREE_DEVICE_SIZE_MAX);
  }
  *out_size = size;
  return iree_ok_status();
}
