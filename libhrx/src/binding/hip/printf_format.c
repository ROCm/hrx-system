// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/printf_format.h"

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <string.h>
#include <wchar.h>

// Length modifier carried by one supported printf conversion.
typedef enum iree_hip_printf_length_modifier_e {
  IREE_HIP_PRINTF_LENGTH_DEFAULT = 0,
  IREE_HIP_PRINTF_LENGTH_HH,
  IREE_HIP_PRINTF_LENGTH_H,
  IREE_HIP_PRINTF_LENGTH_L,
  IREE_HIP_PRINTF_LENGTH_LL,
  IREE_HIP_PRINTF_LENGTH_J,
  IREE_HIP_PRINTF_LENGTH_Z,
  IREE_HIP_PRINTF_LENGTH_T,
} iree_hip_printf_length_modifier_t;

// Parsed printf conversion that can be materialized safely from a device ABI
// scalar slot and passed to the host formatter.
typedef struct iree_hip_printf_spec_t {
  // Terminal conversion character, such as d, s, or f.
  char conversion;
  // Parsed length modifier affecting the host vararg type.
  iree_hip_printf_length_modifier_t length_modifier;
  // Number of width and precision arguments supplied with *.
  uint8_t star_count;
  // True when the field width is supplied by an argument.
  bool has_width_star;
  // True when the field precision is supplied by an argument.
  bool has_precision_star;
  // True when the conversion consumes a signed integer argument.
  bool is_signed_integer;
  // True when the conversion consumes a wide character argument.
  bool is_wide_character;
} iree_hip_printf_spec_t;

// Outcome of parsing one conversion specifier.
typedef enum iree_hip_printf_parse_result_e {
  // Conversion grammar is valid.
  IREE_HIP_PRINTF_PARSE_RESULT_OK = 0,
  // Conversion grammar or its type combination is unsupported.
  IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED = 1,
} iree_hip_printf_parse_result_t;

#if SIZE_MAX == UINT_MAX
typedef int iree_hip_printf_signed_size_t;
#elif SIZE_MAX == ULONG_MAX
typedef long iree_hip_printf_signed_size_t;
#elif SIZE_MAX == ULLONG_MAX
typedef long long iree_hip_printf_signed_size_t;
#else
#error "unsupported size_t representation for HIP printf formatting"
#endif

#if PTRDIFF_MAX == INT_MAX
typedef unsigned int iree_hip_printf_unsigned_ptrdiff_t;
#elif PTRDIFF_MAX == LONG_MAX
typedef unsigned long iree_hip_printf_unsigned_ptrdiff_t;
#elif PTRDIFF_MAX == LLONG_MAX
typedef unsigned long long iree_hip_printf_unsigned_ptrdiff_t;
#else
#error "unsupported ptrdiff_t representation for HIP printf formatting"
#endif

static iree_host_size_t iree_hip_printf_strnlen(const char* data,
                                                iree_host_size_t data_length) {
  iree_host_size_t length = 0;
  while (length < data_length && data[length] != '\0') ++length;
  return length;
}

static iree_host_size_t iree_hip_printf_align8(iree_host_size_t value) {
  return (value + 7u) & ~(iree_host_size_t)7u;
}

static bool iree_hip_printf_is_flag(char c) {
  switch (c) {
    case '#':
    case '0':
    case '-':
    case ' ':
    case '+':
    case '\'':
      return true;
    default:
      return false;
  }
}

static void iree_hip_printf_parse_field_length(
    const char* spec, iree_host_size_t conversion_index,
    iree_host_size_t* index) {
  while (*index < conversion_index && spec[*index] >= '0' &&
         spec[*index] <= '9') {
    ++*index;
  }
}

// Parses the supported C printf grammar. Long doubles and wide strings cannot
// be mapped to a safe host-side vararg representation.
static iree_hip_printf_parse_result_t iree_hip_printf_parse_spec(
    const char* spec, iree_host_size_t spec_length,
    iree_hip_printf_spec_t* out_spec) {
  if (!spec || !out_spec || spec_length < 2 || spec[0] != '%') {
    return IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
  }

  *out_spec = (iree_hip_printf_spec_t){0};
  const iree_host_size_t conversion_index = spec_length - 1;
  out_spec->conversion = spec[conversion_index];

  iree_host_size_t index = 1;
  while (index < conversion_index && iree_hip_printf_is_flag(spec[index])) {
    ++index;
  }
  if (index < conversion_index && spec[index] == '*') {
    ++out_spec->star_count;
    out_spec->has_width_star = true;
    ++index;
  } else {
    iree_hip_printf_parse_field_length(spec, conversion_index, &index);
  }
  if (index < conversion_index && spec[index] == '.') {
    ++index;
    if (index < conversion_index && spec[index] == '*') {
      ++out_spec->star_count;
      out_spec->has_precision_star = true;
      ++index;
    } else {
      iree_hip_printf_parse_field_length(spec, conversion_index, &index);
    }
  }

  if (index < conversion_index) {
    switch (spec[index++]) {
      case 'h':
        if (index < conversion_index && spec[index] == 'h') {
          ++index;
          out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_HH;
        } else {
          out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_H;
        }
        break;
      case 'l':
        if (index < conversion_index && spec[index] == 'l') {
          ++index;
          out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_LL;
        } else {
          out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_L;
        }
        break;
      case 'j':
        out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_J;
        break;
      case 'z':
        out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_Z;
        break;
      case 't':
        out_spec->length_modifier = IREE_HIP_PRINTF_LENGTH_T;
        break;
      case 'L':
        return IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
      default:
        return IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
    }
  }
  if (index != conversion_index) {
    return IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
  }

  switch (out_spec->conversion) {
    case 'd':
    case 'i':
      out_spec->is_signed_integer = true;
      return IREE_HIP_PRINTF_PARSE_RESULT_OK;
    case 'o':
    case 'u':
    case 'x':
    case 'X':
      return IREE_HIP_PRINTF_PARSE_RESULT_OK;
    case 'c':
      if (out_spec->length_modifier == IREE_HIP_PRINTF_LENGTH_L) {
        out_spec->is_wide_character = true;
        return IREE_HIP_PRINTF_PARSE_RESULT_OK;
      }
      return out_spec->length_modifier == IREE_HIP_PRINTF_LENGTH_DEFAULT
                 ? IREE_HIP_PRINTF_PARSE_RESULT_OK
                 : IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
      return out_spec->length_modifier == IREE_HIP_PRINTF_LENGTH_DEFAULT ||
                     out_spec->length_modifier == IREE_HIP_PRINTF_LENGTH_L
                 ? IREE_HIP_PRINTF_PARSE_RESULT_OK
                 : IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
    case 's':
    case 'p':
      return out_spec->length_modifier == IREE_HIP_PRINTF_LENGTH_DEFAULT
                 ? IREE_HIP_PRINTF_PARSE_RESULT_OK
                 : IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
    case 'n':
      return IREE_HIP_PRINTF_PARSE_RESULT_OK;
    default:
      return IREE_HIP_PRINTF_PARSE_RESULT_MALFORMED;
  }
}

// Appends one host-formatted conversion to the unpublished message builder.
static iree_status_t iree_hip_printf_append_format(
    iree_string_builder_t* builder, const char* format, ...) {
  va_list args;
  va_start(args, format);
  va_list measure_args;
  va_copy(measure_args, args);
  const int required_length = vsnprintf(NULL, 0, format, measure_args);
  va_end(measure_args);
  if (IREE_UNLIKELY(required_length < 0)) {
    va_end(args);
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "HIP device printf conversion failed");
  }
  const iree_host_size_t current_length = iree_string_builder_size(builder);
  if (IREE_UNLIKELY(current_length > INT_MAX ||
                    (iree_host_size_t)required_length >
                        (iree_host_size_t)INT_MAX - current_length)) {
    va_end(args);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HIP device printf output exceeds its signed int result range");
  }

  char* output = NULL;
  iree_host_size_t output_capacity = 0;
  iree_status_t status = iree_string_builder_reserve_for_append(
      builder, (iree_host_size_t)required_length, &output, &output_capacity);
  if (!iree_status_is_ok(status)) {
    va_end(args);
    return status;
  }
  const int result = vsnprintf(output, output_capacity + 1, format, args);
  va_end(args);
  if (IREE_UNLIKELY(result != required_length)) {
    return iree_make_status(IREE_STATUS_UNKNOWN,
                            "HIP device printf conversion changed length");
  }
  iree_string_builder_commit_append(builder, (iree_host_size_t)result);
  return iree_ok_status();
}

static iree_status_t iree_hip_printf_append_bytes(
    iree_string_builder_t* builder, const char* data, iree_host_size_t length) {
  if (length == 0) return iree_ok_status();
  const iree_host_size_t current_length = iree_string_builder_size(builder);
  if (IREE_UNLIKELY(current_length > INT_MAX ||
                    length > (iree_host_size_t)INT_MAX - current_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "HIP device printf output exceeds its signed int result range");
  }
  return iree_string_builder_append_string(builder,
                                           iree_make_string_view(data, length));
}

static bool iree_hip_printf_read_u64(const uint8_t** inout_argument,
                                     const uint8_t* argument_end,
                                     uint64_t* out_value) {
  const uint8_t* argument = *inout_argument;
  if (IREE_UNLIKELY((iree_host_size_t)(argument_end - argument) <
                    sizeof(*out_value))) {
    return false;
  }
  memcpy(out_value, argument, sizeof(*out_value));
  *inout_argument = argument + sizeof(*out_value);
  return true;
}

static iree_status_t iree_hip_printf_process_spec(
    iree_string_builder_t* builder, iree_string_builder_t* scratch_builder,
    const char* spec_begin, iree_host_size_t spec_length,
    const uint8_t** inout_argument, const uint8_t* argument_end) {
  iree_string_builder_reset(scratch_builder);
  IREE_RETURN_IF_ERROR(iree_string_builder_append_string(
      scratch_builder, iree_make_string_view(spec_begin, spec_length)));
  const char* spec = iree_string_builder_buffer(scratch_builder);

  iree_hip_printf_spec_t parsed_spec;
  const iree_hip_printf_parse_result_t parse_result =
      iree_hip_printf_parse_spec(spec, spec_length, &parsed_spec);
  if (IREE_UNLIKELY(parse_result != IREE_HIP_PRINTF_PARSE_RESULT_OK)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device printf conversion `%s` is unsupported "
                            "or malformed",
                            spec);
  }

  const int star_count = parsed_spec.star_count;
  int star0 = 0;
  int star1 = 0;
  uint64_t raw_value = 0;
  if (star_count >= 1) {
    if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP device printf width or precision argument is truncated");
    }
    star0 = (int)raw_value;
  }
  if (star_count >= 2) {
    if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HIP device printf width or precision argument is truncated");
    }
    star1 = (int)raw_value;
  }
  if (parsed_spec.conversion == 'n') {
    if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP device printf %%n argument is truncated");
    }
    return iree_ok_status();
  }

#define IREE_HIP_PRINTF_WRITE_VALUE(value)                                    \
  do {                                                                        \
    if (star_count == 0) {                                                    \
      IREE_RETURN_IF_ERROR(                                                   \
          iree_hip_printf_append_format(builder, spec, value));               \
    } else if (star_count == 1) {                                             \
      IREE_RETURN_IF_ERROR(                                                   \
          iree_hip_printf_append_format(builder, spec, star0, value));        \
    } else {                                                                  \
      IREE_RETURN_IF_ERROR(                                                   \
          iree_hip_printf_append_format(builder, spec, star0, star1, value)); \
    }                                                                         \
  } while (0)

  switch (parsed_spec.conversion) {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X': {
      if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf integer argument is truncated");
      }
      if (parsed_spec.is_signed_integer) {
        switch (parsed_spec.length_modifier) {
          case IREE_HIP_PRINTF_LENGTH_DEFAULT:
          case IREE_HIP_PRINTF_LENGTH_H:
          case IREE_HIP_PRINTF_LENGTH_HH: {
            const int value = (int)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_L: {
            const long value = (long)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_LL: {
            const long long value = (long long)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_J: {
            const intmax_t value = (intmax_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_Z: {
            const iree_hip_printf_signed_size_t value =
                (iree_hip_printf_signed_size_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_T: {
            const ptrdiff_t value = (ptrdiff_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
        }
      } else {
        switch (parsed_spec.length_modifier) {
          case IREE_HIP_PRINTF_LENGTH_DEFAULT:
          case IREE_HIP_PRINTF_LENGTH_H:
          case IREE_HIP_PRINTF_LENGTH_HH: {
            const unsigned int value = (unsigned int)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_L: {
            const unsigned long value = (unsigned long)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_LL: {
            const unsigned long long value = (unsigned long long)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_J: {
            const uintmax_t value = (uintmax_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_Z: {
            const size_t value = (size_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
          case IREE_HIP_PRINTF_LENGTH_T: {
            const iree_hip_printf_unsigned_ptrdiff_t value =
                (iree_hip_printf_unsigned_ptrdiff_t)raw_value;
            IREE_HIP_PRINTF_WRITE_VALUE(value);
            break;
          }
        }
      }
      break;
    }
    case 'c': {
      if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf character argument is truncated");
      }
      if (parsed_spec.is_wide_character) {
        const wint_t value = (wint_t)raw_value;
        IREE_HIP_PRINTF_WRITE_VALUE(value);
      } else {
        const int value = (int)raw_value;
        IREE_HIP_PRINTF_WRITE_VALUE(value);
      }
      break;
    }
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A': {
      if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf floating-point argument is truncated");
      }
      double value = 0;
      memcpy(&value, &raw_value, sizeof(value));
      IREE_HIP_PRINTF_WRITE_VALUE(value);
      break;
    }
    case 's': {
      const uint8_t* argument = *inout_argument;
      const iree_host_size_t available_bytes =
          (iree_host_size_t)(argument_end - argument);
      const iree_host_size_t string_length =
          iree_hip_printf_strnlen((const char*)argument, available_bytes);
      if (IREE_UNLIKELY(string_length == available_bytes)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf string argument is unterminated");
      }
      const iree_host_size_t occupied_length =
          iree_hip_printf_align8(string_length + 1);
      if (IREE_UNLIKELY(occupied_length > available_bytes)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf string argument padding is truncated");
      }
      const char* value = (const char*)argument;
      IREE_HIP_PRINTF_WRITE_VALUE(value);
      *inout_argument = argument + occupied_length;
      break;
    }
    case 'p': {
      if (!iree_hip_printf_read_u64(inout_argument, argument_end, &raw_value)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HIP device printf pointer argument is truncated");
      }
      void* value = (void*)(uintptr_t)raw_value;
      IREE_HIP_PRINTF_WRITE_VALUE(value);
      break;
    }
    default:
      IREE_CHECK_UNREACHABLE("unhandled parsed HIP printf conversion");
      break;
  }

#undef IREE_HIP_PRINTF_WRITE_VALUE

  return iree_ok_status();
}

static iree_status_t iree_hip_printf_format_impl(
    iree_string_builder_t* builder, iree_string_builder_t* scratch_builder,
    iree_string_view_t format, const uint8_t* arguments,
    iree_host_size_t argument_length) {
  uint8_t empty_arguments = 0;
  if (!arguments) arguments = &empty_arguments;
  const uint8_t* argument = arguments;
  const uint8_t* argument_end = arguments + argument_length;
  iree_host_size_t cursor = 0;
  while (cursor < format.size) {
    const char* percent =
        memchr(format.data + cursor, '%', format.size - cursor);
    if (!percent) {
      IREE_RETURN_IF_ERROR(iree_hip_printf_append_bytes(
          builder, format.data + cursor, format.size - cursor));
      return iree_ok_status();
    }

    const iree_host_size_t percent_offset =
        (iree_host_size_t)(percent - format.data);
    IREE_RETURN_IF_ERROR(iree_hip_printf_append_bytes(
        builder, format.data + cursor, percent_offset - cursor));

    iree_host_size_t spec_end = percent_offset + 1;
    if (spec_end < format.size && format.data[spec_end] == '%') {
      IREE_RETURN_IF_ERROR(iree_hip_printf_append_bytes(builder, "%", 1));
      cursor = spec_end + 1;
      continue;
    }

    static const char kConversionSpecifiers[] = "diouxXfFeEgGaAcspn";
    while (spec_end < format.size &&
           !strchr(kConversionSpecifiers, format.data[spec_end])) {
      ++spec_end;
    }
    if (IREE_UNLIKELY(spec_end == format.size)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "HIP device printf format is unterminated");
    }
    ++spec_end;

    IREE_RETURN_IF_ERROR(iree_hip_printf_process_spec(
        builder, scratch_builder, percent, spec_end - percent_offset, &argument,
        argument_end));
    cursor = spec_end;
  }
  return iree_ok_status();
}

iree_status_t iree_hip_printf_format(iree_string_builder_t* builder,
                                     iree_string_builder_t* scratch_builder,
                                     iree_string_view_t format,
                                     const uint8_t* arguments,
                                     iree_host_size_t argument_length) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(scratch_builder);
  iree_string_builder_reset(builder);
  iree_string_builder_reset(scratch_builder);
  if (IREE_UNLIKELY(format.size != 0 && !format.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device printf format data is NULL");
  }
  if (IREE_UNLIKELY(argument_length != 0 && !arguments)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HIP device printf argument data is NULL");
  }
  iree_status_t status = iree_hip_printf_format_impl(
      builder, scratch_builder, format, arguments, argument_length);
  if (!iree_status_is_ok(status)) {
    iree_string_builder_reset(builder);
    iree_string_builder_reset(scratch_builder);
  }
  return status;
}
