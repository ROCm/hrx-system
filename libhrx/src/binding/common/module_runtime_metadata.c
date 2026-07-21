// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#include "common/internal.h"
#include "common/printf_format.h"

#define IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE (2 * sizeof(uint32_t))
#define IREE_HAL_STREAMING_PRINTF_RECORD_FLAG_STDERR 1u
#define IREE_HAL_STREAMING_PRINTF_RECORD_FLAG_CONSTANT_FORMAT 2u

//===----------------------------------------------------------------------===//
// Runtime Parameter Slots
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_streaming_module_set_runtime_service_slot(
    iree_string_view_t name,
    iree_hal_executable_function_runtime_parameter_info_t parameter_info,
    iree_hal_streaming_runtime_parameter_slot_t* out_slot) {
  if (parameter_info.length != sizeof(uint64_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "runtime parameter `%.*s` has unsupported size %" PRIu32,
        (int)name.size, name.data, parameter_info.length);
  }
  if (IREE_UNLIKELY(out_slot->present)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "runtime parameter `%.*s` is declared more than "
                            "once",
                            (int)name.size, name.data);
  }

  *out_slot = (iree_hal_streaming_runtime_parameter_slot_t){
      .offset = parameter_info.offset,
      .length = parameter_info.length,
      .present = true,
  };
  return iree_ok_status();
}

static uint16_t iree_hal_streaming_implicit_runtime_parameter_length(
    iree_string_view_t name) {
  // The backend initializes the complete implicit launch block, including
  // fields a given kernel does not read. Streaming only supplies the runtime
  // services that are external to that block.
  if (iree_string_view_equal(name, IREE_SV("hidden_block_count_x")) ||
      iree_string_view_equal(name, IREE_SV("hidden_block_count_y")) ||
      iree_string_view_equal(name, IREE_SV("hidden_block_count_z")) ||
      iree_string_view_equal(name, IREE_SV("hidden_dynamic_lds_size")) ||
      iree_string_view_equal(name, IREE_SV("hidden_private_base")) ||
      iree_string_view_equal(name, IREE_SV("hidden_shared_base"))) {
    return sizeof(uint32_t);
  }
  if (iree_string_view_equal(name, IREE_SV("hidden_group_size_x")) ||
      iree_string_view_equal(name, IREE_SV("hidden_group_size_y")) ||
      iree_string_view_equal(name, IREE_SV("hidden_group_size_z")) ||
      iree_string_view_equal(name, IREE_SV("hidden_remainder_x")) ||
      iree_string_view_equal(name, IREE_SV("hidden_remainder_y")) ||
      iree_string_view_equal(name, IREE_SV("hidden_remainder_z")) ||
      iree_string_view_equal(name, IREE_SV("hidden_grid_dims"))) {
    return sizeof(uint16_t);
  }
  if (iree_string_view_equal(name, IREE_SV("hidden_tool_correlation_id")) ||
      iree_string_view_equal(name, IREE_SV("hidden_global_offset_x")) ||
      iree_string_view_equal(name, IREE_SV("hidden_global_offset_y")) ||
      iree_string_view_equal(name, IREE_SV("hidden_global_offset_z")) ||
      iree_string_view_equal(name, IREE_SV("hidden_multigrid_sync_arg")) ||
      iree_string_view_equal(name, IREE_SV("hidden_default_queue")) ||
      iree_string_view_equal(name, IREE_SV("hidden_completion_action")) ||
      iree_string_view_equal(name, IREE_SV("hidden_queue_ptr"))) {
    return sizeof(uint64_t);
  }
  return 0;
}

static iree_status_t iree_hal_streaming_module_initialize_runtime_parameter(
    iree_hal_streaming_symbol_t* symbol,
    iree_hal_executable_function_runtime_parameter_info_t parameter_info) {
  const iree_string_view_t name = parameter_info.name;
  if (iree_string_view_equal(name, IREE_SV("hidden_printf_buffer"))) {
    return iree_hal_streaming_module_set_runtime_service_slot(
        name, parameter_info, &symbol->runtime_services.printf_buffer);
  }
  if (iree_string_view_equal(name, IREE_SV("hidden_hostcall_buffer"))) {
    return iree_hal_streaming_module_set_runtime_service_slot(
        name, parameter_info, &symbol->runtime_services.hostcall_buffer);
  }
  if (iree_string_view_equal(name, IREE_SV("hidden_heap_v1"))) {
    return iree_hal_streaming_module_set_runtime_service_slot(
        name, parameter_info, &symbol->runtime_services.heap_v1);
  }
  const uint16_t implicit_parameter_length =
      iree_hal_streaming_implicit_runtime_parameter_length(name);
  if (implicit_parameter_length != 0) {
    if (parameter_info.length != implicit_parameter_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "implicit runtime parameter `%.*s` has size "
                              "%" PRIu32 "; expected %" PRIu16,
                              (int)name.size, name.data, parameter_info.length,
                              implicit_parameter_length);
    }
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "runtime parameter `%.*s` has no streaming provider",
                          (int)name.size, name.data);
}

static iree_status_t iree_hal_streaming_module_validate_runtime_service_slots(
    const iree_hal_streaming_symbol_t* symbol) {
  typedef struct iree_hal_streaming_runtime_service_slot_info_t {
    // Runtime parameter name for diagnostics.
    iree_string_view_t name;
    // Slot populated by the runtime service.
    const iree_hal_streaming_runtime_parameter_slot_t* slot;
  } iree_hal_streaming_runtime_service_slot_info_t;
  const iree_hal_streaming_runtime_service_slot_info_t slots[] = {
      {
          .name = IREE_SV("hidden_printf_buffer"),
          .slot = &symbol->runtime_services.printf_buffer,
      },
      {
          .name = IREE_SV("hidden_hostcall_buffer"),
          .slot = &symbol->runtime_services.hostcall_buffer,
      },
      {
          .name = IREE_SV("hidden_heap_v1"),
          .slot = &symbol->runtime_services.heap_v1,
      },
  };
  iree_host_size_t present_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(slots); ++i) {
    if (!slots[i].slot->present) continue;
    ++present_count;
    const uint64_t slot_begin = slots[i].slot->offset;
    const uint64_t slot_end = slot_begin + slots[i].slot->length;
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (!slots[j].slot->present) continue;
      const uint64_t prior_begin = slots[j].slot->offset;
      const uint64_t prior_end = prior_begin + slots[j].slot->length;
      if (prior_begin < slot_end && slot_begin < prior_end) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "runtime parameters `%.*s` and `%.*s` overlap",
                                (int)slots[j].name.size, slots[j].name.data,
                                (int)slots[i].name.size, slots[i].name.data);
      }
    }
  }
  if (IREE_UNLIKELY(present_count >
                    IREE_HAL_DISPATCH_MAX_RUNTIME_PARAMETER_PATCHES)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "runtime parameter patch capacity %u is smaller than %" PRIhsz
        " declared runtime services",
        IREE_HAL_DISPATCH_MAX_RUNTIME_PARAMETER_PATCHES, present_count);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_symbol_runtime_slots(
    iree_hal_streaming_module_t* module) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < module->symbol_count; ++i) {
    iree_hal_streaming_symbol_t* symbol = &module->symbols[i];
    if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) continue;

    const iree_hal_executable_function_t function =
        iree_hal_executable_function_from_index(symbol->export_ordinal);
    iree_hal_executable_function_info_t function_info;
    status = iree_hal_executable_function_info(symbol->executable, function,
                                               &function_info);
    iree_hal_executable_function_runtime_parameter_info_t* parameters = NULL;
    if (iree_status_is_ok(status) && function_info.runtime_parameter_count) {
      status = iree_allocator_malloc_array(
          module->host_allocator, function_info.runtime_parameter_count,
          sizeof(*parameters), (void**)&parameters);
    }
    if (iree_status_is_ok(status) && function_info.runtime_parameter_count) {
      status = iree_hal_executable_function_runtime_parameters(
          symbol->executable, function, function_info.runtime_parameter_count,
          parameters);
    }
    for (iree_host_size_t j = 0;
         iree_status_is_ok(status) && j < function_info.runtime_parameter_count;
         ++j) {
      status = iree_hal_streaming_module_initialize_runtime_parameter(
          symbol, parameters[j]);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_streaming_module_validate_runtime_service_slots(symbol);
    }
    iree_allocator_free(module->host_allocator, parameters);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Buffered Printf Runtime
//===----------------------------------------------------------------------===//

typedef struct iree_hal_streaming_printf_drain_state_t {
  // Module retained while the host call needs the printf format table.
  iree_hal_streaming_module_t* module;
  // Transient host-visible printf FIFO owned by the host call.
  iree_hal_streaming_buffer_t* buffer;
} iree_hal_streaming_printf_drain_state_t;

static iree_host_size_t iree_hal_streaming_printf_strnlen(
    const char* data, iree_host_size_t data_length) {
  iree_host_size_t length = 0;
  while (length < data_length && data[length] != '\0') ++length;
  return length;
}

static iree_host_size_t iree_hal_streaming_printf_align8(
    iree_host_size_t value) {
  return (value + 7u) & ~(iree_host_size_t)7u;
}

static iree_status_t iree_hal_streaming_printf_find_format(
    iree_hal_streaming_module_t* module, uint64_t hash,
    iree_string_view_t* out_format) {
  *out_format = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < module->printf_format_count; ++i) {
    const iree_hal_streaming_printf_format_t* record =
        &module->printf_formats[i];
    if (record->hash == hash) {
      *out_format = record->format;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "buffered printf record references unknown format hash 0x%016" PRIx64,
      hash);
}

static bool iree_hal_streaming_printf_write(FILE* stream, int* out_count,
                                            const char* format, ...) {
  va_list args;
  va_start(args, format);
  const int result = vfprintf(stream, format, args);
  va_end(args);
  if (result < 0) {
    *out_count = result;
    return false;
  }
  if (*out_count <= INT_MAX - result) {
    *out_count += result;
  } else {
    *out_count = INT_MAX;
  }
  return true;
}

static bool iree_hal_streaming_printf_write_bytes(FILE* stream, int* out_count,
                                                  const char* data,
                                                  iree_host_size_t length) {
  if (length == 0) return true;
  if (IREE_UNLIKELY(length > (iree_host_size_t)INT_MAX)) {
    *out_count = -1;
    return false;
  }
  if (fwrite(data, 1, length, stream) != length) {
    *out_count = -1;
    return false;
  }
  if (*out_count <= INT_MAX - (int)length) {
    *out_count += (int)length;
  } else {
    *out_count = INT_MAX;
  }
  return true;
}

static const uint8_t* iree_hal_streaming_printf_read_u64(
    const uint8_t* argument, const uint8_t* argument_end, uint64_t* out_value) {
  if (IREE_UNLIKELY((iree_host_size_t)(argument_end - argument) <
                    sizeof(*out_value))) {
    return NULL;
  }
  memcpy(out_value, argument, sizeof(*out_value));
  return argument + sizeof(*out_value);
}

static const uint8_t* iree_hal_streaming_printf_process_spec(
    FILE* stream, int* out_count, const char* spec_begin,
    iree_host_size_t spec_length, const uint8_t* argument,
    const uint8_t* argument_end) {
  if (IREE_UNLIKELY(spec_length >= 128)) return NULL;
  char spec[128];
  memcpy(spec, spec_begin, spec_length);
  spec[spec_length] = '\0';
  iree_hal_streaming_printf_spec_t parsed_spec;
  if (IREE_UNLIKELY(!iree_hal_streaming_printf_parse_spec(spec, spec_length,
                                                          &parsed_spec))) {
    return NULL;
  }

  const int star_count = parsed_spec.star_count;
  int star0 = 0;
  int star1 = 0;
  uint64_t raw_value = 0;
  if (star_count >= 1) {
    argument =
        iree_hal_streaming_printf_read_u64(argument, argument_end, &raw_value);
    if (!argument) return NULL;
    star0 = (int)raw_value;
  }
  if (star_count >= 2) {
    argument =
        iree_hal_streaming_printf_read_u64(argument, argument_end, &raw_value);
    if (!argument) return NULL;
    star1 = (int)raw_value;
  }

  const char conversion = parsed_spec.conversion;

#define IREE_HAL_STREAMING_PRINTF_CALL(value)                                 \
  do {                                                                        \
    if (star_count == 0) {                                                    \
      iree_hal_streaming_printf_write(stream, out_count, spec, value);        \
    } else if (star_count == 1) {                                             \
      iree_hal_streaming_printf_write(stream, out_count, spec, star0, value); \
    } else {                                                                  \
      iree_hal_streaming_printf_write(stream, out_count, spec, star0, star1,  \
                                      value);                                 \
    }                                                                         \
  } while (0)

  switch (conversion) {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      if (parsed_spec.is_signed_integer) {
        switch (parsed_spec.length_modifier) {
          case IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_H:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_HH: {
            const int value = (int)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_L: {
            const long value = (long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_LL: {
            const long long value = (long long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_J: {
            const intmax_t value = (intmax_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_Z: {
            const iree_hal_streaming_printf_signed_size_t value =
                (iree_hal_streaming_printf_signed_size_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_T: {
            const ptrdiff_t value = (ptrdiff_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
        }
      } else {
        switch (parsed_spec.length_modifier) {
          case IREE_HAL_STREAMING_PRINTF_LENGTH_DEFAULT:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_H:
          case IREE_HAL_STREAMING_PRINTF_LENGTH_HH: {
            const unsigned int value = (unsigned int)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_L: {
            const unsigned long value = (unsigned long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_LL: {
            const unsigned long long value = (unsigned long long)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_J: {
            const uintmax_t value = (uintmax_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_Z: {
            const size_t value = (size_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
          case IREE_HAL_STREAMING_PRINTF_LENGTH_T: {
            const iree_hal_streaming_printf_unsigned_ptrdiff_t value =
                (iree_hal_streaming_printf_unsigned_ptrdiff_t)raw_value;
            IREE_HAL_STREAMING_PRINTF_CALL(value);
            break;
          }
        }
      }
      break;
    }
    case 'c': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      if (parsed_spec.is_wide_character) {
        const wint_t value = (wint_t)raw_value;
        IREE_HAL_STREAMING_PRINTF_CALL(value);
      } else {
        const int value = (int)raw_value;
        IREE_HAL_STREAMING_PRINTF_CALL(value);
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
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      double value = 0;
      memcpy(&value, &raw_value, sizeof(value));
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      break;
    }
    case 's': {
      const iree_host_size_t available_bytes =
          (iree_host_size_t)(argument_end - argument);
      const iree_host_size_t string_length = iree_hal_streaming_printf_strnlen(
          (const char*)argument, available_bytes);
      if (IREE_UNLIKELY(string_length == available_bytes)) return NULL;
      const char* value = (const char*)argument;
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      argument += iree_hal_streaming_printf_align8(string_length + 1);
      if (IREE_UNLIKELY(argument > argument_end)) return NULL;
      break;
    }
    case 'p': {
      argument = iree_hal_streaming_printf_read_u64(argument, argument_end,
                                                    &raw_value);
      if (!argument) return NULL;
      void* value = (void*)(uintptr_t)raw_value;
      IREE_HAL_STREAMING_PRINTF_CALL(value);
      break;
    }
    default:
      return NULL;
  }

#undef IREE_HAL_STREAMING_PRINTF_CALL

  return argument;
}

static iree_status_t iree_hal_streaming_printf_write_formatted(
    FILE* stream, iree_string_view_t format, const uint8_t* arguments,
    iree_host_size_t argument_length) {
  const uint8_t* argument = arguments;
  const uint8_t* argument_end = arguments + argument_length;
  int out_count = 0;
  iree_host_size_t cursor = 0;
  while (cursor < format.size) {
    const char* percent =
        memchr(format.data + cursor, '%', format.size - cursor);
    if (!percent) {
      if (!iree_hal_streaming_printf_write_bytes(
              stream, &out_count, format.data + cursor, format.size - cursor)) {
        return iree_make_status(IREE_STATUS_UNKNOWN,
                                "buffered printf output write failed");
      }
      return iree_ok_status();
    }

    const iree_host_size_t percent_offset =
        (iree_host_size_t)(percent - format.data);
    if (!iree_hal_streaming_printf_write_bytes(stream, &out_count,
                                               format.data + cursor,
                                               percent_offset - cursor)) {
      return iree_make_status(IREE_STATUS_UNKNOWN,
                              "buffered printf output write failed");
    }

    iree_host_size_t spec_end = percent_offset + 1;
    if (spec_end < format.size && format.data[spec_end] == '%') {
      if (!iree_hal_streaming_printf_write_bytes(stream, &out_count, "%", 1)) {
        return iree_make_status(IREE_STATUS_UNKNOWN,
                                "buffered printf output write failed");
      }
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
                              "buffered printf format is unterminated");
    }
    ++spec_end;

    argument = iree_hal_streaming_printf_process_spec(
        stream, &out_count, percent, spec_end - percent_offset, argument,
        argument_end);
    if (IREE_UNLIKELY(out_count < 0 || !argument)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffered printf arguments are malformed");
    }
    cursor = spec_end;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_printf_drain_constant_record(
    iree_hal_streaming_module_t* module, FILE* stream, const uint8_t* payload,
    iree_host_size_t payload_length) {
  if (IREE_UNLIKELY(payload_length < sizeof(uint64_t))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffered printf constant record is truncated");
  }
  uint64_t hash = 0;
  memcpy(&hash, payload, sizeof(hash));

  iree_string_view_t format = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      iree_hal_streaming_printf_find_format(module, hash, &format));
  return iree_hal_streaming_printf_write_formatted(
      stream, format, payload + sizeof(uint64_t),
      payload_length - sizeof(uint64_t));
}

static iree_status_t iree_hal_streaming_printf_drain_inline_record(
    FILE* stream, const uint8_t* payload, iree_host_size_t payload_length) {
  const iree_host_size_t format_length =
      iree_hal_streaming_printf_strnlen((const char*)payload, payload_length);
  if (IREE_UNLIKELY(format_length == payload_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffered printf inline format is unterminated");
  }
  const iree_host_size_t argument_offset =
      iree_hal_streaming_printf_align8(format_length + 1);
  if (IREE_UNLIKELY(argument_offset > payload_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffered printf inline record is malformed");
  }
  const iree_string_view_t format =
      iree_make_string_view((const char*)payload, format_length);
  return iree_hal_streaming_printf_write_formatted(
      stream, format, payload + argument_offset,
      payload_length - argument_offset);
}

static iree_status_t iree_hal_streaming_printf_drain(
    iree_hal_streaming_module_t* module, iree_hal_streaming_buffer_t* buffer) {
  if (IREE_UNLIKELY(!buffer->host_ptr)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "buffered printf FIFO is not host-visible");
  }
  if (IREE_UNLIKELY(buffer->size <
                    IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffered printf FIFO is smaller than its header");
  }

  const uint8_t* data = (const uint8_t*)buffer->host_ptr;
  uint32_t used_bytes = 0;
  uint32_t capacity_bytes = 0;
  memcpy(&used_bytes, data, sizeof(used_bytes));
  memcpy(&capacity_bytes, data + sizeof(used_bytes), sizeof(capacity_bytes));

  const iree_host_size_t storage_capacity =
      (iree_host_size_t)buffer->size -
      IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE;
  if (IREE_UNLIKELY(capacity_bytes > storage_capacity ||
                    used_bytes > capacity_bytes)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffered printf FIFO header is out of range");
  }

  const uint8_t* cursor = data + IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE;
  iree_host_size_t remaining = used_bytes;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && remaining > 0) {
    if (IREE_UNLIKELY(remaining < sizeof(uint32_t))) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffered printf record is truncated");
      break;
    }

    uint32_t control = 0;
    memcpy(&control, cursor, sizeof(control));
    const iree_host_size_t record_length = control >> 2;
    if (IREE_UNLIKELY(record_length < sizeof(uint32_t) ||
                      record_length > remaining)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "buffered printf record length is invalid");
      break;
    }

    FILE* stream = (control & IREE_HAL_STREAMING_PRINTF_RECORD_FLAG_STDERR)
                       ? stderr
                       : stdout;
    const uint8_t* payload = cursor + sizeof(uint32_t);
    const iree_host_size_t payload_length = record_length - sizeof(uint32_t);
    if (control & IREE_HAL_STREAMING_PRINTF_RECORD_FLAG_CONSTANT_FORMAT) {
      status = iree_hal_streaming_printf_drain_constant_record(
          module, stream, payload, payload_length);
    } else {
      status = iree_hal_streaming_printf_drain_inline_record(stream, payload,
                                                             payload_length);
    }

    cursor += record_length;
    remaining -= record_length;
  }

  fflush(stdout);
  fflush(stderr);
  return status;
}

static iree_status_t iree_hal_streaming_printf_drain_host_call(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* call_context) {
  (void)args;
  (void)call_context;
  iree_hal_streaming_printf_drain_state_t* state =
      (iree_hal_streaming_printf_drain_state_t*)user_data;
  iree_status_t status =
      iree_hal_streaming_printf_drain(state->module, state->buffer);
  iree_hal_streaming_memory_release_transient_buffer(state->buffer);
  iree_hal_streaming_module_release(state->module);
  iree_allocator_free(iree_allocator_system(), state);
  return status;
}

iree_status_t iree_hal_streaming_symbol_create_printf_buffer(
    iree_hal_streaming_symbol_t* symbol,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  if (!symbol->runtime_services.printf_buffer.present) return iree_ok_status();

  size_t buffer_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_context_limit(
      symbol->module->context,
      IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE, &buffer_size));
  if (IREE_UNLIKELY(buffer_size <
                    IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffered printf FIFO size is too small");
  }
  if (IREE_UNLIKELY(buffer_size >
                    (size_t)UINT32_MAX +
                        IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffered printf FIFO size exceeds ABI limit");
  }

  iree_hal_streaming_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_streaming_memory_allocate_runtime_host(
      symbol->module->context, (iree_host_size_t)buffer_size, &buffer));

  uint32_t used_bytes = 0;
  uint32_t capacity_bytes =
      (uint32_t)(buffer_size - IREE_HAL_STREAMING_PRINTF_BUFFER_HEADER_SIZE);
  memcpy(buffer->host_ptr, &used_bytes, sizeof(used_bytes));
  memcpy((uint8_t*)buffer->host_ptr + sizeof(used_bytes), &capacity_bytes,
         sizeof(capacity_bytes));
  *out_buffer = buffer;
  return iree_ok_status();
}

// Module initialization validates the fixed service slots for overlap and
// runtime-parameter-list capacity before dispatches can reach this hot path.
static inline void
iree_hal_streaming_dispatch_config_append_validated_runtime_pointer(
    iree_hal_dispatch_runtime_parameter_list_t* runtime_parameters,
    const iree_hal_streaming_runtime_parameter_slot_t* slot,
    uint64_t device_ptr) {
  iree_hal_dispatch_runtime_parameter_patch_t* patch =
      &runtime_parameters->patches[runtime_parameters->count++];
  patch->offset = slot->offset;
  patch->length = (uint16_t)slot->length;
  patch->reserved = 0;
  memcpy(patch->data, &device_ptr, sizeof(device_ptr));
}

iree_status_t iree_hal_streaming_symbol_prepare_runtime_dispatch_config(
    iree_hal_streaming_symbol_t* symbol, iree_hal_streaming_context_t* context,
    bool enable_printf,
    iree_hal_dispatch_runtime_parameter_list_t* runtime_parameters,
    iree_hal_dispatch_config_t* config,
    iree_hal_streaming_buffer_t** out_printf_buffer) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(runtime_parameters);
  IREE_ASSERT_ARGUMENT(config);
  if (out_printf_buffer) *out_printf_buffer = NULL;
  *runtime_parameters = (iree_hal_dispatch_runtime_parameter_list_t){0};
  config->runtime_parameters = NULL;

  iree_hal_streaming_buffer_t* printf_buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (enable_printf && symbol->runtime_services.printf_buffer.present) {
    if (IREE_UNLIKELY(!out_printf_buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "printf runtime buffer requested without an output owner");
    }
    status =
        iree_hal_streaming_symbol_create_printf_buffer(symbol, &printf_buffer);
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_dispatch_config_append_validated_runtime_pointer(
          runtime_parameters, &symbol->runtime_services.printf_buffer,
          printf_buffer->device_ptr);
    }
  }

  if (iree_status_is_ok(status) &&
      symbol->runtime_services.hostcall_buffer.present) {
    uint64_t hostcall_buffer_device_ptr =
        iree_hal_streaming_context_cached_rocm_hostcall_buffer(context);
    if (hostcall_buffer_device_ptr == 0) {
      status = iree_hal_streaming_context_rocm_hostcall_buffer(
          context, &hostcall_buffer_device_ptr);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_dispatch_config_append_validated_runtime_pointer(
          runtime_parameters, &symbol->runtime_services.hostcall_buffer,
          hostcall_buffer_device_ptr);
    }
  }

  if (iree_status_is_ok(status) && symbol->runtime_services.heap_v1.present) {
    uint64_t heap_device_ptr =
        iree_hal_streaming_context_cached_rocm_malloc_heap(context);
    if (heap_device_ptr == 0) {
      status = iree_hal_streaming_context_rocm_device_malloc_heap(
          context, &heap_device_ptr);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_streaming_dispatch_config_append_validated_runtime_pointer(
          runtime_parameters, &symbol->runtime_services.heap_v1,
          heap_device_ptr);
    }
  }

  if (!iree_status_is_ok(status)) {
    if (printf_buffer) {
      iree_hal_streaming_memory_release_transient_buffer(printf_buffer);
    }
    return status;
  }
  if (runtime_parameters->count != 0) {
    config->runtime_parameters = runtime_parameters;
  }
  if (out_printf_buffer) {
    *out_printf_buffer = printf_buffer;
  } else if (printf_buffer) {
    iree_hal_streaming_memory_release_transient_buffer(printf_buffer);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_symbol_queue_printf_drain(
    iree_hal_streaming_symbol_t* symbol, iree_hal_streaming_stream_t* stream,
    iree_hal_streaming_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(stream);
  if (!buffer) return iree_ok_status();

  iree_hal_streaming_printf_drain_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_memory_release_transient_buffer(buffer);
    return status;
  }
  state->module = symbol->module;
  iree_hal_streaming_module_retain(state->module);
  state->buffer = buffer;

  const uint64_t args[4] = {0, 0, 0, 0};
  iree_hal_host_call_t call =
      iree_hal_make_host_call(iree_hal_streaming_printf_drain_host_call, state);
  status = iree_hal_streaming_queue_host_call(stream, call, args,
                                              IREE_HAL_HOST_CALL_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_hal_streaming_memory_release_transient_buffer(buffer);
    iree_hal_streaming_module_release(state->module);
    iree_allocator_free(iree_allocator_system(), state);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Printf Metadata
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_hal_streaming_module_executable_count(
    const iree_hal_streaming_module_t* module) {
  return module->executables ? module->executable_count : 1;
}

static iree_hal_executable_t* iree_hal_streaming_module_executable_at(
    const iree_hal_streaming_module_t* module, iree_host_size_t index) {
  return module->executables ? module->executables[index] : module->executable;
}

static iree_status_t iree_hal_streaming_parse_printf_metadata_record(
    iree_string_view_t record, uint64_t* out_hash,
    iree_string_view_t* out_format) {
  *out_hash = 0;
  *out_format = iree_string_view_empty();

  if (!iree_string_view_consume_prefix(&record, IREE_SV("0:0:"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "printf metadata record must start with fixed `0:0:` fields");
  }
  iree_host_size_t comma = iree_string_view_find_char(record, ',', 0);
  if (comma == IREE_STRING_VIEW_NPOS || comma == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "printf metadata record is missing hash/format delimiter");
  }
  iree_string_view_t hash_text = iree_string_view_substr(record, 0, comma);
  if (hash_text.size > 16) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "printf metadata hash exceeds 64 bits");
  }
  uint64_t hash = 0;
  if (!iree_string_view_atoi_uint64_base(hash_text, 16, &hash)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "printf metadata hash is not hexadecimal");
  }

  *out_hash = hash;
  *out_format =
      iree_string_view_substr(record, comma + 1, IREE_STRING_VIEW_NPOS);
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_append_printf_format(
    iree_hal_streaming_module_t* module, uint64_t hash,
    iree_string_view_t format) {
  for (iree_host_size_t i = 0; i < module->printf_format_count; ++i) {
    iree_hal_streaming_printf_format_t* existing = &module->printf_formats[i];
    if (existing->hash != hash) continue;
    if (iree_string_view_equal(existing->format, format))
      return iree_ok_status();
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "printf metadata hash collision for hash 0x%016" PRIx64, hash);
  }
  module->printf_formats[module->printf_format_count++] =
      (iree_hal_streaming_printf_format_t){
          .hash = hash,
          .format = format,
      };
  return iree_ok_status();
}

static iree_status_t iree_hal_streaming_module_initialize_printf_metadata(
    iree_hal_streaming_module_t* module) {
  iree_status_t status = iree_ok_status();

  iree_host_size_t total_record_count = 0;
  const iree_host_size_t executable_count =
      iree_hal_streaming_module_executable_count(module);
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < executable_count; ++i) {
    iree_host_size_t record_count = 0;
    status = iree_hal_executable_runtime_metadata_count(
        iree_hal_streaming_module_executable_at(module, i),
        IREE_SV("amdhsa.printf"), &record_count);
    if (!iree_status_is_ok(status)) break;
    if (!iree_host_size_checked_add(total_record_count, record_count,
                                    &total_record_count)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "printf metadata record count overflow");
    }
  }
  if (!iree_status_is_ok(status) || total_record_count == 0) return status;

  iree_host_size_t format_storage_size = 0;
  if (!iree_host_size_checked_mul(total_record_count,
                                  sizeof(module->printf_formats[0]),
                                  &format_storage_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "printf metadata table size overflow");
  }
  status = iree_allocator_malloc(module->host_allocator, format_storage_size,
                                 (void**)&module->printf_formats);
  if (iree_status_is_ok(status)) {
    memset(module->printf_formats, 0, format_storage_size);
  }

  iree_hal_executable_runtime_metadata_record_t* records = NULL;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < executable_count; ++i) {
    iree_hal_executable_t* executable =
        iree_hal_streaming_module_executable_at(module, i);
    iree_host_size_t record_count = 0;
    status = iree_hal_executable_runtime_metadata_count(
        executable, IREE_SV("amdhsa.printf"), &record_count);
    if (!iree_status_is_ok(status) || record_count == 0) continue;

    iree_host_size_t records_size = 0;
    if (!iree_host_size_checked_mul(record_count, sizeof(records[0]),
                                    &records_size)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "printf metadata record storage overflow");
      break;
    }
    status = iree_allocator_malloc(module->host_allocator, records_size,
                                   (void**)&records);
    if (!iree_status_is_ok(status)) break;
    memset(records, 0, records_size);

    status = iree_hal_executable_runtime_metadata_records(
        executable, IREE_SV("amdhsa.printf"), record_count, records);
    for (iree_host_size_t j = 0; iree_status_is_ok(status) && j < record_count;
         ++j) {
      uint64_t hash = 0;
      iree_string_view_t format = iree_string_view_empty();
      status = iree_hal_streaming_parse_printf_metadata_record(records[j].value,
                                                               &hash, &format);
      if (iree_status_is_ok(status)) {
        status = iree_hal_streaming_module_append_printf_format(module, hash,
                                                                format);
      }
    }

    iree_allocator_free(module->host_allocator, records);
    records = NULL;
  }
  if (records) iree_allocator_free(module->host_allocator, records);
  return status;
}

//===----------------------------------------------------------------------===//
// Module Runtime Metadata
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_module_initialize_runtime_metadata(
    iree_hal_streaming_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_streaming_module_initialize_symbol_runtime_slots(module);
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_module_initialize_printf_metadata(module);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
