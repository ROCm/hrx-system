// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "module.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/io/file_contents.h"
#include "iree/io/stdio_stream.h"
#include "iree/io/stream.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/format/text/printer.h"
#include "loom/link/linker.h"
#include "loom/ops/low/ops.h"
#include "loomc/iree.h"
#include "result.h"
#include "source.h"
#include "target.h"
#include "workspace.h"

enum {
  LOOMC_MODULE_SERIALIZE_BLOCK_SIZE = 32 * 1024,
};

struct loomc_module_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release module handle storage.
  loomc_allocator_t allocator;

  // Context retained by the module handle.
  loomc_context_t* context;

  // Workspace retained for the module arena block pool.
  loomc_workspace_t* workspace;

  // Internal linked, parsed, or optimized module.
  loom_module_t* module;
};

typedef struct loomc_module_resolved_serialize_options_t {
  // Serialized source format.
  loomc_source_format_t format;

  // Identifier attached to returned source handles.
  loomc_string_view_t identifier;

  // Text presentation policy for textual output.
  loomc_module_text_presentation_t text_presentation;

  // Requested target-low descriptor-set key for low assembly text.
  loomc_string_view_t low_asm_descriptor_set_key;
} loomc_module_resolved_serialize_options_t;

typedef struct loomc_module_resolved_deserialize_options_t {
  // Input source format.
  loomc_source_format_t format;

  // Identifier used for diagnostics and module provenance.
  loomc_string_view_t identifier;

} loomc_module_resolved_deserialize_options_t;

typedef struct loomc_module_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;

  // Source associated with emitted diagnostics.
  const loomc_source_t* source;
} loomc_module_diagnostic_capture_t;

typedef struct loomc_module_byte_buffer_stream_t {
  // Base stream header.
  iree_io_stream_t base;

  // Allocator used for stream and buffer storage.
  loomc_allocator_t allocator;

  // Owned contiguous byte buffer.
  uint8_t* data;

  // Number of valid bytes in data.
  iree_host_size_t length;

  // Allocated capacity of data.
  iree_host_size_t capacity;

  // Current stream cursor.
  iree_io_stream_pos_t position;
} loomc_module_byte_buffer_stream_t;

static void loomc_module_destroy(loomc_module_t* module) {
  loomc_allocator_t allocator = module->allocator;
  loom_module_free(module->module);
  loomc_workspace_release(module->workspace);
  loomc_context_release(module->context);
  loomc_allocator_free(allocator, module);
}

static loomc_string_view_t loomc_module_default_identifier(
    loomc_source_format_t format) {
  switch (format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return loomc_make_cstring_view("module.loom");
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return loomc_make_cstring_view("module.loombc");
    default:
      return loomc_string_view_empty();
  }
}

static bool loomc_module_serialize_options_has_field(
    const loomc_module_serialize_options_t* options, iree_host_size_t offset,
    iree_host_size_t length) {
  return options != NULL && options->structure_size >= offset + length;
}

#define LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options, field)  \
  loomc_module_serialize_options_has_field(                       \
      options, offsetof(loomc_module_serialize_options_t, field), \
      sizeof(((loomc_module_serialize_options_t*)0)->field))

static loomc_status_t loomc_module_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_resolve_serialize_options(
    const loomc_module_serialize_options_t* options,
    loomc_module_resolved_serialize_options_t* out_options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_SERIALIZE_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module serialize option extensions are not supported");
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_module_validate_string_view(options->identifier));
    if (LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options,
                                                 low_asm_descriptor_set_key)) {
      LOOMC_RETURN_IF_ERROR(loomc_module_validate_string_view(
          options->low_asm_descriptor_set_key));
    }
  }

  loomc_source_format_t format =
      options ? options->format : LOOMC_SOURCE_FORMAT_TEXT;
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = LOOMC_SOURCE_FORMAT_TEXT;
  }
  if (format != LOOMC_SOURCE_FORMAT_TEXT &&
      format != LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module serialize format must be text or bytecode");
  }

  loomc_string_view_t identifier =
      options ? options->identifier : loomc_string_view_empty();
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_module_default_identifier(format);
  }

  loomc_module_text_presentation_t text_presentation =
      LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT;
  if (LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options, text_presentation)) {
    text_presentation = options->text_presentation;
  }
  switch (text_presentation) {
    case LOOMC_MODULE_TEXT_PRESENTATION_DEFAULT:
    case LOOMC_MODULE_TEXT_PRESENTATION_GENERIC:
    case LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM:
      break;
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize text_presentation is not supported");
  }

  loomc_string_view_t low_asm_descriptor_set_key = loomc_string_view_empty();
  if (LOOMC_MODULE_SERIALIZE_OPTIONS_HAS_FIELD(options,
                                               low_asm_descriptor_set_key)) {
    low_asm_descriptor_set_key = options->low_asm_descriptor_set_key;
  }
  if (text_presentation == LOOMC_MODULE_TEXT_PRESENTATION_GENERIC &&
      !loomc_string_view_is_empty(low_asm_descriptor_set_key)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module serialize low_asm_descriptor_set_key requires low-asm text "
        "presentation");
  }

  out_options->format = format;
  out_options->identifier = identifier;
  out_options->text_presentation = text_presentation;
  out_options->low_asm_descriptor_set_key = low_asm_descriptor_set_key;
  return loomc_ok_status();
}

static bool loomc_module_byte_span_has_bytecode_magic(
    loomc_byte_span_t contents) {
  return contents.data_length >= LOOM_BYTECODE_MAGIC_LENGTH &&
         memcmp(contents.data, LOOM_BYTECODE_MAGIC,
                LOOM_BYTECODE_MAGIC_LENGTH) == 0;
}

static loomc_status_t loomc_module_resolve_deserialize_options(
    const loomc_module_deserialize_options_t* options,
    loomc_string_view_t fallback_identifier,
    loomc_source_format_t fallback_format, loomc_byte_span_t contents,
    loomc_module_resolved_deserialize_options_t* out_options) {
  if (options != NULL) {
    if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
        options->type != LOOMC_STRUCTURE_TYPE_MODULE_DESERIALIZE_OPTIONS) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module deserialize options have an unknown structure type");
    }
    if (options->structure_size != 0 &&
        options->structure_size < sizeof(*options)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module deserialize options structure_size is too small");
    }
    if (options->next != NULL) {
      return loomc_make_status(
          LOOMC_STATUS_UNIMPLEMENTED,
          "module deserialize option extensions are not supported");
    }
  }

  loomc_source_format_t format =
      options ? options->format : LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = fallback_format;
  }
  if (format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    format = loomc_module_byte_span_has_bytecode_magic(contents)
                 ? LOOMC_SOURCE_FORMAT_BYTECODE
                 : LOOMC_SOURCE_FORMAT_TEXT;
  }
  if (format != LOOMC_SOURCE_FORMAT_TEXT &&
      format != LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "module deserialize format must be text or bytecode");
  }

  loomc_string_view_t identifier =
      options ? options->identifier : loomc_string_view_empty();
  if (loomc_string_view_is_empty(identifier)) {
    identifier = fallback_identifier;
  }
  if (loomc_string_view_is_empty(identifier)) {
    identifier = loomc_module_default_identifier(format);
  }

  out_options->format = format;
  out_options->identifier = identifier;
  return loomc_ok_status();
}

static loomc_status_t loomc_module_require_internal(
    const loomc_module_t* module, const loom_module_t** out_internal_module) {
  if (module == NULL || out_internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and out_internal_module must not be NULL");
  }
  if (module->module == NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module does not contain internal IR");
  }
  *out_internal_module = module->module;
  return loomc_ok_status();
}

static iree_string_view_t loomc_module_internal_name(
    const loom_module_t* internal_module) {
  if (internal_module == NULL ||
      internal_module->name_id == LOOM_STRING_ID_INVALID) {
    return IREE_SV("module");
  }
  return internal_module->strings.entries[internal_module->name_id];
}

static iree_status_t loomc_module_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_module_diagnostic_capture_t* capture =
      (loomc_module_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, capture->source, diagnostic));
}

static iree_status_t loomc_module_write_iree_stream(void* user_data,
                                                    iree_string_view_t text) {
  return iree_io_stream_write((iree_io_stream_t*)user_data, text.size,
                              text.data);
}

static bool loomc_module_is_low_function_op(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op) ||
         loom_low_func_decl_isa(op);
}

static loomc_status_t loomc_module_select_single_low_descriptor_set_key(
    const loom_module_t* internal_module,
    const loomc_target_pass_environment_t* target_environment,
    loomc_string_view_t* out_descriptor_set_key, bool* out_found_low_function,
    bool* out_found_descriptor_set) {
  *out_descriptor_set_key = loomc_string_view_empty();
  *out_found_low_function = false;
  *out_found_descriptor_set = false;
  if (internal_module->body == NULL) {
    return loomc_ok_status();
  }
  const loom_block_t* block =
      loom_region_const_entry_block(internal_module->body);
  if (block == NULL) {
    return loomc_ok_status();
  }
  const loom_low_descriptor_registry_t* registry =
      target_environment != NULL
          ? &target_environment->low_descriptor_registry.registry
          : NULL;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!loomc_module_is_low_function_op(op)) {
      continue;
    }
    *out_found_low_function = true;
    if (registry == NULL) {
      continue;
    }
    loom_low_resolved_target_t target = {0};
    LOOMC_RETURN_IF_ERROR(
        loomc_status_from_iree(loom_low_resolve_function_target(
            internal_module, op, registry, loom_target_selection_empty(),
            (iree_diagnostic_emitter_t){0}, &target)));
    if (target.descriptor_set == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_FAILED_PRECONDITION,
          "module text serialization could not resolve a target-low "
          "descriptor set for one or more low functions");
    }
    if (!*out_found_descriptor_set) {
      *out_descriptor_set_key =
          loomc_string_view_from_iree(target.descriptor_set_key);
      *out_found_descriptor_set = true;
      continue;
    }
    if (!loomc_string_view_equal(
            *out_descriptor_set_key,
            loomc_string_view_from_iree(target.descriptor_set_key))) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module text serialization found multiple target-low descriptor "
          "sets; pass low_asm_descriptor_set_key or request generic text");
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_module_text_print_options(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loom_text_print_options_t* out_options) {
  *out_options = (loom_text_print_options_t){
      .flags = LOOM_TEXT_PRINT_DEFAULT,
  };
  const loomc_target_pass_environment_t* target_environment =
      loomc_context_target_pass_environment(module->context);
  loomc_target_pass_environment_initialize_text_asm_environment(
      target_environment, &out_options->low_asm_environment);
  if (options->text_presentation == LOOMC_MODULE_TEXT_PRESENTATION_GENERIC) {
    return loomc_ok_status();
  }
  if (!loomc_string_view_is_empty(options->low_asm_descriptor_set_key)) {
    out_options->low_asm_descriptor_set_key =
        iree_string_view_from_loomc(options->low_asm_descriptor_set_key);
    return loomc_ok_status();
  }

  loomc_string_view_t descriptor_set_key = loomc_string_view_empty();
  bool found_low_function = false;
  bool found_descriptor_set = false;
  LOOMC_RETURN_IF_ERROR(loomc_module_select_single_low_descriptor_set_key(
      internal_module, target_environment, &descriptor_set_key,
      &found_low_function, &found_descriptor_set));
  if (found_descriptor_set) {
    out_options->low_asm_descriptor_set_key =
        iree_string_view_from_loomc(descriptor_set_key);
    return loomc_ok_status();
  }
  if (options->text_presentation == LOOMC_MODULE_TEXT_PRESENTATION_LOW_ASM) {
    return loomc_make_status(
        found_low_function ? LOOMC_STATUS_FAILED_PRECONDITION
                           : LOOMC_STATUS_NOT_FOUND,
        "module text serialization could not infer a target-low descriptor "
        "set");
  }
  return loomc_ok_status();
}

static loomc_module_byte_buffer_stream_t* loomc_module_byte_buffer_stream_cast(
    iree_io_stream_t* base_stream) {
  return (loomc_module_byte_buffer_stream_t*)base_stream;
}

static void loomc_module_byte_buffer_stream_destroy(
    iree_io_stream_t* base_stream) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  loomc_allocator_t allocator = stream->allocator;
  loomc_allocator_free(allocator, stream->data);
  loomc_allocator_free(allocator, stream);
}

static iree_io_stream_pos_t loomc_module_byte_buffer_stream_offset(
    iree_io_stream_t* base_stream) {
  return loomc_module_byte_buffer_stream_cast(base_stream)->position;
}

static iree_io_stream_pos_t loomc_module_byte_buffer_stream_length(
    iree_io_stream_t* base_stream) {
  return (iree_io_stream_pos_t)loomc_module_byte_buffer_stream_cast(base_stream)
      ->length;
}

static iree_status_t loomc_module_byte_buffer_stream_seek(
    iree_io_stream_t* base_stream, iree_io_stream_seek_mode_t seek_mode,
    iree_io_stream_pos_t seek_offset) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  iree_io_stream_pos_t position = 0;
  switch (seek_mode) {
    case IREE_IO_STREAM_SEEK_SET:
      position = seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_CURRENT:
      position = stream->position + seek_offset;
      break;
    case IREE_IO_STREAM_SEEK_FROM_END:
      position = (iree_io_stream_pos_t)stream->length + seek_offset;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stream seek mode is invalid");
  }
  if (position < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream seek before start");
  }
  stream->position = position;
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_reserve(
    loomc_module_byte_buffer_stream_t* stream, iree_host_size_t capacity) {
  if (stream->capacity >= capacity) {
    return iree_ok_status();
  }
  iree_host_size_t new_capacity = stream->capacity ? stream->capacity : 4096;
  while (new_capacity < capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2;
  }
  loomc_allocator_alloc_params_t params = {
      .byte_length = new_capacity,
  };
  void* data = stream->data;
  iree_status_t status = iree_status_from_loomc(stream->allocator.ctl(
      stream->allocator.self, LOOMC_ALLOCATOR_COMMAND_REALLOC, &params, &data));
  if (iree_status_is_ok(status)) {
    stream->data = (uint8_t*)data;
    stream->capacity = new_capacity;
  }
  return status;
}

static iree_status_t loomc_module_byte_buffer_stream_read(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_capacity,
    void* buffer, iree_host_size_t* out_buffer_length) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  if (stream->position < 0 || (uint64_t)stream->position > stream->length) {
    if (out_buffer_length != NULL) {
      *out_buffer_length = 0;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "read past stream end");
  }
  const iree_host_size_t remaining =
      stream->length - (iree_host_size_t)stream->position;
  const iree_host_size_t read_length =
      remaining < buffer_capacity ? remaining : buffer_capacity;
  if (read_length != 0) {
    memcpy(buffer, stream->data + stream->position, read_length);
  }
  stream->position += read_length;
  if (out_buffer_length != NULL) {
    *out_buffer_length = read_length;
    return iree_ok_status();
  }
  if (read_length != buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "short stream read");
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_write(
    iree_io_stream_t* base_stream, iree_host_size_t buffer_length,
    const void* buffer) {
  loomc_module_byte_buffer_stream_t* stream =
      loomc_module_byte_buffer_stream_cast(base_stream);
  if (stream->position < 0 || (uint64_t)stream->position > IREE_HOST_SIZE_MAX ||
      buffer_length > IREE_HOST_SIZE_MAX - (iree_host_size_t)stream->position) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream write length is too large");
  }
  const iree_host_size_t position = (iree_host_size_t)stream->position;
  const iree_host_size_t new_position = position + buffer_length;
  IREE_RETURN_IF_ERROR(
      loomc_module_byte_buffer_stream_reserve(stream, new_position));
  if (position > stream->length) {
    memset(stream->data + stream->length, 0, position - stream->length);
  }
  if (buffer_length != 0) {
    memcpy(stream->data + position, buffer, buffer_length);
  }
  stream->position = (iree_io_stream_pos_t)new_position;
  if (new_position > stream->length) {
    stream->length = new_position;
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_fill(
    iree_io_stream_t* base_stream, iree_io_stream_pos_t count,
    const void* pattern, iree_host_size_t pattern_length) {
  if (count < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "stream fill count is negative");
  }
  for (iree_io_stream_pos_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(loomc_module_byte_buffer_stream_write(
        base_stream, pattern_length, pattern));
  }
  return iree_ok_status();
}

static iree_status_t loomc_module_byte_buffer_stream_map_read(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_const_byte_span_t* out_span) {
  (void)base_stream;
  (void)length;
  (void)out_span;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "module byte buffer streams do not support mapping");
}

static iree_status_t loomc_module_byte_buffer_stream_map_write(
    iree_io_stream_t* base_stream, iree_host_size_t length,
    iree_byte_span_t* out_span) {
  (void)base_stream;
  (void)length;
  (void)out_span;
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "module byte buffer streams do not support mapping");
}

static const iree_io_stream_vtable_t loomc_module_byte_buffer_stream_vtable = {
    .destroy = loomc_module_byte_buffer_stream_destroy,
    .offset = loomc_module_byte_buffer_stream_offset,
    .length = loomc_module_byte_buffer_stream_length,
    .seek = loomc_module_byte_buffer_stream_seek,
    .read = loomc_module_byte_buffer_stream_read,
    .write = loomc_module_byte_buffer_stream_write,
    .fill = loomc_module_byte_buffer_stream_fill,
    .map_read = loomc_module_byte_buffer_stream_map_read,
    .map_write = loomc_module_byte_buffer_stream_map_write,
};

static loomc_status_t loomc_module_byte_buffer_stream_create(
    loomc_allocator_t allocator,
    loomc_module_byte_buffer_stream_t** out_stream) {
  *out_stream = NULL;
  loomc_module_byte_buffer_stream_t* stream = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*stream), (void**)&stream));
  memset(stream, 0, sizeof(*stream));
  iree_atomic_ref_count_init(&stream->base.ref_count);
  stream->base.vtable = &loomc_module_byte_buffer_stream_vtable;
  stream->base.mode =
      IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
      IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE;
  stream->allocator = allocator;
  *out_stream = stream;
  return loomc_ok_status();
}

static uint8_t* loomc_module_byte_buffer_stream_take_storage(
    loomc_module_byte_buffer_stream_t* stream, iree_host_size_t* out_length) {
  uint8_t* data = stream->data;
  *out_length = stream->length;
  stream->data = NULL;
  stream->length = 0;
  stream->capacity = 0;
  stream->position = 0;
  return data;
}

static loomc_status_t loomc_module_serialize_text_to_iree_stream(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    iree_io_stream_t* target_stream) {
  loom_output_stream_t output_stream = {
      .write = loomc_module_write_iree_stream,
      .user_data = target_stream,
  };
  loom_text_print_options_t print_options;
  LOOMC_RETURN_IF_ERROR(loomc_module_text_print_options(
      module, internal_module, options, &print_options));
  return loomc_status_from_iree(loom_text_print_module_with_options(
      internal_module, &output_stream, &print_options));
}

static loomc_status_t loomc_module_serialize_text_to_file(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options, FILE* file) {
  loom_output_stream_t output_stream;
  loom_output_stream_for_file(file, &output_stream);
  loom_text_print_options_t print_options;
  LOOMC_RETURN_IF_ERROR(loomc_module_text_print_options(
      module, internal_module, options, &print_options));
  return loomc_status_from_iree(loom_text_print_module_with_options(
      internal_module, &output_stream, &print_options));
}

static loomc_status_t loomc_module_serialize_text_to_source(
    const loomc_module_t* module, const loom_module_t* internal_module,
    const loomc_module_resolved_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_text_print_options_t print_options;
  loomc_status_t status = loomc_module_text_print_options(
      module, internal_module, options, &print_options);
  if (loomc_status_is_ok(status)) {
    status =
        loomc_status_from_iree(loom_text_print_module_to_builder_with_options(
            internal_module, &builder, &print_options));
  }
  const iree_host_size_t length = iree_string_builder_size(&builder);
  char* storage = NULL;
  if (loomc_status_is_ok(status)) {
    storage = iree_string_builder_take_storage(&builder);
    status = loomc_source_create_take_contents(
        LOOMC_SOURCE_FORMAT_TEXT, options->identifier,
        loomc_make_byte_span(storage, length), allocator, out_source);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, storage);
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}

static loomc_status_t loomc_module_serialize_bytecode_to_stream(
    const loom_module_t* internal_module, iree_io_stream_t* target_stream,
    loomc_allocator_t allocator) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(LOOMC_MODULE_SERIALIZE_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &block_pool);
  loomc_status_t status = loomc_status_from_iree(loom_bytecode_write_module(
      internal_module, target_stream, /*options=*/NULL, &block_pool));
  iree_arena_block_pool_deinitialize(&block_pool);
  return status;
}

static loomc_status_t loomc_module_serialize_bytecode_to_source(
    const loom_module_t* internal_module, loomc_string_view_t identifier,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  loomc_module_byte_buffer_stream_t* stream = NULL;
  loomc_status_t status =
      loomc_module_byte_buffer_stream_create(allocator, &stream);
  if (loomc_status_is_ok(status)) {
    status = loomc_module_serialize_bytecode_to_stream(
        internal_module, &stream->base, allocator);
  }

  uint8_t* storage = NULL;
  iree_host_size_t stream_length = 0;
  if (loomc_status_is_ok(status)) {
    storage =
        loomc_module_byte_buffer_stream_take_storage(stream, &stream_length);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_take_contents(
        LOOMC_SOURCE_FORMAT_BYTECODE, identifier,
        loomc_make_byte_span(storage, stream_length), allocator, out_source);
  }
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, storage);
  }
  if (stream != NULL) {
    iree_io_stream_release(&stream->base);
  }
  return status;
}

static loomc_status_t loomc_module_result_fail_status(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_string_view_t code, loomc_status_t status) {
  LOOMC_RETURN_IF_ERROR(loomc_result_add_status_diagnostic(
      result, source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR, code, status));
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

static loomc_status_t loomc_module_mark_deserialize_failed(
    loomc_result_t* result, const loomc_source_t* source,
    loomc_host_size_t before_diagnostic_count) {
  if (loomc_result_diagnostic_count(result) != before_diagnostic_count) {
    return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
  }
  loomc_status_t status = loomc_make_status(
      LOOMC_STATUS_INVALID_ARGUMENT, "module source did not deserialize");
  loomc_status_t add_status = loomc_module_result_fail_status(
      result, source, loomc_make_cstring_view("MODULE/DESERIALIZE"), status);
  loomc_status_free(status);
  return add_status;
}

static loomc_status_t loomc_module_deserialize_text_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loomc_module_t* module, loomc_result_t* result,
    loom_module_t** out_internal_module) {
  *out_internal_module = NULL;
  loomc_byte_span_t contents = loomc_source_contents(source);
  loomc_module_diagnostic_capture_t capture = {
      .result = result,
      .source = source,
  };
  loom_text_parse_options_t parse_options = {
      .diagnostic_sink =
          {
              .fn = loomc_module_capture_diagnostic,
              .user_data = &capture,
          },
  };
  loomc_target_pass_environment_initialize_text_asm_environment(
      loomc_context_target_pass_environment(context),
      &parse_options.low_asm_environment);
  return loomc_status_from_iree(loom_text_parse(
      iree_make_string_view((const char*)contents.data, contents.data_length),
      iree_string_view_from_loomc(options->identifier),
      loomc_context_loom_context(context), loomc_module_block_pool(module),
      &parse_options, out_internal_module));
}

static loomc_status_t loomc_module_deserialize_bytecode_source(
    loomc_context_t* context, const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t* module, loomc_result_t* result,
    loom_module_t** out_internal_module) {
  *out_internal_module = NULL;
  loomc_byte_span_t contents = loomc_source_contents(source);
  loomc_module_diagnostic_capture_t capture = {
      .result = result,
      .source = source,
  };
  loom_bytecode_read_options_t read_options = {
      .diagnostic_sink =
          {
              .fn = loomc_module_capture_diagnostic,
              .user_data = &capture,
          },
  };
  loom_bytecode_read_result_t read_result = {0};
  loomc_status_t status = loomc_status_from_iree(loom_bytecode_read_module(
      iree_make_const_byte_span(contents.data, contents.data_length),
      iree_string_view_from_loomc(options->identifier),
      loomc_context_loom_context(context), loomc_module_block_pool(module),
      &read_options, &read_result, out_internal_module,
      iree_allocator_from_loomc(allocator)));
  if (loomc_status_is_ok(status) && read_result.error_count != 0) {
    loom_module_free(*out_internal_module);
    *out_internal_module = NULL;
  }
  return status;
}

static loomc_status_t loomc_module_deserialize_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_resolved_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));

  loomc_module_t* module = NULL;
  loomc_status_t status =
      loomc_module_create_empty(context, workspace, allocator, &module);
  loom_module_t* internal_module = NULL;
  const loomc_host_size_t before_diagnostic_count =
      loomc_result_diagnostic_count(result);
  if (loomc_status_is_ok(status)) {
    if (options->format == LOOMC_SOURCE_FORMAT_TEXT) {
      status = loomc_module_deserialize_text_source(
          context, source, options, module, result, &internal_module);
    } else {
      status = loomc_module_deserialize_bytecode_source(
          context, source, options, allocator, module, result,
          &internal_module);
    }
  }
  if (loomc_status_is_ok(status) && internal_module == NULL) {
    status = loomc_module_mark_deserialize_failed(result, source,
                                                  before_diagnostic_count);
  }
  if (loomc_status_is_ok(status) && internal_module != NULL) {
    status = loomc_module_set_loom_module(module, internal_module);
    if (loomc_status_is_ok(status)) {
      internal_module = NULL;
    }
  }

  if (loomc_status_is_ok(status)) {
    if (loomc_result_succeeded(result)) {
      *out_module = module;
      module = NULL;
    }
    *out_result = result;
    result = NULL;
  }
  loom_module_free(internal_module);
  loomc_module_release(module);
  loomc_result_release(result);
  return status;
}

static void loomc_module_release_file_contents(void* user_data,
                                               loomc_byte_span_t contents) {
  (void)contents;
  iree_io_file_contents_free((iree_io_file_contents_t*)user_data);
}

loomc_status_t loomc_module_create_empty(loomc_context_t* context,
                                         loomc_workspace_t* workspace,
                                         loomc_allocator_t allocator,
                                         loomc_module_t** out_module) {
  if (context == NULL || workspace == NULL || out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context, workspace, and out_module must not be "
                             "NULL");
  }
  *out_module = NULL;

  loomc_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->allocator = allocator;
  module->context = context;
  loomc_context_retain(context);
  module->workspace = workspace;
  loomc_workspace_retain(workspace);

  *out_module = module;
  return loomc_ok_status();
}

loomc_allocator_t loomc_module_allocator(const loomc_module_t* module) {
  IREE_ASSERT_ARGUMENT(module);
  return module->allocator;
}

loomc_context_t* loomc_module_context(const loomc_module_t* module) {
  return module ? module->context : NULL;
}

iree_arena_block_pool_t* loomc_module_block_pool(loomc_module_t* module) {
  return module ? loomc_workspace_block_pool(module->workspace) : NULL;
}

loomc_status_t loomc_module_set_loom_module(loomc_module_t* module,
                                            loom_module_t* internal_module) {
  if (module == NULL || internal_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module and internal_module must not be NULL");
  }
  if (module->module != NULL) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "module already owns internal module storage");
  }
  module->module = internal_module;
  return loomc_ok_status();
}

loom_module_t* loomc_module_loom_module(loomc_module_t* module) {
  return module ? module->module : NULL;
}

const loom_module_t* loomc_module_const_loom_module(
    const loomc_module_t* module) {
  return module ? module->module : NULL;
}

void loomc_module_retain(loomc_module_t* module) {
  if (module == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&module->ref_count);
}

void loomc_module_release(loomc_module_t* module) {
  if (module == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    loomc_module_destroy(module);
  }
}

loomc_status_t loomc_module_clone(const loomc_module_t* source_module,
                                  loomc_workspace_t* workspace,
                                  loomc_allocator_t allocator,
                                  loomc_module_t** out_module) {
  if (out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_module must not be NULL");
  }
  *out_module = NULL;
  if (workspace == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "workspace must not be NULL");
  }

  const loom_module_t* source_internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(source_module, &source_internal_module));

  loomc_module_t* module = NULL;
  loom_module_t* cloned_internal_module = NULL;
  loomc_status_t status = loomc_module_create_empty(
      source_module->context, workspace, allocator, &module);
  if (loomc_status_is_ok(status)) {
    const loom_module_t* source_modules[] = {source_internal_module};
    loom_link_options_t link_options = {
        .module_name = loomc_module_internal_name(source_internal_module),
    };
    status = loomc_status_from_iree(loom_link_materialized_modules(
        source_modules, IREE_ARRAYSIZE(source_modules), &link_options,
        loomc_module_block_pool(module), iree_allocator_from_loomc(allocator),
        &cloned_internal_module));
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_set_loom_module(module, cloned_internal_module);
    if (loomc_status_is_ok(status)) {
      cloned_internal_module = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    *out_module = module;
    module = NULL;
  }

  loom_module_free(cloned_internal_module);
  loomc_module_release(module);
  return status;
}

loomc_status_t loomc_module_deserialize_from_source(
    loomc_context_t* context, loomc_workspace_t* workspace,
    const loomc_source_t* source,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL || source == NULL ||
      out_module == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, source, out_module, and out_result must not be "
        "NULL");
  }

  loomc_module_resolved_deserialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_module_resolve_deserialize_options(
      options, loomc_source_identifier(source), loomc_source_format(source),
      loomc_source_contents(source), &resolved_options));
  return loomc_module_deserialize_source(context, workspace, source,
                                         &resolved_options, allocator,
                                         out_module, out_result);
}

loomc_status_t loomc_module_deserialize_from_file(
    loomc_context_t* context, loomc_workspace_t* workspace, FILE* file,
    const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL || file == NULL ||
      out_module == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, file, out_module, and out_result must not be "
        "NULL");
  }

  uint8_t* contents = NULL;
  loomc_host_size_t contents_length = 0;
  LOOMC_RETURN_IF_ERROR(loomc_source_read_file_to_storage(
      file, allocator, &contents, &contents_length));

  loomc_module_resolved_deserialize_options_t resolved_options = {0};
  loomc_status_t status = loomc_module_resolve_deserialize_options(
      options, loomc_string_view_empty(), LOOMC_SOURCE_FORMAT_UNKNOWN,
      loomc_make_byte_span(contents, contents_length), &resolved_options);

  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    status = loomc_source_create_take_contents(
        resolved_options.format, resolved_options.identifier,
        loomc_make_byte_span(contents, contents_length), allocator, &source);
    if (loomc_status_is_ok(status)) {
      contents = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_source(context, workspace, source,
                                             &resolved_options, allocator,
                                             out_module, out_result);
  }
  loomc_source_release(source);
  loomc_allocator_free(allocator, contents);
  return status;
}

loomc_status_t loomc_module_deserialize_from_path(
    loomc_context_t* context, loomc_workspace_t* workspace,
    loomc_string_view_t path, const loomc_module_deserialize_options_t* options,
    loomc_allocator_t allocator, loomc_module_t** out_module,
    loomc_result_t** out_result) {
  if (out_module != NULL) {
    *out_module = NULL;
  }
  if (out_result != NULL) {
    *out_result = NULL;
  }
  if (context == NULL || workspace == NULL ||
      loomc_string_view_is_empty(path) || out_module == NULL ||
      out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context, workspace, path, out_module, and out_result must be valid");
  }

  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  iree_io_file_contents_t* file_contents = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_io_file_contents_read(
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &file_contents)));

  loomc_byte_span_t contents =
      loomc_byte_span_from_iree(file_contents->const_buffer);
  loomc_module_resolved_deserialize_options_t resolved_options = {0};
  loomc_status_t status = loomc_module_resolve_deserialize_options(
      options, path, LOOMC_SOURCE_FORMAT_UNKNOWN, contents, &resolved_options);

  loomc_source_t* source = NULL;
  if (loomc_status_is_ok(status)) {
    loomc_source_options_t source_options = {
        .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
        .structure_size = sizeof(source_options),
        .format = resolved_options.format,
        .identifier = resolved_options.identifier,
        .contents = contents,
        .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
        .release = loomc_module_release_file_contents,
        .release_user_data = file_contents,
    };
    status = loomc_source_create(&source_options, allocator, &source);
    if (loomc_status_is_ok(status)) {
      file_contents = NULL;
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_module_deserialize_source(context, workspace, source,
                                             &resolved_options, allocator,
                                             out_module, out_result);
  }
  loomc_source_release(source);
  iree_io_file_contents_free(file_contents);
  return status;
}

loomc_status_t loomc_module_serialize_to_source(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source) {
  if (out_source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_source must not be NULL");
  }
  *out_source = NULL;

  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_module_resolve_serialize_options(options, &resolved_options));
  const loom_module_t* internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(module, &internal_module));

  switch (resolved_options.format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return loomc_module_serialize_text_to_source(
          module, internal_module, &resolved_options, allocator, out_source);
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return loomc_module_serialize_bytecode_to_source(
          internal_module, resolved_options.identifier, allocator, out_source);
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "module serialize format must be text or bytecode");
  }
}

loomc_status_t loomc_module_serialize_to_file(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, FILE* file) {
  if (file == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "file must not be NULL");
  }

  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_module_resolve_serialize_options(options, &resolved_options));
  const loom_module_t* internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(module, &internal_module));

  if (resolved_options.format == LOOMC_SOURCE_FORMAT_TEXT) {
    return loomc_module_serialize_text_to_file(module, internal_module,
                                               &resolved_options, file);
  }

  loomc_source_t* source = NULL;
  loomc_status_t status = loomc_module_serialize_bytecode_to_source(
      internal_module, resolved_options.identifier,
      loomc_module_allocator(module), &source);
  if (loomc_status_is_ok(status)) {
    loomc_byte_span_t contents = loomc_source_contents(source);
    if (contents.data_length != 0 &&
        fwrite(contents.data, 1, contents.data_length, file) !=
            contents.data_length) {
      status = loomc_make_status(LOOMC_STATUS_UNKNOWN,
                                 "failed to write serialized module");
    }
  }
  loomc_source_release(source);
  return status;
}

loomc_status_t loomc_module_serialize_to_path(
    const loomc_module_t* module,
    const loomc_module_serialize_options_t* options, loomc_string_view_t path,
    loomc_allocator_t allocator) {
  if (loomc_string_view_is_empty(path)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "path must not be empty");
  }

  loomc_module_resolved_serialize_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_module_resolve_serialize_options(options, &resolved_options));
  const loom_module_t* internal_module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_module_require_internal(module, &internal_module));

  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator.ctl must not be NULL");
  }
  iree_io_stream_t* stream = NULL;
  loomc_status_t status = loomc_status_from_iree(iree_io_stdio_stream_open(
      IREE_IO_STDIO_STREAM_MODE_WRITE | IREE_IO_STDIO_STREAM_MODE_DISCARD,
      iree_string_view_from_loomc(path), iree_allocator_from_loomc(allocator),
      &stream));
  if (loomc_status_is_ok(status)) {
    if (resolved_options.format == LOOMC_SOURCE_FORMAT_TEXT) {
      status = loomc_module_serialize_text_to_iree_stream(
          module, internal_module, &resolved_options, stream);
    } else {
      status = loomc_module_serialize_bytecode_to_stream(internal_module,
                                                         stream, allocator);
    }
  }
  if (stream != NULL) {
    iree_io_stream_release(stream);
  }
  return status;
}
