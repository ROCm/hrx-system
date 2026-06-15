// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "link_index.h"

#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "loom/format/bytecode/format.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/text/parser.h"
#include "loom/ir/ir.h"
#include "loomc/iree.h"
#include "result.h"
#include "source.h"

enum {
  LOOMC_LINK_INDEX_DEFAULT_BLOCK_SIZE = 32 * 1024,
};

typedef struct loomc_link_index_builder_source_t {
  // Source retained until builder release or finish.
  loomc_source_t* source;
  // Copied provider label.
  loomc_string_view_t provider_name;
  // Provider search role.
  loomc_link_provider_role_t role;
} loomc_link_index_builder_source_t;

struct loomc_link_index_builder_t {
  // Allocator used for builder storage.
  loomc_allocator_t allocator;
  // Context retained while indexing.
  loomc_context_t* context;
  // Mutable source slots.
  loomc_link_index_builder_source_t* sources;
  // Number of reserved source slots.
  loomc_host_size_t source_count;
  // Allocated source slot capacity.
  loomc_host_size_t source_capacity;
  // Stable block pool backing text modules and index arena metadata.
  iree_arena_block_pool_t* block_pool;
  // Internal mutable module index.
  loom_link_module_index_t* index;
  // Collected indexing result.
  loomc_result_t* result;
  // True after finish has produced a result.
  bool finished;
};

struct loomc_link_index_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used to release index storage.
  loomc_allocator_t allocator;
  // Context retained by the frozen index.
  loomc_context_t* context;
  // Stable block pool backing text modules and index arena metadata.
  iree_arena_block_pool_t* block_pool;
  // Internal frozen module index.
  loom_link_module_index_t* index;
  // Sources retained to keep bytecode metadata and future materialization
  // alive.
  loomc_link_index_builder_source_t* sources;
  // Number of retained source records.
  loomc_host_size_t source_count;
};

static iree_allocator_t loomc_link_index_iree_allocator(
    loomc_allocator_t allocator) {
  return iree_allocator_from_loomc(allocator);
}

static loomc_status_t loomc_link_index_validate_builder_options(
    const loomc_link_index_builder_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LINK_INDEX_BUILDER_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link index builder options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "link index builder options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "link index builder option extensions are not supported");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_link_index_validate_source_options(
    const loomc_link_index_source_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->role > LOOMC_LINK_PROVIDER_ROLE_LIBRARY) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "unknown link provider role");
  }
  return loomc_ok_status();
}

static void loomc_link_index_block_pool_release(
    loomc_allocator_t allocator, iree_arena_block_pool_t* block_pool) {
  if (block_pool == NULL) {
    return;
  }
  iree_arena_block_pool_deinitialize(block_pool);
  loomc_allocator_free(allocator, block_pool);
}

static loomc_status_t loomc_link_index_builder_require_open(
    const loomc_link_index_builder_t* builder) {
  if (builder == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link index builder must not be NULL");
  }
  if (builder->finished) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "link index builder is already finished");
  }
  return loomc_ok_status();
}

static void loomc_link_index_builder_source_deinitialize(
    loomc_allocator_t allocator, loomc_link_index_builder_source_t* source) {
  loomc_source_release(source->source);
  loomc_allocator_free(allocator, (void*)source->provider_name.data);
  *source = (loomc_link_index_builder_source_t){0};
}

static loomc_status_t loomc_link_index_builder_reserve_sources(
    loomc_link_index_builder_t* builder, loomc_host_size_t required_count) {
  if (required_count <= builder->source_capacity) {
    return loomc_ok_status();
  }
  loomc_host_size_t new_capacity =
      builder->source_capacity == 0 ? 4 : builder->source_capacity * 2;
  while (new_capacity < required_count) {
    new_capacity *= 2;
  }
  loomc_link_index_builder_source_t* new_sources = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      builder->allocator, new_capacity * sizeof(*new_sources),
      (void**)&new_sources));
  if (builder->source_count != 0) {
    memcpy(new_sources, builder->sources,
           builder->source_count * sizeof(*new_sources));
  }
  memset(new_sources + builder->source_count, 0,
         (new_capacity - builder->source_count) * sizeof(*new_sources));
  loomc_allocator_free(builder->allocator, builder->sources);
  builder->sources = new_sources;
  builder->source_capacity = new_capacity;
  return loomc_ok_status();
}

static loom_link_provider_role_t loomc_link_provider_role_to_loom(
    loomc_link_provider_role_t role) {
  switch (role) {
    case LOOMC_LINK_PROVIDER_ROLE_INPUT:
      return LOOM_LINK_PROVIDER_ROLE_INPUT;
    case LOOMC_LINK_PROVIDER_ROLE_LIBRARY:
      return LOOM_LINK_PROVIDER_ROLE_LIBRARY;
  }
  return LOOM_LINK_PROVIDER_ROLE_INPUT;
}

static loomc_link_provider_role_t loomc_link_provider_role_from_loom(
    loom_link_provider_role_t role) {
  switch (role) {
    case LOOM_LINK_PROVIDER_ROLE_INPUT:
      return LOOMC_LINK_PROVIDER_ROLE_INPUT;
    case LOOM_LINK_PROVIDER_ROLE_LIBRARY:
      return LOOMC_LINK_PROVIDER_ROLE_LIBRARY;
  }
  return LOOMC_LINK_PROVIDER_ROLE_INPUT;
}

static loomc_link_provider_kind_t loomc_link_provider_kind_from_loom(
    loom_link_provider_kind_t kind) {
  switch (kind) {
    case LOOM_LINK_PROVIDER_MATERIALIZED:
      return LOOMC_LINK_PROVIDER_KIND_MATERIALIZED;
    case LOOM_LINK_PROVIDER_BYTECODE:
      return LOOMC_LINK_PROVIDER_KIND_BYTECODE;
    case LOOM_LINK_PROVIDER_TEXT:
      return LOOMC_LINK_PROVIDER_KIND_TEXT;
  }
  return LOOMC_LINK_PROVIDER_KIND_UNKNOWN;
}

static loomc_link_symbol_identity_t loomc_link_symbol_identity_from_loom(
    loom_link_symbol_identity_t identity) {
  switch (identity) {
    case LOOM_LINK_SYMBOL_IDENTITY_PRIVATE:
      return LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE;
    case LOOM_LINK_SYMBOL_IDENTITY_GLOBAL:
      return LOOMC_LINK_SYMBOL_IDENTITY_GLOBAL;
  }
  return LOOMC_LINK_SYMBOL_IDENTITY_PRIVATE;
}

static loomc_link_symbol_kind_t loomc_link_symbol_kind_from_loom(
    loom_symbol_kind_t kind) {
  switch (kind) {
    case LOOM_SYMBOL_FUNC_DEF:
      return LOOMC_LINK_SYMBOL_KIND_FUNCTION_DEFINITION;
    case LOOM_SYMBOL_FUNC_DECL:
      return LOOMC_LINK_SYMBOL_KIND_FUNCTION_DECLARATION;
    case LOOM_SYMBOL_FUNC_TEMPLATE:
      return LOOMC_LINK_SYMBOL_KIND_FUNCTION_TEMPLATE;
    case LOOM_SYMBOL_FUNC_UKERNEL:
      return LOOMC_LINK_SYMBOL_KIND_FUNCTION_UKERNEL;
    case LOOM_SYMBOL_GLOBAL:
      return LOOMC_LINK_SYMBOL_KIND_GLOBAL;
    case LOOM_SYMBOL_EXECUTABLE:
      return LOOMC_LINK_SYMBOL_KIND_EXECUTABLE;
    case LOOM_SYMBOL_RECORD:
      return LOOMC_LINK_SYMBOL_KIND_RECORD;
    case LOOM_SYMBOL_NONE:
      break;
  }
  return LOOMC_LINK_SYMBOL_KIND_UNKNOWN;
}

static loomc_link_symbol_flags_t loomc_link_symbol_flags_from_loom(
    loom_link_symbol_flags_t flags) {
  loomc_link_symbol_flags_t result = 0;
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_PUBLIC)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_PUBLIC;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_IMPORT)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_IMPORT;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_EXPORT)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_EXPORT;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_DECLARATION;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_HAS_BODY)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_HAS_BODY;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_CONFIG;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_CHECK_CASE)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_CHECK_CASE;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_CHECK_BENCHMARK)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_CHECK_BENCHMARK;
  }
  return result;
}

static bool loomc_link_index_source_is_bytecode(const loomc_source_t* source) {
  loomc_source_format_t format = loomc_source_format(source);
  if (format == LOOMC_SOURCE_FORMAT_BYTECODE) {
    return true;
  }
  if (format == LOOMC_SOURCE_FORMAT_TEXT) {
    return false;
  }
  loomc_byte_span_t contents = loomc_source_contents(source);
  return contents.data_length >= LOOM_BYTECODE_MAGIC_LENGTH &&
         memcmp(contents.data, LOOM_BYTECODE_MAGIC,
                LOOM_BYTECODE_MAGIC_LENGTH) == 0;
}

static loomc_status_t loomc_link_index_add_missing_source_diagnostic(
    loomc_link_index_builder_t* builder, loomc_host_size_t ordinal) {
  loomc_status_t status = loomc_status_allocate(
      LOOMC_STATUS_INVALID_ARGUMENT, __FILE__, __LINE__,
      loomc_make_cstring_view("reserved source slot is empty"));
  loomc_status_t add_status = loomc_result_add_status_diagnostic(
      builder->result, /*source=*/NULL, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
      loomc_make_cstring_view("LINK_INDEX/EMPTY_SLOT"), status);
  loomc_status_free(status);
  if (!loomc_status_is_ok(add_status)) {
    return add_status;
  }
  (void)ordinal;
  return loomc_ok_status();
}

typedef struct loomc_link_index_diagnostic_capture_t {
  // Result receiving converted diagnostics.
  loomc_result_t* result;
  // Source associated with emitted diagnostics.
  loomc_source_t* source;
} loomc_link_index_diagnostic_capture_t;

static iree_status_t loomc_link_index_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loomc_link_index_diagnostic_capture_t* capture =
      (loomc_link_index_diagnostic_capture_t*)user_data;
  return iree_status_from_loomc(loomc_result_add_loom_diagnostic(
      capture->result, capture->source, diagnostic));
}

static loomc_status_t loomc_link_index_mark_failed(loomc_result_t* result) {
  return loomc_result_set_state(result, LOOMC_RESULT_STATE_FAILED);
}

static loomc_status_t loomc_link_index_add_source_to_index(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_builder_source_t* source) {
  iree_host_size_t before_diagnostics =
      loomc_result_diagnostic_count(builder->result);
  loomc_link_index_diagnostic_capture_t capture = {
      .result = builder->result,
      .source = source->source,
  };
  loom_link_module_index_add_options_t options = {
      .provider_name = iree_string_view_from_loomc(source->provider_name),
      .role = loomc_link_provider_role_to_loom(source->role),
  };
  if (iree_string_view_is_empty(options.provider_name)) {
    options.provider_name =
        iree_string_view_from_loomc(loomc_source_identifier(source->source));
  }

  loomc_byte_span_t contents = loomc_source_contents(source->source);
  loomc_string_view_t identifier = loomc_source_identifier(source->source);
  iree_status_t status = iree_ok_status();
  if (loomc_link_index_source_is_bytecode(source->source)) {
    loom_bytecode_read_options_t read_options = {
        .diagnostic_sink =
            {
                .fn = loomc_link_index_capture_diagnostic,
                .user_data = &capture,
            },
    };
    status = loom_link_module_index_add_bytecode(
        builder->index,
        iree_make_const_byte_span(contents.data, contents.data_length),
        iree_string_view_from_loomc(identifier), &read_options, &options,
        /*out_provider_ordinal=*/NULL);
  } else {
    loom_text_parse_options_t parse_options = {
        .diagnostic_sink =
            {
                .fn = loomc_link_index_capture_diagnostic,
                .user_data = &capture,
            },
    };
    status = loom_link_module_index_add_text(
        builder->index,
        iree_make_string_view((const char*)contents.data, contents.data_length),
        iree_string_view_from_loomc(identifier), &parse_options, &options,
        /*out_provider_ordinal=*/NULL);
  }

  if (iree_status_is_ok(status)) {
    return loomc_ok_status();
  }

  loomc_status_t public_status = loomc_status_from_iree(status);
  if (loomc_result_diagnostic_count(builder->result) == before_diagnostics) {
    loomc_status_t add_status = loomc_result_add_status_diagnostic(
        builder->result, source->source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        loomc_make_cstring_view("LINK_INDEX/SOURCE"), public_status);
    if (!loomc_status_is_ok(add_status)) {
      loomc_status_free(public_status);
      return add_status;
    }
  }
  loomc_status_free(public_status);
  return loomc_link_index_mark_failed(builder->result);
}

static void loomc_link_index_destroy(loomc_link_index_t* link_index) {
  loomc_allocator_t allocator = link_index->allocator;
  loom_link_module_index_free(link_index->index);
  for (loomc_host_size_t i = 0; i < link_index->source_count; ++i) {
    loomc_link_index_builder_source_deinitialize(allocator,
                                                 &link_index->sources[i]);
  }
  loomc_allocator_free(allocator, link_index->sources);
  loomc_link_index_block_pool_release(allocator, link_index->block_pool);
  loomc_context_release(link_index->context);
  loomc_allocator_free(allocator, link_index);
}

static bool loomc_link_index_provider_from_loom(
    const loom_link_module_index_provider_t* source,
    loomc_link_index_provider_t* out_provider) {
  if (source == NULL || out_provider == NULL) {
    return false;
  }
  *out_provider = (loomc_link_index_provider_t){
      .ordinal = source->ordinal,
      .kind = loomc_link_provider_kind_from_loom(source->kind),
      .role = loomc_link_provider_role_from_loom(source->role),
      .name = loomc_string_view_from_iree(source->name),
      .module_start_ordinal = source->module_start_ordinal,
      .module_count = source->module_count,
  };
  return true;
}

static bool loomc_link_index_module_from_loom(
    const loom_link_module_index_module_t* source,
    loomc_link_index_module_t* out_module) {
  if (source == NULL || out_module == NULL) {
    return false;
  }
  *out_module = (loomc_link_index_module_t){
      .ordinal = source->ordinal,
      .provider_ordinal = source->provider_ordinal,
      .provider_module_ordinal = source->provider_module_ordinal,
      .name = loomc_string_view_from_iree(source->name),
      .symbol_start_ordinal = source->symbol_start_ordinal,
      .symbol_count = source->symbol_count,
  };
  return true;
}

static bool loomc_link_index_symbol_from_loom(
    const loom_link_module_index_symbol_t* source,
    loomc_link_index_symbol_t* out_symbol) {
  if (source == NULL || out_symbol == NULL) {
    return false;
  }
  *out_symbol = (loomc_link_index_symbol_t){
      .ordinal = source->ordinal,
      .provider_ordinal = source->provider_ordinal,
      .module_ordinal = source->module_ordinal,
      .provider_module_ordinal = source->provider_module_ordinal,
      .module_symbol_ordinal = source->module_symbol_ordinal,
      .name = loomc_string_view_from_iree(source->name),
      .kind = loomc_link_symbol_kind_from_loom(source->kind),
      .identity = loomc_link_symbol_identity_from_loom(source->identity),
      .flags = loomc_link_symbol_flags_from_loom(source->flags),
  };
  return true;
}

loomc_status_t loomc_link_index_builder_create(
    loomc_context_t* context, const loomc_link_index_builder_options_t* options,
    loomc_allocator_t allocator, loomc_link_index_builder_t** out_builder) {
  if (context == NULL || out_builder == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "context and out_builder must not be NULL");
  }
  *out_builder = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_link_index_validate_builder_options(options));

  loomc_link_index_builder_t* builder = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*builder), (void**)&builder));
  memset(builder, 0, sizeof(*builder));
  builder->allocator = allocator;
  builder->context = context;
  loomc_context_retain(context);

  iree_host_size_t block_size = LOOMC_LINK_INDEX_DEFAULT_BLOCK_SIZE;
  if (options && options->block_size != 0) {
    block_size = options->block_size;
  }
  loomc_status_t status = loomc_allocator_malloc(
      allocator, sizeof(*builder->block_pool), (void**)&builder->block_pool);
  if (loomc_status_is_ok(status)) {
    iree_arena_block_pool_initialize(block_size,
                                     loomc_link_index_iree_allocator(allocator),
                                     builder->block_pool);
  }

  if (loomc_status_is_ok(status)) {
    status = loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator,
                                 &builder->result);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_status_from_iree(loom_link_module_index_create(
        loomc_context_loom_context(context), builder->block_pool,
        loomc_link_index_iree_allocator(allocator), &builder->index));
  }
  if (loomc_status_is_ok(status)) {
    *out_builder = builder;
  } else {
    loomc_link_index_builder_release(builder);
  }
  return status;
}

void loomc_link_index_builder_release(loomc_link_index_builder_t* builder) {
  if (builder == NULL) {
    return;
  }
  loomc_allocator_t allocator = builder->allocator;
  for (loomc_host_size_t i = 0; i < builder->source_count; ++i) {
    loomc_link_index_builder_source_deinitialize(allocator,
                                                 &builder->sources[i]);
  }
  loomc_allocator_free(allocator, builder->sources);
  loomc_result_release(builder->result);
  if (builder->index != NULL) {
    loom_link_module_index_free(builder->index);
  }
  loomc_link_index_block_pool_release(allocator, builder->block_pool);
  loomc_context_release(builder->context);
  loomc_allocator_free(allocator, builder);
}

loomc_status_t loomc_link_index_builder_reserve_source_slot(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_source_options_t* options,
    loomc_link_index_source_slot_t* out_slot) {
  if (out_slot == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_slot must not be NULL");
  }
  *out_slot = (loomc_link_index_source_slot_t){
      .ordinal = LOOMC_HOST_SIZE_MAX,
  };
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  LOOMC_RETURN_IF_ERROR(loomc_link_index_validate_source_options(options));

  loomc_host_size_t ordinal = builder->source_count;
  LOOMC_RETURN_IF_ERROR(
      loomc_link_index_builder_reserve_sources(builder, ordinal + 1));
  loomc_link_index_builder_source_t* source = &builder->sources[ordinal];
  *source = (loomc_link_index_builder_source_t){
      .role = options ? options->role : LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  if (options) {
    LOOMC_RETURN_IF_ERROR(loomc_string_view_clone(
        options->provider_name, builder->allocator, &source->provider_name));
  }
  builder->source_count = ordinal + 1;
  *out_slot = (loomc_link_index_source_slot_t){
      .ordinal = ordinal,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_fill_source_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_source_slot_t slot,
    loomc_source_t* source) {
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  if (slot.ordinal >= builder->source_count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "link index source slot is out of range");
  }
  loomc_link_index_builder_source_t* target = &builder->sources[slot.ordinal];
  if (target->source != NULL) {
    return loomc_make_status(LOOMC_STATUS_ALREADY_EXISTS,
                             "link index source slot is already filled");
  }
  loomc_source_retain(source);
  target->source = source;
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_add_source(
    loomc_link_index_builder_t* builder, loomc_source_t* source,
    const loomc_link_index_source_options_t* options,
    loomc_link_index_source_slot_t* out_slot) {
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  loomc_link_index_source_slot_t slot = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_link_index_builder_reserve_source_slot(builder, options, &slot));
  loomc_status_t status =
      loomc_link_index_builder_fill_source_slot(builder, slot, source);
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  if (out_slot != NULL) {
    *out_slot = slot;
  }
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_finish(
    loomc_link_index_builder_t* builder, loomc_link_index_t** out_link_index,
    loomc_result_t** out_result) {
  if (out_link_index == NULL || out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_link_index and out_result must not be NULL");
  }
  *out_link_index = NULL;
  *out_result = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  builder->finished = true;

  for (loomc_host_size_t i = 0; i < builder->source_count; ++i) {
    const loomc_link_index_builder_source_t* source = &builder->sources[i];
    if (source->source == NULL) {
      LOOMC_RETURN_IF_ERROR(
          loomc_link_index_add_missing_source_diagnostic(builder, i));
      LOOMC_RETURN_IF_ERROR(loomc_link_index_mark_failed(builder->result));
      continue;
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_link_index_add_source_to_index(builder, source));
  }

  if (!loomc_result_succeeded(builder->result)) {
    *out_result = builder->result;
    builder->result = NULL;
    return loomc_ok_status();
  }

  loomc_link_index_t* link_index = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      builder->allocator, sizeof(*link_index), (void**)&link_index));
  memset(link_index, 0, sizeof(*link_index));
  iree_atomic_ref_count_init(&link_index->ref_count);
  link_index->allocator = builder->allocator;
  link_index->context = builder->context;
  link_index->block_pool = builder->block_pool;
  link_index->index = builder->index;
  link_index->sources = builder->sources;
  link_index->source_count = builder->source_count;

  builder->context = NULL;
  builder->block_pool = NULL;
  builder->index = NULL;
  builder->sources = NULL;
  builder->source_count = 0;
  builder->source_capacity = 0;
  *out_link_index = link_index;
  *out_result = builder->result;
  builder->result = NULL;
  return loomc_ok_status();
}

void loomc_link_index_retain(loomc_link_index_t* link_index) {
  if (link_index == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&link_index->ref_count);
}

void loomc_link_index_release(loomc_link_index_t* link_index) {
  if (link_index == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&link_index->ref_count) == 1) {
    loomc_link_index_destroy(link_index);
  }
}

loomc_host_size_t loomc_link_index_provider_count(
    const loomc_link_index_t* link_index) {
  return link_index ? loom_link_module_index_provider_count(link_index->index)
                    : 0;
}

bool loomc_link_index_provider_at(const loomc_link_index_t* link_index,
                                  loomc_host_size_t ordinal,
                                  loomc_link_index_provider_t* out_provider) {
  if (link_index == NULL) {
    return false;
  }
  return loomc_link_index_provider_from_loom(
      loom_link_module_index_provider_at(link_index->index, ordinal),
      out_provider);
}

loomc_host_size_t loomc_link_index_module_count(
    const loomc_link_index_t* link_index) {
  return link_index ? loom_link_module_index_module_count(link_index->index)
                    : 0;
}

bool loomc_link_index_module_at(const loomc_link_index_t* link_index,
                                loomc_host_size_t ordinal,
                                loomc_link_index_module_t* out_module) {
  if (link_index == NULL) {
    return false;
  }
  return loomc_link_index_module_from_loom(
      loom_link_module_index_module_at(link_index->index, ordinal), out_module);
}

loomc_host_size_t loomc_link_index_symbol_count(
    const loomc_link_index_t* link_index) {
  return link_index ? loom_link_module_index_symbol_count(link_index->index)
                    : 0;
}

bool loomc_link_index_symbol_at(const loomc_link_index_t* link_index,
                                loomc_host_size_t ordinal,
                                loomc_link_index_symbol_t* out_symbol) {
  if (link_index == NULL) {
    return false;
  }
  return loomc_link_index_symbol_from_loom(
      loom_link_module_index_symbol_at(link_index->index, ordinal), out_symbol);
}

bool loomc_link_index_lookup_global(const loomc_link_index_t* link_index,
                                    loomc_string_view_t name,
                                    loomc_link_index_symbol_t* out_symbol) {
  if (link_index == NULL) {
    return false;
  }
  return loomc_link_index_symbol_from_loom(
      loom_link_module_index_lookup_global(link_index->index,
                                           iree_string_view_from_loomc(name)),
      out_symbol);
}

bool loomc_link_index_next_global_duplicate(
    const loomc_link_index_t* link_index,
    const loomc_link_index_symbol_t* symbol,
    loomc_link_index_symbol_t* out_symbol) {
  if (link_index == NULL || symbol == NULL ||
      symbol->ordinal >=
          loom_link_module_index_symbol_count(link_index->index)) {
    return false;
  }
  const loom_link_module_index_symbol_t* internal_symbol =
      loom_link_module_index_symbol_at(link_index->index, symbol->ordinal);
  return loomc_link_index_symbol_from_loom(
      loom_link_module_index_next_global_duplicate(link_index->index,
                                                   internal_symbol),
      out_symbol);
}

bool loomc_link_index_lookup_private(const loomc_link_index_t* link_index,
                                     const loomc_link_index_module_t* module,
                                     loomc_string_view_t name,
                                     loomc_link_index_symbol_t* out_symbol) {
  if (link_index == NULL || module == NULL ||
      module->ordinal >=
          loom_link_module_index_module_count(link_index->index)) {
    return false;
  }
  const loom_link_module_index_module_t* internal_module =
      loom_link_module_index_module_at(link_index->index, module->ordinal);
  return loomc_link_index_symbol_from_loom(
      loom_link_module_index_lookup_private(link_index->index, internal_module,
                                            iree_string_view_from_loomc(name)),
      out_symbol);
}

loomc_context_t* loomc_link_index_context(
    const loomc_link_index_t* link_index) {
  return link_index ? link_index->context : NULL;
}

const loom_link_module_index_t* loomc_link_index_module_index(
    const loomc_link_index_t* link_index) {
  return link_index ? link_index->index : NULL;
}

const loomc_source_t* loomc_link_index_source_for_provider(
    const loomc_link_index_t* link_index, loomc_host_size_t provider_ordinal) {
  if (link_index == NULL || provider_ordinal >= link_index->source_count) {
    return NULL;
  }
  return link_index->sources[provider_ordinal].source;
}
