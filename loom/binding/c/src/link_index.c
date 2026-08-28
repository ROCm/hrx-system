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
#include "loom/format/bytecode/index.h"
#include "loom/format/text/parser.h"
#include "loom/ir/ir.h"
#include "loomc/iree.h"
#include "module.h"
#include "result.h"
#include "source.h"
#include "target.h"

enum {
  LOOMC_LINK_INDEX_DEFAULT_BLOCK_SIZE = 32 * 1024,
};

typedef enum loomc_link_index_builder_provider_kind_e {
  // Reserved provider slot that has not been filled.
  LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY = 0,
  // Provider backed by an immutable source handle.
  LOOMC_LINK_INDEX_BUILDER_PROVIDER_SOURCE = 1,
  // Provider backed by an already materialized module.
  LOOMC_LINK_INDEX_BUILDER_PROVIDER_MODULE = 2,
} loomc_link_index_builder_provider_kind_t;

typedef struct loomc_link_index_builder_provider_t {
  // Representation currently filling this provider slot.
  loomc_link_index_builder_provider_kind_t kind;
  // Retained provider payload selected by |kind|.
  union {
    // Source retained until builder or frozen-index release.
    loomc_source_t* source;
    // Module retained until builder or frozen-index release.
    loomc_module_t* module;
  } payload;
  // Copied provider label.
  loomc_string_view_t name;
  // Provider linkage role.
  loomc_link_provider_role_t role;
} loomc_link_index_builder_provider_t;

struct loomc_link_index_builder_t {
  // Allocator used for builder storage.
  loomc_allocator_t allocator;
  // Context retained while indexing.
  loomc_context_t* context;
  // Mutable provider slot storage.
  struct {
    // Provider slot records.
    loomc_link_index_builder_provider_t* values;
    // Number of reserved provider slots.
    loomc_host_size_t count;
    // Allocated provider slot capacity.
    loomc_host_size_t capacity;
  } providers;
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
  // Providers retained to keep source bytes or materialized modules alive.
  struct {
    // Retained provider records.
    loomc_link_index_builder_provider_t* values;
    // Number of retained provider records.
    loomc_host_size_t count;
  } providers;
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
  if (options->block_size != 0) {
    if (options->block_size < sizeof(iree_arena_block_t)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "link index builder block_size is too small");
    }
    if (!iree_arena_block_pool_is_valid_total_size(options->block_size)) {
      return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                               "link index builder block_size is too large");
    }
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_link_index_validate_provider_options(
    const loomc_link_index_provider_options_t* options) {
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

static void loomc_link_index_builder_provider_deinitialize(
    loomc_allocator_t allocator,
    loomc_link_index_builder_provider_t* provider) {
  switch (provider->kind) {
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_SOURCE:
      loomc_source_release(provider->payload.source);
      break;
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_MODULE:
      loomc_module_release(provider->payload.module);
      break;
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY:
      break;
  }
  loomc_allocator_free(allocator, (void*)provider->name.data);
  *provider = (loomc_link_index_builder_provider_t){0};
}

static loomc_status_t loomc_link_index_builder_provider_initialize(
    const loomc_link_index_provider_options_t* options,
    loomc_allocator_t allocator,
    loomc_link_index_builder_provider_t* out_provider) {
  loomc_link_index_builder_provider_t provider = {
      .role = options ? options->role : LOOMC_LINK_PROVIDER_ROLE_INPUT,
  };
  loomc_status_t status = loomc_ok_status();
  if (options != NULL) {
    status = loomc_string_view_clone(options->provider_name, allocator,
                                     &provider.name);
  }
  if (loomc_status_is_ok(status)) {
    *out_provider = provider;
  } else {
    loomc_link_index_builder_provider_deinitialize(allocator, &provider);
  }
  return status;
}

static loomc_status_t loomc_link_index_builder_reserve_providers(
    loomc_link_index_builder_t* builder, loomc_host_size_t required_count) {
  if (required_count <= builder->providers.capacity) {
    return loomc_ok_status();
  }
  const loomc_host_size_t old_capacity = builder->providers.capacity;
  const loomc_host_size_t minimum_capacity = iree_max(required_count, 4u);
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(iree_allocator_grow_array(
      loomc_link_index_iree_allocator(builder->allocator), minimum_capacity,
      sizeof(*builder->providers.values), &builder->providers.capacity,
      (void**)&builder->providers.values)));
  memset(builder->providers.values + old_capacity, 0,
         (builder->providers.capacity - old_capacity) *
             sizeof(*builder->providers.values));
  return loomc_ok_status();
}

static void loomc_link_index_builder_provider_set_source(
    loomc_link_index_builder_provider_t* provider, loomc_source_t* source) {
  loomc_source_retain(source);
  provider->payload.source = source;
  provider->kind = LOOMC_LINK_INDEX_BUILDER_PROVIDER_SOURCE;
}

static void loomc_link_index_builder_provider_set_module(
    loomc_link_index_builder_provider_t* provider, loomc_module_t* module) {
  loomc_module_retain(module);
  provider->payload.module = module;
  provider->kind = LOOMC_LINK_INDEX_BUILDER_PROVIDER_MODULE;
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
    case LOOM_SYMBOL_TEMPLATE_DEF:
      return LOOMC_LINK_SYMBOL_KIND_FUNCTION_TEMPLATE;
    case LOOM_SYMBOL_TEMPLATE_UKERNEL:
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
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_CONFIG;
  }
  if (iree_all_bits_set(flags, LOOM_LINK_SYMBOL_FLAG_TEST_ONLY)) {
    result |= LOOMC_LINK_SYMBOL_FLAG_TEST_ONLY;
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

static loomc_status_t loomc_link_index_add_empty_provider_diagnostic(
    loomc_link_index_builder_t* builder, loomc_host_size_t ordinal) {
  loomc_status_t status = loomc_status_allocate(
      LOOMC_STATUS_INVALID_ARGUMENT, __FILE__, __LINE__,
      loomc_make_cstring_view("reserved provider slot is empty"));
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

static loomc_status_t loomc_link_index_add_provider_to_index(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_builder_provider_t* provider) {
  iree_host_size_t before_diagnostics =
      loomc_result_diagnostic_count(builder->result);
  loom_link_module_index_add_options_t options = {
      .provider_name = iree_string_view_from_loomc(provider->name),
      .role = loomc_link_provider_role_to_loom(provider->role),
  };

  iree_status_t status = iree_ok_status();
  const loomc_source_t* diagnostic_source = NULL;
  loomc_string_view_t diagnostic_code = loomc_string_view_empty();
  switch (provider->kind) {
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_SOURCE: {
      loomc_source_t* source = provider->payload.source;
      diagnostic_source = source;
      diagnostic_code = loomc_make_cstring_view("LINK_INDEX/SOURCE");
      loomc_link_index_diagnostic_capture_t capture = {
          .result = builder->result,
          .source = source,
      };
      if (iree_string_view_is_empty(options.provider_name)) {
        options.provider_name =
            iree_string_view_from_loomc(loomc_source_identifier(source));
      }
      const loomc_byte_span_t contents = loomc_source_contents(source);
      const loomc_string_view_t identifier = loomc_source_identifier(source);
      if (loomc_link_index_source_is_bytecode(source)) {
        const loom_bytecode_index_options_t index_options = {
            .diagnostic_sink =
                {
                    .fn = loomc_link_index_capture_diagnostic,
                    .user_data = &capture,
                },
        };
        status = loom_link_module_index_add_bytecode(
            builder->index,
            iree_make_const_byte_span(contents.data, contents.data_length),
            iree_string_view_from_loomc(identifier), &index_options, &options,
            /*out_provider_ordinal=*/NULL);
      } else {
        loom_text_parse_options_t parse_options = {
            .diagnostic_sink =
                {
                    .fn = loomc_link_index_capture_diagnostic,
                    .user_data = &capture,
                },
        };
        loomc_target_pass_environment_initialize_text_asm_environment(
            loomc_context_target_pass_environment(builder->context),
            &parse_options.low_asm_environment);
        status = loom_link_module_index_add_text(
            builder->index,
            iree_make_string_view((const char*)contents.data,
                                  contents.data_length),
            iree_string_view_from_loomc(identifier), &parse_options, &options,
            /*out_provider_ordinal=*/NULL);
      }
      break;
    }
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_MODULE:
      diagnostic_code = loomc_make_cstring_view("LINK_INDEX/MODULE");
      status = loom_link_module_index_add_materialized(
          builder->index,
          loomc_module_const_loom_module(provider->payload.module), &options,
          /*out_provider_ordinal=*/NULL);
      break;
    case LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY:
      IREE_ASSERT(false && "empty provider reached index construction");
      break;
  }

  if (iree_status_is_ok(status)) {
    return loomc_ok_status();
  }

  loomc_status_t public_status = loomc_status_from_iree(status);
  if (loomc_result_diagnostic_count(builder->result) == before_diagnostics) {
    loomc_status_t add_status = loomc_result_add_status_diagnostic(
        builder->result, diagnostic_source, LOOMC_DIAGNOSTIC_SEVERITY_ERROR,
        diagnostic_code, public_status);
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
  for (loomc_host_size_t i = 0; i < link_index->providers.count; ++i) {
    loomc_link_index_builder_provider_deinitialize(
        allocator, &link_index->providers.values[i]);
  }
  loomc_allocator_free(allocator, link_index->providers.values);
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
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* source,
    loomc_link_index_symbol_t* out_symbol) {
  if (index == NULL || source == NULL || out_symbol == NULL) {
    return false;
  }
  const loom_link_module_index_module_t* module =
      loom_link_module_index_symbol_module(index, source);
  if (module == NULL) {
    return false;
  }
  *out_symbol = (loomc_link_index_symbol_t){
      .ordinal = source->ordinal,
      .provider_ordinal = module->provider_ordinal,
      .module_ordinal = source->module_ordinal,
      .provider_module_ordinal = module->provider_module_ordinal,
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
    status = loomc_status_from_iree(loom_link_module_index_allocate(
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
  for (loomc_host_size_t i = 0; i < builder->providers.count; ++i) {
    loomc_link_index_builder_provider_deinitialize(
        allocator, &builder->providers.values[i]);
  }
  loomc_allocator_free(allocator, builder->providers.values);
  loomc_result_release(builder->result);
  if (builder->index != NULL) {
    loom_link_module_index_free(builder->index);
  }
  loomc_link_index_block_pool_release(allocator, builder->block_pool);
  loomc_context_release(builder->context);
  loomc_allocator_free(allocator, builder);
}

loomc_status_t loomc_link_index_builder_reserve_provider_slot(
    loomc_link_index_builder_t* builder,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot) {
  if (out_slot == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_slot must not be NULL");
  }
  *out_slot = (loomc_link_index_provider_slot_t){
      .ordinal = LOOMC_HOST_SIZE_MAX,
  };
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  LOOMC_RETURN_IF_ERROR(loomc_link_index_validate_provider_options(options));

  const loomc_host_size_t ordinal = builder->providers.count;
  loomc_host_size_t required_count = 0;
  if (!iree_host_size_checked_add(ordinal, 1, &required_count)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "link index provider count overflow");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_link_index_builder_reserve_providers(builder, required_count));
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_provider_initialize(
      options, builder->allocator, &builder->providers.values[ordinal]));
  builder->providers.count = required_count;
  *out_slot = (loomc_link_index_provider_slot_t){
      .ordinal = ordinal,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_fill_source_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_provider_slot_t slot,
    loomc_source_t* source) {
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  if (slot.ordinal >= builder->providers.count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "link index provider slot is out of range");
  }
  loomc_link_index_builder_provider_t* target =
      &builder->providers.values[slot.ordinal];
  if (target->kind != LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY) {
    return loomc_make_status(LOOMC_STATUS_ALREADY_EXISTS,
                             "link index provider slot is already filled");
  }
  loomc_link_index_builder_provider_set_source(target, source);
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_fill_module_slot(
    loomc_link_index_builder_t* builder, loomc_link_index_provider_slot_t slot,
    loomc_module_t* module) {
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_link_index_builder_require_open(builder));
  if (loomc_module_context(module) != builder->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module was created with another context");
  }
  if (slot.ordinal >= builder->providers.count) {
    return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                             "link index provider slot is out of range");
  }
  loomc_link_index_builder_provider_t* target =
      &builder->providers.values[slot.ordinal];
  if (target->kind != LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY) {
    return loomc_make_status(LOOMC_STATUS_ALREADY_EXISTS,
                             "link index provider slot is already filled");
  }
  loomc_link_index_builder_provider_set_module(target, module);
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_add_source(
    loomc_link_index_builder_t* builder, loomc_source_t* source,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot) {
  if (source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "source must not be NULL");
  }
  loomc_link_index_provider_slot_t slot = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_link_index_builder_reserve_provider_slot(builder, options, &slot));
  loomc_link_index_builder_provider_set_source(
      &builder->providers.values[slot.ordinal], source);
  if (out_slot != NULL) {
    *out_slot = slot;
  }
  return loomc_ok_status();
}

loomc_status_t loomc_link_index_builder_add_module(
    loomc_link_index_builder_t* builder, loomc_module_t* module,
    const loomc_link_index_provider_options_t* options,
    loomc_link_index_provider_slot_t* out_slot) {
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module must not be NULL");
  }
  if (builder == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "link index builder must not be NULL");
  }
  if (loomc_module_context(module) != builder->context) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "module was created with another context");
  }
  loomc_link_index_provider_slot_t slot = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_link_index_builder_reserve_provider_slot(builder, options, &slot));
  loomc_link_index_builder_provider_set_module(
      &builder->providers.values[slot.ordinal], module);
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

  for (loomc_host_size_t i = 0; i < builder->providers.count; ++i) {
    const loomc_link_index_builder_provider_t* provider =
        &builder->providers.values[i];
    if (provider->kind == LOOMC_LINK_INDEX_BUILDER_PROVIDER_EMPTY) {
      LOOMC_RETURN_IF_ERROR(
          loomc_link_index_add_empty_provider_diagnostic(builder, i));
      LOOMC_RETURN_IF_ERROR(loomc_link_index_mark_failed(builder->result));
      continue;
    }
    LOOMC_RETURN_IF_ERROR(
        loomc_link_index_add_provider_to_index(builder, provider));
  }

  if (!loomc_result_succeeded(builder->result)) {
    *out_result = builder->result;
    builder->result = NULL;
    return loomc_ok_status();
  }

  loomc_link_index_t* link_index = NULL;
  loomc_status_t status = loomc_allocator_malloc(
      builder->allocator, sizeof(*link_index), (void**)&link_index);
  if (!loomc_status_is_ok(status)) {
    return status;
  }
  memset(link_index, 0, sizeof(*link_index));
  iree_atomic_ref_count_init(&link_index->ref_count);
  link_index->allocator = builder->allocator;
  link_index->context = builder->context;
  link_index->block_pool = builder->block_pool;
  link_index->index = builder->index;
  link_index->providers.values = builder->providers.values;
  link_index->providers.count = builder->providers.count;

  builder->context = NULL;
  builder->block_pool = NULL;
  builder->index = NULL;
  builder->providers.values = NULL;
  builder->providers.count = 0;
  builder->providers.capacity = 0;
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
      link_index->index,
      loom_link_module_index_symbol_at(link_index->index, ordinal), out_symbol);
}

bool loomc_link_index_lookup_global(const loomc_link_index_t* link_index,
                                    loomc_string_view_t name,
                                    loomc_link_index_symbol_t* out_symbol) {
  if (link_index == NULL) {
    return false;
  }
  return loomc_link_index_symbol_from_loom(
      link_index->index,
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
      link_index->index,
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
      link_index->index,
      loom_link_module_index_lookup_private(link_index->index, internal_module,
                                            iree_string_view_from_loomc(name)),
      out_symbol);
}

loomc_context_t* loomc_link_index_context(
    const loomc_link_index_t* link_index) {
  return link_index ? link_index->context : NULL;
}

loomc_allocator_t loomc_link_index_allocator(
    const loomc_link_index_t* link_index) {
  IREE_ASSERT_ARGUMENT(link_index);
  return link_index->allocator;
}

const loom_link_module_index_t* loomc_link_index_module_index(
    const loomc_link_index_t* link_index) {
  return link_index ? link_index->index : NULL;
}

const loomc_source_t* loomc_link_index_source_for_provider(
    const loomc_link_index_t* link_index, loomc_host_size_t provider_ordinal) {
  if (link_index == NULL || provider_ordinal >= link_index->providers.count) {
    return NULL;
  }
  const loomc_link_index_builder_provider_t* provider =
      &link_index->providers.values[provider_ordinal];
  return provider->kind == LOOMC_LINK_INDEX_BUILDER_PROVIDER_SOURCE
             ? provider->payload.source
             : NULL;
}
