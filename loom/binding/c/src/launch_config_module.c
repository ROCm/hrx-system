// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config_module.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "context.h"
#include "iree/base/alignment.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "loom/analysis/symbol_value_constraints.h"
#include "loom/format/bytecode/reader.h"
#include "loom/ir/attribute.h"
#include "loom/ir/facts.h"
#include "loom/ir/float_facts.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/pass/value_facts.h"
#include "loomc/iree.h"

enum {
  // Small host artifacts should remain small while retaining enough room for
  // decoded metadata and ordinary launch arithmetic.
  LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE = 4 * 1024,

  // One block owns the direct value table and one remains available for
  // touched-value or transient fact storage during steady evaluation.
  LOOMC_LAUNCH_CONFIG_CONTEXT_PREALLOCATED_BLOCK_COUNT = 2,
};

static_assert(sizeof(loomc_dimension3_t) == 3 * sizeof(uint32_t),
              "launch count tuples must be three packed u32 values");
static_assert(offsetof(loomc_dimension3_t, x) == 0 &&
                  offsetof(loomc_dimension3_t, y) == sizeof(uint32_t) &&
                  offsetof(loomc_dimension3_t, z) == 2 * sizeof(uint32_t),
              "launch count tuple fields must be densely packed");

typedef struct loomc_launch_config_function_storage_t {
  // Prepared public function borrowed from the immutable module.
  loom_func_like_t function;

  // Public function name borrowed from the immutable module.
  iree_string_view_t name;

  // Positional function argument value IDs.
  const loom_value_id_t* argument_ids;

  // Positional returned value IDs.
  const loom_value_id_t* result_ids;

  // Number of entries in argument_ids.
  uint16_t argument_count;

  // Number of entries in result_ids.
  uint16_t result_count;
} loomc_launch_config_function_storage_t;

typedef struct loomc_launch_config_module_resolved_load_options_t {
  // Storage policy for artifact contents.
  loomc_source_storage_t storage;

  // Callback used to release externally owned contents.
  loomc_source_release_fn_t release;

  // User data passed to release.
  void* release_user_data;
} loomc_launch_config_module_resolved_load_options_t;

struct loomc_launch_config_module_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for handle and metadata storage.
  loomc_allocator_t allocator;

  // Immutable language context retained by the loaded module.
  loomc_context_t* context;

  // Arena block pool owning decoded module IR.
  iree_arena_block_pool_t block_pool;

  // Verified evaluation-ready host module.
  loom_module_t* module;

  // Dense exported launch-function table.
  loomc_launch_config_function_storage_t* functions;

  // Number of entries in functions.
  iree_host_size_t function_count;

  // Function ordinal for each module symbol, or UINT32_MAX when the symbol is
  // not an exported launch function.
  uint32_t* symbol_function_ordinals;
};

struct loomc_launch_config_context_t {
  // Atomic reference count for explicit shared ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release context storage.
  loomc_allocator_t allocator;

  // Immutable launch module retained by the context.
  loomc_launch_config_module_t* module;

  // Preallocated block pool for reusable evaluator state.
  iree_arena_block_pool_t block_pool;

  // Reusable value-fact analysis state.
  loom_pass_value_fact_owner_t fact_owner;
};

static bool loomc_launch_config_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_launch_config_module_resolve_load_options(
    const loomc_launch_config_module_load_options_t* options,
    loomc_launch_config_module_resolved_load_options_t* out_options) {
  *out_options = (loomc_launch_config_module_resolved_load_options_t){
      .storage = LOOMC_SOURCE_STORAGE_BORROWED,
  };
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_MODULE_LOAD_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module load options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module load options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config module load option extensions are not supported");
  }
  if (options->storage > LOOMC_SOURCE_STORAGE_EXTERNAL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module artifact storage policy is invalid");
  }
  if (options->storage == LOOMC_SOURCE_STORAGE_EXTERNAL) {
    if (options->release == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "external launch config artifact storage requires a release");
    }
  } else if (options->release != NULL || options->release_user_data != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "only external launch config artifact storage accepts a release");
  }
  *out_options = (loomc_launch_config_module_resolved_load_options_t){
      .storage = options->storage,
      .release = options->release,
      .release_user_data = options->release_user_data,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_module_block_size(
    loomc_host_size_t artifact_length, iree_host_size_t* out_block_size) {
  *out_block_size = 0;
  const iree_host_size_t maximum_scaled_length = IREE_HOST_SIZE_MAX / 4;
  if (artifact_length > maximum_scaled_length) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact is too large to materialize");
  }
  iree_host_size_t requested_size = artifact_length * 4;
  requested_size =
      iree_max(requested_size,
               (iree_host_size_t)LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE);
  const iree_host_size_t block_size =
      iree_host_size_next_power_of_two(requested_size);
  if (!iree_arena_block_pool_is_valid_total_size(block_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact requires an unsupported arena block size");
  }
  *out_block_size = block_size;
  return loomc_ok_status();
}

static void loomc_launch_config_module_destroy(
    loomc_launch_config_module_t* module) {
  loomc_allocator_t allocator = module->allocator;
  loomc_allocator_free(allocator, module->symbol_function_ordinals);
  loomc_allocator_free(allocator, module->functions);
  loom_module_free(module->module);
  iree_arena_block_pool_deinitialize(&module->block_pool);
  loomc_context_release(module->context);
  loomc_allocator_free(allocator, module);
}

static iree_string_view_t loomc_launch_config_function_name(
    const loom_module_t* module, loom_func_like_t function) {
  const loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return module->strings.entries[symbol->name_id];
}

static loomc_status_t loomc_launch_config_module_bind_function(
    const loom_module_t* module, loom_func_like_t function,
    loomc_launch_config_function_storage_t* out_function) {
  const iree_string_view_t name =
      loomc_launch_config_function_name(module, function);
  if (!loom_func_def_isa(function.op)) {
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "launch config function '@%.*s' must be a func.def",
                         (int)name.size, name.data));
  }
  if (loom_func_like_purity(function) != LOOM_FUNC_PURITY_PURE) {
    return loomc_status_from_iree(
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "launch config function '@%.*s' must be pure",
                         (int)name.size, name.data));
  }
  const uint8_t calling_convention = loom_func_like_cc(function);
  if (calling_convention != 0 && calling_convention != LOOM_FUNC_CC_HOST) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' must use the host calling convention",
        (int)name.size, name.data));
  }

  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(function, &argument_count);
  for (uint16_t i = 0; i < argument_count; ++i) {
    if (!loom_type_is_scalar(loom_module_value_type(module, argument_ids[i]))) {
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "launch config function '@%.*s' argument %u must be scalar",
          (int)name.size, name.data, (unsigned)i));
    }
  }

  loom_region_t* body = loom_func_like_body(function);
  if (body == NULL || body->block_count != 1 ||
      iree_any_bit_set(body->flags, LOOM_REGION_INSTANCE_FLAG_CFG)) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' must have one non-CFG entry block",
        (int)name.size, name.data));
  }
  const loom_block_t* block = loom_region_const_entry_block(body);
  if (block->last_op == NULL || !loom_func_return_isa(block->last_op)) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' must end in func.return",
        (int)name.size, name.data));
  }
  const loom_value_slice_t results = loom_func_return_operands(block->last_op);
  if (results.count % 3 != 0) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' must return complete xyz tuples",
        (int)name.size, name.data));
  }
  for (uint16_t i = 0; i < results.count; ++i) {
    const loom_type_t type = loom_module_value_type(module, results.values[i]);
    if (!loom_type_is_scalar(type) ||
        loom_type_element_type(type) != LOOM_SCALAR_TYPE_INDEX) {
      return loomc_status_from_iree(iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "launch config function '@%.*s' result %u must be index",
          (int)name.size, name.data, (unsigned)i));
    }
  }

  *out_function = (loomc_launch_config_function_storage_t){
      .function = function,
      .name = name,
      .argument_ids = argument_ids,
      .result_ids = results.values,
      .argument_count = argument_count,
      .result_count = results.count,
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_module_bind_functions(
    loomc_launch_config_module_t* module) {
  const loom_module_t* internal_module = module->module;
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < internal_module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &internal_module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(internal_module, symbol->defining_op);
    if (loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
      continue;
    }
    ++function_count;
  }
  if (function_count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config artifact contains no public functions");
  }
  if (function_count > UINT32_MAX) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config artifact contains too many public functions");
  }

  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      module->allocator, function_count * sizeof(*module->functions),
      (void**)&module->functions));
  memset(module->functions, 0, function_count * sizeof(*module->functions));
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(module->allocator,
                             internal_module->symbols.count *
                                 sizeof(*module->symbol_function_ordinals),
                             (void**)&module->symbol_function_ordinals));
  memset(module->symbol_function_ordinals, 0xFF,
         internal_module->symbols.count *
             sizeof(*module->symbol_function_ordinals));

  iree_host_size_t function_ordinal = 0;
  for (iree_host_size_t symbol_ordinal = 0;
       symbol_ordinal < internal_module->symbols.count; ++symbol_ordinal) {
    const loom_symbol_t* symbol =
        &internal_module->symbols.entries[symbol_ordinal];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(internal_module, symbol->defining_op);
    if (loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
      continue;
    }

    loomc_launch_config_function_storage_t* function_storage =
        &module->functions[function_ordinal];
    LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_bind_function(
        internal_module, function, function_storage));
    module->symbol_function_ordinals[symbol_ordinal] =
        (uint32_t)function_ordinal;
    ++function_ordinal;
  }
  module->function_count = function_count;
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_module_load_impl(
    const loomc_artifact_t* artifact, loomc_allocator_t allocator,
    loomc_launch_config_module_t** out_module) {
  if (artifact == NULL || out_module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact and out_module must not be NULL");
  }
  *out_module = NULL;
  if (artifact->kind != LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "artifact kind is not LOOMC_ARTIFACT_KIND_LAUNCH_CONFIG");
  }
  if (!loomc_launch_config_string_view_is_well_formed(artifact->format) ||
      !loomc_launch_config_string_view_is_well_formed(artifact->identifier)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact string view is malformed");
  }
  if (!loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE))) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "launch config artifact format '%.*s' is not supported",
        (int)artifact->format.size, artifact->format.data));
  }
  if (artifact->contents.data == NULL && artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }
  iree_host_size_t module_block_size = 0;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_block_size(
      artifact->contents.data_length, &module_block_size));

  loomc_launch_config_module_t* module = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*module), (void**)&module));
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->allocator = allocator;
  loomc_status_t status = loomc_context_create(
      /*options=*/NULL, allocator, &module->context);
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, module);
    return status;
  }
  iree_arena_block_pool_initialize(module_block_size,
                                   iree_allocator_from_loomc(allocator),
                                   &module->block_pool);

  loom_bytecode_read_options_t read_options = {
      .verify_module = true,
  };
  loom_bytecode_read_result_t read_result = {0};
  iree_string_view_t identifier =
      iree_string_view_from_loomc(artifact->identifier);
  if (iree_string_view_is_empty(identifier)) {
    identifier = IREE_SV("launch_config.loombc");
  }
  status = loomc_status_from_iree(loom_bytecode_read_module(
      iree_make_const_byte_span(artifact->contents.data,
                                artifact->contents.data_length),
      identifier, loomc_context_loom_context(module->context),
      &module->block_pool, &read_options, &read_result, &module->module,
      iree_allocator_from_loomc(allocator)));
  if (loomc_status_is_ok(status) &&
      (read_result.error_count != 0 || module->module == NULL)) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config artifact is not valid verified Loom bytecode");
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_module_bind_functions(module);
  }
  if (loomc_status_is_ok(status)) {
    *out_module = module;
  } else {
    loomc_launch_config_module_destroy(module);
  }
  return status;
}

loomc_status_t loomc_launch_config_module_load(
    const loomc_artifact_t* artifact,
    const loomc_launch_config_module_load_options_t* options,
    loomc_allocator_t allocator, loomc_launch_config_module_t** out_module) {
  loomc_launch_config_module_resolved_load_options_t resolved_options;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_load_options(
      options, &resolved_options));

  const loomc_byte_span_t contents =
      artifact ? artifact->contents : loomc_byte_span_empty();
  loomc_status_t status =
      loomc_launch_config_module_load_impl(artifact, allocator, out_module);
  if (resolved_options.storage == LOOMC_SOURCE_STORAGE_EXTERNAL) {
    resolved_options.release(resolved_options.release_user_data, contents);
  }
  return status;
}

void loomc_launch_config_module_retain(loomc_launch_config_module_t* module) {
  if (module == NULL) return;
  iree_atomic_ref_count_inc(&module->ref_count);
}

void loomc_launch_config_module_release(loomc_launch_config_module_t* module) {
  if (module == NULL) return;
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    loomc_launch_config_module_destroy(module);
  }
}

loomc_host_size_t loomc_launch_config_module_function_count(
    const loomc_launch_config_module_t* module) {
  return module ? module->function_count : 0;
}

static loomc_status_t loomc_launch_config_module_resolve_function(
    const loomc_launch_config_module_t* module,
    loomc_launch_config_function_t function,
    const loomc_launch_config_function_storage_t** out_function) {
  *out_function = NULL;
  if (module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config module must not be NULL");
  }
  if (function.value > UINT32_MAX || function.value >= module->function_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config function token is out of range");
  }
  *out_function = &module->functions[(uint32_t)function.value];
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_validate_function_info(
    const loomc_launch_config_function_info_t* info) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  if (info->type != LOOMC_STRUCTURE_TYPE_NONE &&
      info->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config function info has an unknown structure type");
  }
  if (info->structure_size != 0 && info->structure_size < sizeof(*info)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config function info structure_size is too small");
  }
  if (info->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config function info extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_module_function_info(
    const loomc_launch_config_module_t* module,
    loomc_launch_config_function_t function,
    loomc_launch_config_function_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_function_info(out_info));
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      module, function, &function_storage));
  *out_info = (loomc_launch_config_function_info_t){
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_FUNCTION_INFO,
      .structure_size = sizeof(*out_info),
      .name = loomc_string_view_from_iree(function_storage->name),
      .workload_argument_count = function_storage->argument_count,
      .result_count = function_storage->result_count / 3,
      .output_byte_length =
          (function_storage->result_count / 3) * sizeof(loomc_dimension3_t),
      .output_alignment = iree_alignof(loomc_dimension3_t),
  };
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_module_lookup_function_by_name(
    const loomc_launch_config_module_t* module, loomc_string_view_t name,
    loomc_launch_config_function_t* out_function) {
  if (module == NULL || out_function == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module and out_function must not be NULL");
  }
  *out_function = loomc_launch_config_function_invalid();
  if (!loomc_launch_config_string_view_is_well_formed(name) ||
      loomc_string_view_is_empty(name)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "function name must be a non-empty string view");
  }

  const loom_string_id_t name_id = loom_module_lookup_string(
      module->module, iree_string_view_from_loomc(name));
  if (name_id == LOOM_STRING_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  const loom_symbol_id_t symbol_id =
      loom_module_find_symbol(module->module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  const uint32_t function_ordinal = module->symbol_function_ordinals[symbol_id];
  if (function_ordinal == UINT32_MAX) {
    return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                             "launch config function was not found");
  }
  *out_function = loomc_launch_config_function_from_index(function_ordinal);
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_context_block_size(
    const loomc_launch_config_module_t* module,
    iree_host_size_t* out_block_size) {
  *out_block_size = 0;
  const iree_host_size_t value_capacity =
      loom_value_table_capacity(&module->module->values);

  iree_host_size_t fact_storage_size = 0;
  iree_host_size_t touched_storage_size = 0;
  if (!iree_host_size_checked_mul(value_capacity, sizeof(loom_value_facts_t),
                                  &fact_storage_size) ||
      !iree_host_size_checked_align(fact_storage_size, iree_max_align_t,
                                    &fact_storage_size) ||
      !iree_host_size_checked_mul(value_capacity, sizeof(loom_value_id_t),
                                  &touched_storage_size) ||
      !iree_host_size_checked_align(touched_storage_size, iree_max_align_t,
                                    &touched_storage_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config module requires too much evaluation storage");
  }

  iree_host_size_t persistent_storage_size = 0;
  iree_host_size_t required_total_size = 0;
  if (!iree_host_size_checked_add(fact_storage_size, touched_storage_size,
                                  &persistent_storage_size) ||
      !iree_host_size_checked_add(persistent_storage_size,
                                  sizeof(iree_arena_block_t),
                                  &required_total_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config module requires too much evaluation storage");
  }
  const iree_host_size_t requested_size =
      iree_max(required_total_size,
               (iree_host_size_t)LOOMC_LAUNCH_CONFIG_MODULE_MINIMUM_BLOCK_SIZE);
  const iree_host_size_t block_size =
      iree_host_size_next_power_of_two(requested_size);
  if (block_size < requested_size ||
      !iree_arena_block_pool_is_valid_total_size(block_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config module requires an unsupported arena block size");
  }
  *out_block_size = block_size;
  return loomc_ok_status();
}

static void loomc_launch_config_context_destroy(
    loomc_launch_config_context_t* context) {
  loomc_allocator_t allocator = context->allocator;
  loom_pass_value_fact_owner_deinitialize(&context->fact_owner);
  iree_arena_block_pool_deinitialize(&context->block_pool);
  loomc_launch_config_module_release(context->module);
  loomc_allocator_free(allocator, context);
}

loomc_status_t loomc_launch_config_context_create(
    loomc_launch_config_module_t* module, loomc_allocator_t allocator,
    loomc_launch_config_context_t** out_context) {
  if (module == NULL || out_context == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config module and out_context must not be NULL");
  }
  *out_context = NULL;
  iree_host_size_t context_block_size = 0;
  LOOMC_RETURN_IF_ERROR(
      loomc_launch_config_context_block_size(module, &context_block_size));

  loomc_launch_config_context_t* context = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*context), (void**)&context));
  memset(context, 0, sizeof(*context));
  iree_atomic_ref_count_init(&context->ref_count);
  context->allocator = allocator;
  context->module = module;
  loomc_launch_config_module_retain(module);
  iree_arena_block_pool_initialize(context_block_size,
                                   iree_allocator_from_loomc(allocator),
                                   &context->block_pool);
  loomc_status_t status =
      loomc_status_from_iree(iree_arena_block_pool_preallocate(
          &context->block_pool,
          LOOMC_LAUNCH_CONFIG_CONTEXT_PREALLOCATED_BLOCK_COUNT));
  if (loomc_status_is_ok(status)) {
    loom_pass_value_fact_owner_initialize(&context->block_pool,
                                          &context->fact_owner);
  }
  if (loomc_status_is_ok(status)) {
    *out_context = context;
  } else {
    iree_arena_block_pool_deinitialize(&context->block_pool);
    loomc_launch_config_module_release(context->module);
    loomc_allocator_free(allocator, context);
  }
  return status;
}

void loomc_launch_config_context_retain(
    loomc_launch_config_context_t* context) {
  if (context == NULL) return;
  iree_atomic_ref_count_inc(&context->ref_count);
}

void loomc_launch_config_context_release(
    loomc_launch_config_context_t* context) {
  if (context == NULL) return;
  if (iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    loomc_launch_config_context_destroy(context);
  }
}

static loomc_status_t loomc_launch_config_validate_arguments(
    const loomc_launch_config_arguments_t* arguments) {
  if (arguments == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config arguments must not be NULL");
  }
  if (arguments->type != LOOMC_STRUCTURE_TYPE_NONE &&
      arguments->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_ARGUMENTS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config arguments have an unknown structure type");
  }
  if (arguments->structure_size != 0 &&
      arguments->structure_size < sizeof(*arguments)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config arguments structure_size is too small");
  }
  if (arguments->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config argument extensions are not supported");
  }
  if (arguments->workload_argument_count != 0 &&
      arguments->workload_argument_bits == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workload_argument_count is non-zero but workload_argument_bits is "
        "NULL");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_validate_outputs(
    const loomc_launch_config_outputs_t* outputs) {
  if (outputs == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config outputs must not be NULL");
  }
  if (outputs->type != LOOMC_STRUCTURE_TYPE_NONE &&
      outputs->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config outputs have an unknown structure type");
  }
  if (outputs->structure_size != 0 &&
      outputs->structure_size < sizeof(*outputs)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config outputs structure_size is too small");
  }
  if (outputs->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config output extensions are not supported");
  }
  if (outputs->storage_length != 0 && outputs->storage == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "launch config output storage has length but no data");
  }
  return loomc_ok_status();
}

static bool loomc_launch_config_argument_facts(loom_scalar_type_t scalar_type,
                                               uint64_t bits,
                                               loom_value_facts_t* out_facts) {
  switch (scalar_type) {
    case LOOM_SCALAR_TYPE_INDEX:
    case LOOM_SCALAR_TYPE_I64:
      *out_facts = loom_value_facts_make_signed_raw_bits(bits, 64);
      return true;
    case LOOM_SCALAR_TYPE_OFFSET:
      return loom_value_facts_make_unsigned_raw_bits(bits, 64, out_facts);
    case LOOM_SCALAR_TYPE_I1:
      *out_facts = loom_value_facts_exact_i64((bits & 1) != 0 ? 1 : 0);
      return true;
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_I32:
      *out_facts = loom_value_facts_make_signed_raw_bits(
          bits, loom_scalar_type_bitwidth(scalar_type));
      return true;
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_F64:
      return loom_value_facts_from_float_bits(scalar_type, bits, out_facts);
    default:
      return false;
  }
}

static iree_status_t loomc_launch_config_check_argument(
    const loom_module_t* module,
    const loomc_launch_config_function_storage_t* function,
    uint16_t argument_ordinal, loom_value_facts_t facts) {
  const loom_value_id_t value_id = function->argument_ids[argument_ordinal];
  const loom_type_t type = loom_module_value_type(module, value_id);
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (loom_scalar_type_is_float(scalar_type)) return iree_ok_status();

  int64_t exact_value = 0;
  if (!loom_value_facts_as_exact_i64(facts, &exact_value)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "launch config function '@%.*s' argument %u is outside the exact "
        "fact domain",
        (int)function->name.size, function->name.data,
        (unsigned)argument_ordinal);
  }
  const loom_attribute_t exact_attribute =
      scalar_type == LOOM_SCALAR_TYPE_I1 ? loom_attr_bool(exact_value != 0)
                                         : loom_attr_i64(exact_value);
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function->function, &predicate_count);
  return loom_symbol_value_constraints_check_exact(
      function->name, type, value_id, exact_attribute,
      loom_attr_predicate_list((loom_predicate_t*)predicates, predicate_count));
}

static iree_status_t loomc_launch_config_evaluate_function(
    loomc_launch_config_context_t* context,
    const loomc_launch_config_function_storage_t* function,
    const uint64_t* argument_bits, iree_host_size_t argument_count,
    uint32_t* outputs, iree_host_size_t output_count) {
  if (argument_count != function->argument_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' expects %u arguments but received "
        "%" PRIhsz,
        (int)function->name.size, function->name.data,
        (unsigned)function->argument_count, argument_count);
  }
  if (output_count != function->result_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' produces %u values but received "
        "%" PRIhsz " output slots",
        (int)function->name.size, function->name.data,
        (unsigned)function->result_count, output_count);
  }

  const loom_module_t* module = context->module->module;
  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare(
      &context->fact_owner, module,
      loom_pass_value_fact_scope_function(function->function), &fact_table));
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(module, function->argument_ids[i]);
    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!loomc_launch_config_argument_facts(loom_type_element_type(type),
                                            argument_bits[i], &facts)) {
      loom_pass_value_fact_owner_invalidate(&context->fact_owner);
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "launch config function '@%.*s' argument %u bit pattern cannot be "
          "represented as exact facts",
          (int)function->name.size, function->name.data, (unsigned)i);
    }
    iree_status_t status =
        loomc_launch_config_check_argument(module, function, i, facts);
    if (iree_status_is_ok(status)) {
      status = loom_value_fact_table_define(fact_table,
                                            function->argument_ids[i], facts);
    }
    if (!iree_status_is_ok(status)) {
      loom_pass_value_fact_owner_invalidate(&context->fact_owner);
      return status;
    }
  }

  iree_status_t status =
      loom_value_fact_table_compute(fact_table, module, function->function);
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(&context->fact_owner);
    return status;
  }
  for (uint16_t i = 0; i < function->result_count; ++i) {
    int64_t value = 0;
    if (!loom_value_facts_as_exact_i64(
            loom_value_fact_table_lookup(fact_table, function->result_ids[i]),
            &value)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "launch config function '@%.*s' result %u is not exact",
          (int)function->name.size, function->name.data, (unsigned)i);
    }
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "launch config function '@%.*s' result %u value %" PRId64
          " is outside the u32 workgroup-count domain",
          (int)function->name.size, function->name.data, (unsigned)i, value);
    }
  }
  for (uint16_t i = 0; i < function->result_count; ++i) {
    int64_t value = 0;
    (void)loom_value_facts_as_exact_i64(
        loom_value_fact_table_lookup(fact_table, function->result_ids[i]),
        &value);
    outputs[i] = (uint32_t)value;
  }
  return iree_ok_status();
}

loomc_status_t loomc_launch_config_context_evaluate(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_outputs_t* outputs) {
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config context must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_arguments(arguments));
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_outputs(outputs));
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      context->module, function, &function_storage));

  const iree_host_size_t output_count = function_storage->result_count;
  const iree_host_size_t required_length = output_count * sizeof(uint32_t);
  if (outputs->storage_length < required_length) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "launch config output storage requires at least %" PRIhsz " bytes",
        required_length));
  }
  if (required_length != 0 &&
      !iree_host_ptr_has_alignment(outputs->storage,
                                   iree_alignof(loomc_dimension3_t))) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config output storage must be aligned to %" PRIhsz " bytes",
        (iree_host_size_t)iree_alignof(loomc_dimension3_t)));
  }

  return loomc_status_from_iree(loomc_launch_config_evaluate_function(
      context, function_storage, arguments->workload_argument_bits,
      arguments->workload_argument_count, (uint32_t*)outputs->storage,
      output_count));
}

static loomc_status_t loomc_launch_config_validate_result_config(
    const loomc_launch_config_t* config) {
  if (config == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_config must not be NULL");
  }
  if (config->type != LOOMC_STRUCTURE_TYPE_NONE &&
      config->type != LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config has an unknown structure type");
  }
  if (config->structure_size != 0 && config->structure_size < sizeof(*config)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config structure_size is too small");
  }
  if (config->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "launch config result extensions are not supported");
  }
  return loomc_ok_status();
}

loomc_status_t loomc_launch_config_context_evaluate_one(
    loomc_launch_config_context_t* context,
    loomc_launch_config_function_t function,
    const loomc_launch_config_arguments_t* arguments,
    loomc_launch_config_t* out_config) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_result_config(out_config));
  if (context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config context must not be NULL");
  }
  const loomc_launch_config_function_storage_t* function_storage = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_module_resolve_function(
      context->module, function, &function_storage));
  if (function_storage->result_count != 3) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "single launch evaluation requires exactly one xyz result tuple");
  }

  loomc_dimension3_t workgroup_count = {0};
  loomc_launch_config_outputs_t outputs = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG_OUTPUTS,
      .structure_size = sizeof(outputs),
      .storage = (uint8_t*)&workgroup_count,
      .storage_length = sizeof(workgroup_count),
  };
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_context_evaluate(
      context, function, arguments, &outputs));
  *out_config = (loomc_launch_config_t){
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(*out_config),
      .fields = LOOMC_LAUNCH_CONFIG_FIELD_FLAG_WORKGROUP_COUNT,
      .workgroup_count = workgroup_count,
  };
  return loomc_ok_status();
}
