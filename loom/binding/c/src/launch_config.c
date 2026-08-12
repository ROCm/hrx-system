// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/launch_config.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "context.h"
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
  // Launch programs are compact but may contain all entry points in an
  // executable. A conventional compiler workspace block avoids tiny-block
  // churn without deriving allocation policy from artifact byte length.
  LOOMC_LAUNCH_CONFIG_BLOCK_SIZE = 128 * 1024,

  // Private host-function result ABI. The compiler emits these values in this
  // fixed order and the loader verifies every public function against it.
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X = 0,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y = 1,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z = 2,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X = 3,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y = 4,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z = 5,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X = 6,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y = 7,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z = 8,
  LOOMC_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE = 9,
  LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES = 10,
  LOOMC_LAUNCH_CONFIG_RESULT_COUNT = 11,
};

typedef struct loomc_launch_config_function_storage_t {
  // Prepared public function borrowed from the immutable module.
  loom_func_like_t function;

  // Public function name borrowed from the immutable module.
  iree_string_view_t name;

  // Positional function argument value IDs.
  const loom_value_id_t* argument_ids;

  // Positional returned value IDs following the private result ABI.
  const loom_value_id_t* result_ids;

  // Number of entries in argument_ids.
  uint16_t argument_count;
} loomc_launch_config_function_storage_t;

typedef struct loomc_launch_config_evaluation_t {
  // Block pool used by reusable evaluation arenas.
  iree_arena_block_pool_t block_pool;

  // Reusable call-local value-fact state.
  loom_pass_value_fact_owner_t fact_owner;
} loomc_launch_config_evaluation_t;

struct loomc_launch_config_program_t {
  // Atomic reference count for retained handle ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for handle and metadata storage.
  loomc_allocator_t allocator;

  // Immutable language context retained by the decoded module.
  loomc_context_t* context;

  // Arena block pool owning decoded module IR.
  iree_arena_block_pool_t module_block_pool;

  // Verified immutable host module.
  loom_module_t* module;

  // Dense exported launch-function table.
  loomc_launch_config_function_storage_t* functions;

  // Number of entries in functions.
  iree_host_size_t function_count;

  // Mutable scratch reused by launch-config invocation.
  loomc_launch_config_evaluation_t evaluation;
};

static bool loomc_launch_config_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static iree_string_view_t loomc_launch_config_function_name(
    const loom_module_t* module, loom_func_like_t function) {
  const loom_symbol_ref_t symbol_ref = loom_func_like_callee(function);
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return module->strings.entries[symbol->name_id];
}

static void loomc_launch_config_evaluation_deinitialize(
    loomc_launch_config_evaluation_t* evaluation) {
  loom_pass_value_fact_owner_deinitialize(&evaluation->fact_owner);
  iree_arena_block_pool_deinitialize(&evaluation->block_pool);
  memset(evaluation, 0, sizeof(*evaluation));
}

static void loomc_launch_config_evaluation_initialize(
    loomc_launch_config_program_t* program,
    loomc_launch_config_evaluation_t* evaluation) {
  iree_arena_block_pool_initialize(
      LOOMC_LAUNCH_CONFIG_BLOCK_SIZE,
      iree_allocator_from_loomc(program->allocator), &evaluation->block_pool);
  loom_pass_value_fact_owner_initialize(&evaluation->block_pool,
                                        &evaluation->fact_owner);
}

static void loomc_launch_config_evaluation_reset(
    loomc_launch_config_evaluation_t* evaluation) {
  loom_pass_value_fact_owner_invalidate(&evaluation->fact_owner);
}

static void loomc_launch_config_program_destroy(
    loomc_launch_config_program_t* program) {
  loomc_allocator_t allocator = program->allocator;
  loomc_launch_config_evaluation_deinitialize(&program->evaluation);
  loomc_allocator_free(allocator, program->functions);
  loom_module_free(program->module);
  iree_arena_block_pool_deinitialize(&program->module_block_pool);
  loomc_context_release(program->context);
  loomc_allocator_free(allocator, program);
}

static loomc_status_t loomc_launch_config_program_bind_function(
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
  if (results.count != LOOMC_LAUNCH_CONFIG_RESULT_COUNT) {
    return loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' must return %u values", (int)name.size,
        name.data, (unsigned)LOOMC_LAUNCH_CONFIG_RESULT_COUNT));
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
  };
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_program_bind_functions(
    loomc_launch_config_program_t* program) {
  const loom_module_t* module = program->module;
  iree_host_size_t function_count = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (loom_func_like_visibility(function) == LOOM_FUNC_VISIBILITY_PUBLIC) {
      ++function_count;
    }
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

  iree_host_size_t storage_size = 0;
  if (!iree_host_size_checked_mul(function_count, sizeof(*program->functions),
                                  &storage_size)) {
    return loomc_make_status(
        LOOMC_STATUS_RESOURCE_EXHAUSTED,
        "launch config function table exceeds the host size domain");
  }
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(program->allocator, storage_size,
                                               (void**)&program->functions));
  memset(program->functions, 0, storage_size);

  iree_host_size_t function_ordinal = 0;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
      continue;
    }
    const loom_func_like_t function =
        loom_func_like_cast(module, symbol->defining_op);
    if (loom_func_like_visibility(function) != LOOM_FUNC_VISIBILITY_PUBLIC) {
      continue;
    }
    LOOMC_RETURN_IF_ERROR(loomc_launch_config_program_bind_function(
        module, function, &program->functions[function_ordinal++]));
  }
  program->function_count = function_count;
  return loomc_ok_status();
}

static loomc_status_t loomc_launch_config_program_prewarm(
    loomc_launch_config_program_t* program) {
  loom_value_fact_table_t* fact_table = NULL;
  iree_status_t status = loom_pass_value_fact_owner_prepare(
      &program->evaluation.fact_owner, program->module,
      loom_pass_value_fact_scope_function(program->functions[0].function),
      &fact_table);
  (void)fact_table;
  loomc_launch_config_evaluation_reset(&program->evaluation);
  return loomc_status_from_iree(status);
}

static loomc_status_t loomc_launch_config_program_load_impl(
    const loomc_artifact_t* artifact, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program) {
  if (out_program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_program must not be NULL");
  }
  *out_program = NULL;
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact must not be NULL");
  }
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

  loomc_launch_config_program_t* program = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_allocator_malloc(allocator, sizeof(*program), (void**)&program));
  memset(program, 0, sizeof(*program));
  iree_atomic_ref_count_init(&program->ref_count);
  program->allocator = allocator;
  iree_arena_block_pool_initialize(LOOMC_LAUNCH_CONFIG_BLOCK_SIZE,
                                   iree_allocator_from_loomc(allocator),
                                   &program->module_block_pool);
  loomc_launch_config_evaluation_initialize(program, &program->evaluation);

  loomc_status_t status = loomc_context_create(
      /*options=*/NULL, allocator, &program->context);
  if (loomc_status_is_ok(status)) {
    loom_bytecode_read_options_t read_options = {
        .verify_module = true,
        .verify_max_errors = 1,
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
        identifier, loomc_context_loom_context(program->context),
        &program->module_block_pool, &read_options, &read_result,
        &program->module, iree_allocator_from_loomc(allocator)));
    if (loomc_status_is_ok(status) &&
        (read_result.error_count != 0 || program->module == NULL)) {
      status = loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "launch config artifact is not valid verified Loom bytecode");
    }
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_program_bind_functions(program);
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_launch_config_program_prewarm(program);
  }
  if (loomc_status_is_ok(status)) {
    *out_program = program;
  } else {
    loomc_launch_config_program_destroy(program);
  }
  return status;
}

loomc_status_t loomc_launch_config_program_load(
    const loomc_artifact_t* artifact, loomc_artifact_release_fn_t release,
    void* release_user_data, loomc_allocator_t allocator,
    loomc_launch_config_program_t** out_program) {
  const loomc_byte_span_t contents =
      artifact != NULL ? artifact->contents : loomc_byte_span_empty();
  loomc_status_t status =
      loomc_launch_config_program_load_impl(artifact, allocator, out_program);
  if (release != NULL && artifact != NULL) {
    release(release_user_data, contents);
  }
  return status;
}

void loomc_launch_config_program_retain(
    loomc_launch_config_program_t* program) {
  if (program == NULL) return;
  iree_atomic_ref_count_inc(&program->ref_count);
}

void loomc_launch_config_program_release(
    loomc_launch_config_program_t* program) {
  if (program == NULL) return;
  if (iree_atomic_ref_count_dec(&program->ref_count) == 1) {
    loomc_launch_config_program_destroy(program);
  }
}

loomc_status_t loomc_launch_config_program_lookup_function(
    const loomc_launch_config_program_t* program,
    loomc_string_view_t export_name,
    loomc_launch_config_function_t* out_function) {
  if (out_function == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_function must not be NULL");
  }
  *out_function = loomc_launch_config_function_invalid();
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config program must not be NULL");
  }
  if (!loomc_launch_config_string_view_is_well_formed(export_name) ||
      loomc_string_view_is_empty(export_name) || export_name.data[0] == '@') {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "export_name must be a non-empty name without a leading '@'");
  }

  const iree_string_view_t name = iree_string_view_from_loomc(export_name);
  for (iree_host_size_t i = 0; i < program->function_count; ++i) {
    if (iree_string_view_equal(program->functions[i].name, name)) {
      *out_function = (loomc_launch_config_function_t){.value = i};
      return loomc_ok_status();
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "launch config function was not found");
}

static loomc_status_t loomc_launch_config_validate_result(
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

static iree_status_t loomc_launch_config_exact_u32(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    iree_string_view_t function_name, const char* field_name,
    bool require_nonzero, uint32_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch config function '@%.*s' %s is not exact",
                            (int)function_name.size, function_name.data,
                            field_name);
  }
  if (value < (require_nonzero ? 1 : 0) || value > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config function '@%.*s' %s value %" PRId64
                            " is outside its u32 domain",
                            (int)function_name.size, function_name.data,
                            field_name, value);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loomc_launch_config_exact_u64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    iree_string_view_t function_name, const char* field_name,
    uint64_t* out_value) {
  int64_t value = 0;
  if (!loom_value_facts_as_exact_i64(
          loom_value_fact_table_lookup(fact_table, value_id), &value)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "launch config function '@%.*s' %s is not exact",
                            (int)function_name.size, function_name.data,
                            field_name);
  }
  if (value < 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "launch config function '@%.*s' %s value %" PRId64
                            " is outside its u64 domain",
                            (int)function_name.size, function_name.data,
                            field_name, value);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

static iree_status_t loomc_launch_config_evaluate(
    loomc_launch_config_evaluation_t* evaluation, const loom_module_t* module,
    const loomc_launch_config_function_storage_t* function,
    const uint64_t* argument_bits, iree_host_size_t argument_count,
    loomc_launch_config_t* out_config) {
  if (argument_count != function->argument_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "launch config function '@%.*s' expects %u arguments but received "
        "%" PRIhsz,
        (int)function->name.size, function->name.data,
        (unsigned)function->argument_count, argument_count);
  }

  loom_value_fact_table_t* fact_table = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_value_fact_owner_prepare(
      &evaluation->fact_owner, module,
      loom_pass_value_fact_scope_function(function->function), &fact_table));
  for (uint16_t i = 0; i < function->argument_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(module, function->argument_ids[i]);
    loom_value_facts_t facts = loom_value_facts_unknown();
    if (!loomc_launch_config_argument_facts(loom_type_element_type(type),
                                            argument_bits[i], &facts)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "launch config function '@%.*s' argument %u bit pattern cannot be "
          "represented as exact facts",
          (int)function->name.size, function->name.data, (unsigned)i);
    }
    IREE_RETURN_IF_ERROR(
        loomc_launch_config_check_argument(module, function, i, facts));
    IREE_RETURN_IF_ERROR(loom_value_fact_table_define(
        fact_table, function->argument_ids[i], facts));
  }
  IREE_RETURN_IF_ERROR(
      loom_value_fact_table_compute(fact_table, module, function->function));

  loomc_launch_config_t config = {
      .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
      .structure_size = sizeof(config),
  };
#define LOOMC_EXTRACT_U32(field, result, nonzero)                       \
  IREE_RETURN_IF_ERROR(loomc_launch_config_exact_u32(                   \
      fact_table, function->result_ids[result], function->name, #field, \
      nonzero, &config.field))
  LOOMC_EXTRACT_U32(workgroup_count.x,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_X, false);
  LOOMC_EXTRACT_U32(workgroup_count.y,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Y, false);
  LOOMC_EXTRACT_U32(workgroup_count.z,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_COUNT_Z, false);
  LOOMC_EXTRACT_U32(workgroup_size.x,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_X, true);
  LOOMC_EXTRACT_U32(workgroup_size.y,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Y, true);
  LOOMC_EXTRACT_U32(workgroup_size.z,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_SIZE_Z, true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.x,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_X, true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.y,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Y, true);
  LOOMC_EXTRACT_U32(workgroup_cluster_size.z,
                    LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_CLUSTER_SIZE_Z, true);
  LOOMC_EXTRACT_U32(subgroup_size, LOOMC_LAUNCH_CONFIG_RESULT_SUBGROUP_SIZE,
                    false);
#undef LOOMC_EXTRACT_U32
  IREE_RETURN_IF_ERROR(loomc_launch_config_exact_u64(
      fact_table,
      function->result_ids[LOOMC_LAUNCH_CONFIG_RESULT_WORKGROUP_STORAGE_BYTES],
      function->name, "workgroup_storage_bytes",
      &config.workgroup_storage_bytes));
  *out_config = config;
  return iree_ok_status();
}

loomc_status_t loomc_launch_config_program_invoke(
    loomc_launch_config_program_t* program,
    loomc_launch_config_function_t function,
    const uint64_t* workload_argument_bits,
    loomc_host_size_t workload_argument_count,
    loomc_launch_config_t* out_config) {
  LOOMC_RETURN_IF_ERROR(loomc_launch_config_validate_result(out_config));
  if (program == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config program must not be NULL");
  }
  if (function.value >= program->function_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "launch config function token is out of range");
  }
  if (workload_argument_count != 0 && workload_argument_bits == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "workload_argument_count is non-zero but workload_argument_bits is "
        "NULL");
  }

  loomc_launch_config_t config = {0};
  loomc_status_t status = loomc_status_from_iree(loomc_launch_config_evaluate(
      &program->evaluation, program->module,
      &program->functions[function.value], workload_argument_bits,
      workload_argument_count, &config));
  loomc_launch_config_evaluation_reset(&program->evaluation);
  if (loomc_status_is_ok(status)) {
    *out_config = config;
  }
  return status;
}
