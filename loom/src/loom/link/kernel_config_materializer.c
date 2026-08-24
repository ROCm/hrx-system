// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/kernel_config_materializer.h"

#include <string.h>

#include "loom/format/bytecode/function_projection_reader.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/ops.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

typedef struct loom_link_bytecode_kernel_config_source_t {
  // Indexed bytecode provider retaining the source storage.
  const loom_link_module_index_provider_t* provider;
  // Provider-local source module metadata.
  const loom_bytecode_module_metadata_t* module;
  // Module-local source kernel metadata.
  const loom_bytecode_symbol_metadata_t* symbol;
  // Module-local source kernel ordinal.
  uint32_t symbol_ordinal;
  // Symbol-local ordinal of the launch-configuration payload.
  uint8_t config_payload_ordinal;
} loom_link_bytecode_kernel_config_source_t;

typedef struct loom_link_kernel_config_symbol_resolver_t {
  // Source metadata used to classify and name reached symbols.
  const loom_bytecode_module_metadata_t* source_module;
  // Output module receiving projected declarations.
  loom_module_t* output_module;
} loom_link_kernel_config_symbol_resolver_t;

iree_status_t loom_link_plan_build_kernel_configuration(
    const loom_link_module_index_t* index,
    iree_host_size_t kernel_symbol_ordinal, iree_allocator_t allocator,
    loom_link_plan_t** out_plan) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = NULL;
  const loom_link_module_index_symbol_t* kernel =
      loom_link_module_index_symbol_at(index, kernel_symbol_ordinal);
  if (kernel == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel symbol ordinal is out of range");
  }
  if (loom_link_module_index_symbol_facet_ordinal(
          kernel, LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION) ==
      LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected symbol has no kernel configuration");
  }
  const loom_link_plan_root_facet_t root = {
      .symbol_ordinal = kernel_symbol_ordinal,
      .kind = LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
  };
  loom_link_plan_options_t options = {0};
  options.mode = LOOM_LINK_PLAN_SELECTIVE;
  options.root_facets.count = 1;
  options.root_facets.values = &root;
  return loom_link_plan_build(index, &options, allocator, out_plan);
}

static loom_diagnostic_sink_t loom_link_kernel_config_diagnostic_sink(
    const loom_link_plan_materialization_environment_t* environment,
    const loom_link_module_index_provider_t* provider) {
  if (environment->diagnostic_sink == NULL) {
    return (loom_diagnostic_sink_t){0};
  }
  return environment->diagnostic_sink(environment->user_data, provider);
}

static iree_status_t loom_link_kernel_config_resolve_source(
    const loom_link_plan_t* plan, iree_host_size_t kernel_symbol_ordinal,
    loom_link_bytecode_kernel_config_source_t* out_source) {
  *out_source = (loom_link_bytecode_kernel_config_source_t){0};
  const loom_link_module_index_t* index = loom_link_plan_index(plan);
  const loom_link_module_index_symbol_t* indexed_symbol =
      loom_link_module_index_symbol_at(index, kernel_symbol_ordinal);
  if (indexed_symbol == NULL) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel symbol ordinal is out of range");
  }
  if (!loom_link_plan_contains_symbol(plan, kernel_symbol_ordinal)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "kernel symbol is not selected by the link plan");
  }
  if (!loom_link_plan_contains_facet(
          plan, kernel_symbol_ordinal,
          LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel configuration is not selected by the link plan");
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(index, indexed_symbol);
  if (provider->kind != LOOM_LINK_PROVIDER_BYTECODE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel configuration source is not a bytecode provider");
  }
  const loom_link_module_index_module_t* indexed_module =
      loom_link_module_index_symbol_module(index, indexed_symbol);
  const loom_bytecode_module_metadata_t* module =
      &provider->bytecode.metadata
           .modules[indexed_module->provider_module_ordinal];
  const loom_bytecode_symbol_metadata_t* symbol =
      &module->symbols[indexed_symbol->module_symbol_ordinal];
  if (symbol->kind != LOOM_BYTECODE_SYMBOL_FUNC_DEF ||
      !iree_any_bit_set(symbol->interfaces, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected symbol is not a kernel definition");
  }
  if (symbol->kernel_workload_region_payload_ordinal_plus_one == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected kernel has no launch-configuration payload");
  }
  const uint8_t config_payload_ordinal =
      symbol->kernel_workload_region_payload_ordinal_plus_one - 1;
  IREE_ASSERT(config_payload_ordinal < symbol->region_payload_count);
  IREE_ASSERT(indexed_symbol->module_symbol_ordinal <= UINT32_MAX);
  *out_source = (loom_link_bytecode_kernel_config_source_t){
      .provider = provider,
      .module = module,
      .symbol = symbol,
      .symbol_ordinal = (uint32_t)indexed_symbol->module_symbol_ordinal,
      .config_payload_ordinal = config_payload_ordinal,
  };
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_allocate_module(
    const loom_link_bytecode_kernel_config_source_t* source,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_string_view_t module_name, iree_allocator_t allocator,
    loom_module_t** out_module) {
  const iree_host_size_t value_count =
      16 + source->symbol->kernel_workload_argument_count +
      source->symbol->argument_count + source->symbol->result_count + 3;
  const loom_module_size_hints_t hints = {
      .value_count = value_count,
      .string_count = 3,
      .type_count = 4,
      .symbol_count = 2,
  };
  return loom_module_allocate(context, module_name, block_pool, &hints,
                              allocator, out_module);
}

static iree_status_t loom_link_kernel_config_add_symbols(
    loom_module_t* module, iree_string_view_t source_name,
    iree_arena_allocator_t* scratch_arena, loom_symbol_ref_t* out_kernel_ref,
    loom_symbol_ref_t* out_config_ref) {
  const iree_string_view_t suffix = IREE_SV("$config");
  loom_string_id_t kernel_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, source_name, &kernel_name_id));
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, kernel_name_id,
                                              &out_kernel_ref->symbol_id));
  out_kernel_ref->module_id = 0;

  iree_host_size_t config_name_length = 0;
  if (!iree_host_size_checked_add(source_name.size, suffix.size,
                                  &config_name_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "configuration symbol name overflow");
  }
  char* config_name_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(scratch_arena, config_name_length,
                                           (void**)&config_name_storage));
  memcpy(config_name_storage, source_name.data, source_name.size);
  memcpy(config_name_storage + source_name.size, suffix.data, suffix.size);
  loom_string_id_t config_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(config_name_storage, config_name_length),
      &config_name_id));
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, config_name_id,
                                              &out_config_ref->symbol_id));
  out_config_ref->module_id = 0;
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_resolve_symbol(
    void* user_data, uint32_t source_symbol_ordinal,
    loom_symbol_ref_t* out_target_symbol_ref) {
  loom_link_kernel_config_symbol_resolver_t* resolver =
      (loom_link_kernel_config_symbol_resolver_t*)user_data;
  IREE_ASSERT(source_symbol_ordinal < resolver->source_module->symbol_count);
  const loom_bytecode_symbol_metadata_t* source_symbol =
      &resolver->source_module->symbols[source_symbol_ordinal];
  if (!iree_any_bit_set(source_symbol->interfaces,
                        LOOM_SYMBOL_INTERFACE_TARGET)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernel configuration references non-target symbol '%.*s'",
        (int)source_symbol->name.size, source_symbol->name.data);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      resolver->output_module, source_symbol->name, &name_id));
  loom_symbol_ref_t target_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(loom_module_add_symbol(resolver->output_module, name_id,
                                              &target_ref.symbol_id));
  target_ref.module_id = 0;

  loom_builder_t builder;
  loom_builder_initialize(resolver->output_module,
                          &resolver->output_module->arena,
                          loom_module_block(resolver->output_module), &builder);
  loom_op_t* target_declaration = NULL;
  IREE_RETURN_IF_ERROR(loom_target_decl_build(
      &builder, target_ref, LOOM_LOCATION_NONE, &target_declaration));
  *out_target_symbol_ref = target_ref;
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_build_declaration(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_builder_t* builder, loom_op_t** out_declaration) {
  if (header->result_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel definition has function results");
  }
  const iree_host_size_t operand_count =
      (iree_host_size_t)header->workload_argument_count +
      header->argument_count;
  if (operand_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel signature exceeds operand capacity");
  }
  const loom_op_vtable_t* vtable =
      loom_context_resolve_op(module->context, LOOM_OP_KERNEL_DECL);
  IREE_ASSERT(vtable != NULL);
  const uint16_t operand_segments[] = {
      header->workload_argument_count,
      header->argument_count,
  };
  loom_op_t* declaration = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op(
      builder, LOOM_OP_KERNEL_DECL, (uint16_t)operand_count, operand_segments,
      IREE_ARRAYSIZE(operand_segments), /*result_count=*/0, /*region_count=*/0,
      /*tied_result_count=*/0, vtable->attribute_count, LOOM_LOCATION_NONE,
      &declaration));
  if (operand_count != 0) {
    memcpy(loom_op_operands(declaration), header->signature_values,
           operand_count * sizeof(loom_value_id_t));
  }
  loom_attribute_t* attributes = loom_op_attrs(declaration);
  attributes[loom_kernel_decl_callee_ATTR_INDEX] =
      header->attributes[loom_kernel_def_callee_ATTR_INDEX];
  attributes[loom_kernel_decl_target_ATTR_INDEX] =
      header->attributes[loom_kernel_def_target_ATTR_INDEX];
  const loom_attribute_t source_predicates =
      header->attributes[loom_kernel_def_predicates_ATTR_INDEX];
  if (!loom_attr_is_absent(source_predicates)) {
    loom_predicate_t* predicates = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&module->arena, source_predicates.count,
                                  sizeof(*predicates), (void**)&predicates));
    memcpy(predicates, source_predicates.predicate_list,
           source_predicates.count * sizeof(*predicates));
    attributes[loom_kernel_decl_predicates_ATTR_INDEX] =
        loom_attr_predicate_list(predicates, source_predicates.count);
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, declaration));
  *out_declaration = declaration;
  return iree_ok_status();
}

static bool loom_link_kernel_config_predicate_uses_mapped_values(
    const loom_ir_remap_t* remap, const loom_predicate_t* predicate) {
  for (uint8_t i = 0; i < predicate->arg_count; ++i) {
    if (predicate->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    loom_value_id_t ignored = LOOM_VALUE_ID_INVALID;
    if (!loom_ir_remap_try_lookup_value(
            remap, (loom_value_id_t)predicate->args[i], &ignored)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_link_kernel_config_copy_workload_predicates(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_ir_remap_t* remap, iree_arena_allocator_t* scratch_arena,
    loom_op_t* helper_op) {
  const loom_attribute_t source_predicates =
      header->attributes[loom_kernel_def_predicates_ATTR_INDEX];
  if (loom_attr_is_absent(source_predicates)) return iree_ok_status();

  loom_predicate_t* workload_predicates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, source_predicates.count, sizeof(*workload_predicates),
      (void**)&workload_predicates));
  uint16_t workload_predicate_count = 0;
  for (uint16_t i = 0; i < source_predicates.count; ++i) {
    if (loom_link_kernel_config_predicate_uses_mapped_values(
            remap, &source_predicates.predicate_list[i])) {
      workload_predicates[workload_predicate_count++] =
          source_predicates.predicate_list[i];
    }
  }
  if (workload_predicate_count == 0) return iree_ok_status();

  loom_predicate_t* target_predicates = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_predicate_list(remap, workload_predicates,
                                                    workload_predicate_count,
                                                    &target_predicates));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, scratch_arena));
  const iree_status_t status = loom_rewriter_set_attr(
      &rewriter, helper_op, loom_func_def_predicates_ATTR_INDEX,
      loom_attr_predicate_list(target_predicates, workload_predicate_count));
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_link_kernel_config_configure_helper_arguments(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_func_like_t helper, iree_arena_allocator_t* scratch_arena) {
  uint16_t helper_argument_count = 0;
  const loom_value_id_t* helper_arguments =
      loom_func_like_arg_ids(helper, &helper_argument_count);
  IREE_ASSERT(helper_argument_count == header->workload_argument_count);

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, module, scratch_arena,
                                                /*options=*/NULL, &remap));
  IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
      &remap, header->signature_values, helper_arguments,
      header->workload_argument_count));
  for (uint16_t i = 0; i < header->workload_argument_count; ++i) {
    loom_type_t target_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap, loom_module_value_type(module, header->signature_values[i]),
        &target_type));
    IREE_RETURN_IF_ERROR(
        loom_module_set_value_type(module, helper_arguments[i], target_type));
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, header->signature_values[i], helper_arguments[i]));
  }
  return loom_link_kernel_config_copy_workload_predicates(
      module, header, &remap, scratch_arena, helper.op);
}

static iree_status_t loom_link_kernel_config_build_helper(
    loom_module_t* module, const loom_bytecode_function_header_t* header,
    loom_symbol_ref_t config_ref, iree_arena_allocator_t* scratch_arena,
    loom_builder_t* builder, loom_func_like_t* out_helper) {
  loom_type_t* argument_types = NULL;
  if (header->workload_argument_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, header->workload_argument_count, sizeof(*argument_types),
        (void**)&argument_types));
    for (uint16_t i = 0; i < header->workload_argument_count; ++i) {
      argument_types[i] = loom_type_none();
    }
  }
  const loom_type_t result_types[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
  };
  const loom_attribute_t target_attr =
      header->attributes[loom_kernel_def_target_ATTR_INDEX];
  const loom_symbol_ref_t target_ref = loom_attr_is_absent(target_attr)
                                           ? loom_symbol_ref_null()
                                           : loom_attr_as_symbol(target_attr);
  loom_func_def_build_flags_t build_flags =
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_PURITY |
      LOOM_FUNC_DEF_BUILD_FLAG_HAS_INLINE_POLICY;
  if (loom_symbol_ref_is_valid(target_ref)) {
    build_flags |= LOOM_FUNC_DEF_BUILD_FLAG_HAS_TARGET;
  }
  loom_op_t* helper_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_def_build(
      builder, build_flags,
      /*visibility=*/0, /*retain=*/0, /*cc=*/0, LOOM_FUNC_PURITY_PURE,
      /*temperature=*/0, LOOM_INLINE_POLICY_INLINE, target_ref,
      /*abi=*/0, loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), config_ref, argument_types,
      header->workload_argument_count, result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, /*predicates=*/NULL,
      /*predicates_count=*/0, LOOM_LOCATION_NONE, &helper_op));
  const loom_func_like_t helper = loom_func_like_cast(module, helper_op);
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_configure_helper_arguments(
      module, header, helper, scratch_arena));
  *out_helper = helper;
  return iree_ok_status();
}

static iree_status_t loom_link_kernel_config_materialize_body(
    const loom_link_bytecode_kernel_config_source_t* source,
    loom_bytecode_function_projection_reader_t* reader,
    const loom_bytecode_function_header_t* header, loom_module_t* module,
    loom_builder_t* builder, loom_func_like_t helper) {
  uint16_t helper_argument_count = 0;
  const loom_value_id_t* helper_arguments =
      loom_func_like_arg_ids(helper, &helper_argument_count);
  IREE_RETURN_IF_ERROR(
      loom_bytecode_function_projection_reader_materialize_region(
          reader, header, source->config_payload_ordinal, builder, helper.op,
          loom_func_like_body_region_index(helper), helper_arguments,
          helper_argument_count, /*low_descriptor_set=*/NULL));

  loom_block_t* block = loom_region_entry_block(loom_func_like_body(helper));
  loom_op_t* launch_config = block->last_op;
  if (!loom_kernel_launch_config_isa(launch_config)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel configuration does not end in kernel.launch.config");
  }
  loom_value_id_t return_values[3];
  memcpy(return_values, loom_op_const_operands(launch_config),
         sizeof(return_values));
  const loom_location_id_t location = launch_config->location;
  IREE_RETURN_IF_ERROR(loom_op_erase(module, launch_config));
  loom_builder_initialize(module, &module->arena, block, builder);
  loom_op_t* return_op = NULL;
  return loom_func_return_build(builder, return_values,
                                IREE_ARRAYSIZE(return_values), location,
                                &return_op);
}

iree_status_t loom_link_plan_materialize_bytecode_kernel_config(
    const loom_link_plan_t* plan, iree_host_size_t kernel_symbol_ordinal,
    const loom_link_plan_materialization_environment_t* environment,
    iree_string_view_t module_name,
    loom_link_kernel_config_materialization_t* out_materialization) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(environment->context);
  IREE_ASSERT_ARGUMENT(environment->block_pool);
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = (loom_link_kernel_config_materialization_t){0};

  loom_link_bytecode_kernel_config_source_t source;
  IREE_RETURN_IF_ERROR(loom_link_kernel_config_resolve_source(
      plan, kernel_symbol_ordinal, &source));

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(environment->block_pool, &scratch_arena);
  loom_module_t* module = NULL;
  iree_status_t status = loom_link_kernel_config_allocate_module(
      &source, environment->context, environment->block_pool, module_name,
      environment->allocator, &module);
  loom_symbol_ref_t kernel_ref = loom_symbol_ref_null();
  loom_symbol_ref_t config_ref = loom_symbol_ref_null();
  if (iree_status_is_ok(status)) {
    status = loom_link_kernel_config_add_symbols(
        module, source.symbol->name, &scratch_arena, &kernel_ref, &config_ref);
  }

  loom_link_kernel_config_symbol_resolver_t symbol_resolver = {
      .source_module = source.module,
      .output_module = module,
  };
  loom_bytecode_function_projection_reader_t* reader = NULL;
  if (iree_status_is_ok(status)) {
    const loom_bytecode_function_projection_reader_options_t reader_options = {
        .diagnostic_sink = loom_link_kernel_config_diagnostic_sink(
            environment, source.provider),
        .low_repr_environment = environment->low_repr_environment,
        .symbol_resolver = loom_link_kernel_config_resolve_symbol,
        .symbol_resolver_user_data = &symbol_resolver,
    };
    status = loom_bytecode_function_projection_reader_allocate(
        source.provider->bytecode.contents, source.provider->bytecode.filename,
        environment->block_pool, source.module, module, &reader_options,
        environment->allocator, &reader);
    if (iree_status_is_ok(status)) {
      status = loom_bytecode_function_projection_reader_bind_symbol(
          reader, source.symbol_ordinal, kernel_ref);
    }
  }
  loom_bytecode_function_header_t header = {0};
  if (iree_status_is_ok(status)) {
    status = loom_bytecode_function_projection_reader_read_header(
        reader, source.symbol_ordinal, &header);
  }

  loom_builder_t builder;
  loom_op_t* declaration = NULL;
  if (iree_status_is_ok(status)) {
    loom_builder_initialize(module, &module->arena, loom_module_block(module),
                            &builder);
    status = loom_link_kernel_config_build_declaration(module, &header,
                                                       &builder, &declaration);
  }
  loom_func_like_t helper = {0};
  if (iree_status_is_ok(status)) {
    status = loom_link_kernel_config_build_helper(
        module, &header, config_ref, &scratch_arena, &builder, &helper);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_kernel_config_materialize_body(&source, reader, &header,
                                                      module, &builder, helper);
  }

  loom_bytecode_function_projection_reader_free(reader);
  iree_arena_deinitialize(&scratch_arena);
  if (iree_status_is_ok(status)) {
    out_materialization->module = module;
    out_materialization->kernel_declaration = kernel_ref;
    out_materialization->configuration_function = config_ref;
    module = NULL;
  }
  loom_module_free(module);
  return status;
}
