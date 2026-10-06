// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/scenario_profile.h"

#include <string.h>

#include "loom/error/diagnostic.h"
#include "loom/error/error_catalog.h"
#include "loom/error/source.h"
#include "loom/ir/value_replacement.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/tooling/execution/hal/testbench_staging.h"

typedef struct loom_run_hal_testbench_scenario_product_t {
  // Host allocator owning this product.
  iree_allocator_t host_allocator;
  // Authored scenario subject accepted by this product.
  const loom_testbench_invocation_plan_t* subject;
  // Target-only module clone containing a generated function adapter.
  loom_run_module_t adapted_run_module;
  // Synthetic kernel invocation owned with |adapted_run_module|.
  loom_testbench_invocation_plan_t adapted_invocation;
  // Eagerly compiled ordinary HAL kernel product.
  loom_run_hal_testbench_actual_provider_t provider;
} loom_run_hal_testbench_scenario_product_t;

typedef struct loom_run_hal_testbench_scenario_batch_t {
  // Host allocator owning all batch arrays.
  iree_allocator_t host_allocator;
  // Per-call retained binding lists.
  loom_run_hal_binding_list_t* binding_lists;
  // Number of initialized entries in |binding_lists|.
  iree_host_size_t initialized_binding_list_count;
  // Per-call dispatch recording descriptors.
  loom_run_hal_dispatch_sequence_step_t* steps;
  // Flat binding byte lengths in call and ABI order.
  iree_device_size_t* binding_lengths;
  // Flat staged binding table in call and ABI order.
  iree_hal_buffer_binding_t* binding_table;
  // Maximum number of entries in the flat binding arrays.
  iree_host_size_t binding_capacity;
  // Number of populated entries in |binding_table|.
  iree_host_size_t binding_count;
  // Prepared command sequence containing every call.
  loom_run_hal_dispatch_sequence_t sequence;
  // Alias-preserving host-to-device staging for the flat binding table.
  loom_run_hal_testbench_staging_t staging;
} loom_run_hal_testbench_scenario_batch_t;

static iree_status_t loom_run_hal_testbench_scenario_emit_unsupported_results(
    const loom_run_hal_testbench_scenario_profile_t* profile,
    const loom_testbench_invocation_plan_t* invocation) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(profile->name),
      loom_param_u32((uint32_t)invocation->result_count),
  };
  loom_source_range_t source_location = {
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
  };
  const loom_run_module_t* run_module = profile->provider_options.run_module;
  if (run_module != NULL && invocation->op != NULL) {
    loom_source_resolve(loom_run_module_source_resolver(run_module),
                        invocation->module, invocation->op->location,
                        &source_location);
  }
  const loom_diagnostic_t diagnostic = {
      .severity = LOOM_DIAGNOSTIC_ERROR,
      .error = LOOM_ERR_TARGET_093,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .emitter = LOOM_EMITTER_BUILDER,
      .origin = source_location,
      .source_location = source_location,
  };
  IREE_RETURN_IF_ERROR(loom_diagnostic_emit(
      &profile->provider_options.diagnostic_sink, &diagnostic));
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "HAL scenario profile cannot transport invocation results");
}

static void loom_run_hal_testbench_scenario_batch_deinitialize(
    loom_run_hal_testbench_scenario_batch_t* batch) {
  loom_run_hal_testbench_staging_deinitialize(&batch->staging);
  loom_run_hal_dispatch_sequence_deinitialize(&batch->sequence);
  for (iree_host_size_t i = batch->initialized_binding_list_count; i > 0; --i) {
    loom_run_hal_binding_list_deinitialize(&batch->binding_lists[i - 1]);
  }
  iree_allocator_free(batch->host_allocator, batch->binding_table);
  iree_allocator_free(batch->host_allocator, batch->binding_lengths);
  iree_allocator_free(batch->host_allocator, batch->steps);
  iree_allocator_free(batch->host_allocator, batch->binding_lists);
  *batch = (loom_run_hal_testbench_scenario_batch_t){0};
}

static iree_status_t loom_run_hal_testbench_scenario_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(allocator, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_batch_initialize(
    iree_host_size_t call_count, iree_host_size_t input_count,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_scenario_batch_t* out_batch) {
  *out_batch = (loom_run_hal_testbench_scenario_batch_t){
      .host_allocator = host_allocator,
  };
  if (!iree_host_size_checked_mul(call_count, input_count,
                                  &out_batch->binding_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL scenario batch binding capacity overflowed");
  }

  iree_status_t status = loom_run_hal_testbench_scenario_allocate_array(
      host_allocator, call_count, sizeof(*out_batch->binding_lists),
      (void**)&out_batch->binding_lists);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_scenario_allocate_array(
        host_allocator, call_count, sizeof(*out_batch->steps),
        (void**)&out_batch->steps);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_scenario_allocate_array(
        host_allocator, out_batch->binding_capacity,
        sizeof(*out_batch->binding_lengths),
        (void**)&out_batch->binding_lengths);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_scenario_allocate_array(
        host_allocator, out_batch->binding_capacity,
        sizeof(*out_batch->binding_table), (void**)&out_batch->binding_table);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_scenario_batch_deinitialize(out_batch);
  }
  return status;
}

static void loom_run_hal_testbench_scenario_product_destroy(void* user_data) {
  loom_run_hal_testbench_scenario_product_t* product =
      (loom_run_hal_testbench_scenario_product_t*)user_data;
  const iree_allocator_t host_allocator = product->host_allocator;
  loom_run_hal_testbench_actual_provider_deinitialize(&product->provider);
  loom_run_module_deinitialize(&product->adapted_run_module);
  iree_allocator_free(host_allocator, product);
}

static iree_status_t loom_run_hal_testbench_scenario_product_validate(
    const loom_run_hal_testbench_scenario_product_t* product,
    const loom_testbench_invocation_plan_t* invocation) {
  const loom_run_hal_testbench_actual_provider_t* provider = &product->provider;
  if (invocation != product->subject) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL scenario product received an unexpected invocation");
  }
  IREE_ASSERT(provider->kernel_launch != NULL);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_batch_prepare(
    loom_run_hal_testbench_scenario_product_t* product,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    loom_run_hal_testbench_scenario_batch_t* out_batch) {
  loom_run_hal_testbench_actual_provider_t* provider = &product->provider;
  const loom_testbench_invocation_plan_t* kernel_launch =
      provider->kernel_launch;
  iree_status_t status = loom_run_hal_testbench_scenario_batch_initialize(
      call_count, kernel_launch->input_count, product->host_allocator,
      out_batch);
  iree_host_size_t binding_count = 0;
  for (iree_host_size_t call_index = 0;
       iree_status_is_ok(status) && call_index < call_count; ++call_index) {
    loom_testbench_product_call_t* call = &calls[call_index];
    loom_run_hal_binding_list_t* bindings =
        &out_batch->binding_lists[call_index];
    out_batch->initialized_binding_list_count = call_index + 1;
    loom_run_hal_invocation_options_t invocation_options = {0};
    status = loom_run_hal_testbench_actual_provider_materialize_invocation(
        provider, kernel_launch->workload_count, call->call_parameters,
        kernel_launch->input_count, call->arguments, &invocation_options,
        bindings);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status,
          "materializing HAL scenario trial configuration %zu domain %zu "
          "ordinal %zu",
          call->identity->configuration_ordinal, call->identity->trial_index,
          call->identity->trial_ordinal);
      break;
    }
    IREE_ASSERT(binding_count + bindings->count <= out_batch->binding_capacity);
    const iree_host_size_t first_binding = binding_count;
    for (iree_host_size_t binding_index = 0; binding_index < bindings->count;
         ++binding_index) {
      const iree_tooling_buffer_binding_t* binding =
          &bindings->values[binding_index];
      out_batch->binding_lengths[binding_count] = binding->byte_length;
      out_batch->binding_table[binding_count] = (iree_hal_buffer_binding_t){
          .buffer = binding->buffer,
          .offset = binding->byte_offset,
          .length = binding->byte_length,
      };
      ++binding_count;
    }
    out_batch->steps[call_index] = (loom_run_hal_dispatch_sequence_step_t){
        .candidate = &provider->prepared_candidate,
        .execution_epoch = 0,
        .options = invocation_options,
        .binding_lengths = bindings->count == 0
                               ? NULL
                               : &out_batch->binding_lengths[first_binding],
        .binding_count = bindings->count,
    };
  }

  if (iree_status_is_ok(status)) {
    status = loom_run_hal_dispatch_sequence_prepare(
        &provider->context->runtime, call_count, out_batch->steps,
        &out_batch->sequence);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_initialize(
        &provider->context->runtime, binding_count, out_batch->binding_table,
        product->host_allocator, &out_batch->staging);
  }
  if (iree_status_is_ok(status)) {
    out_batch->binding_count = binding_count;
  }
  return status;
}

typedef struct loom_run_hal_testbench_scenario_batch_execution_t {
  // Prepared product owning the runtime used for execution.
  loom_run_hal_testbench_scenario_product_t* product;
  // Prepared target-local batch replayed by each benchmark iteration.
  loom_run_hal_testbench_scenario_batch_t* batch;
} loom_run_hal_testbench_scenario_batch_execution_t;

static iree_status_t loom_run_hal_testbench_scenario_batch_execute(
    void* user_data) {
  loom_run_hal_testbench_scenario_batch_execution_t* context =
      (loom_run_hal_testbench_scenario_batch_execution_t*)user_data;
  return loom_run_hal_dispatch_sequence_execute(
      &context->product->provider.context->runtime, &context->batch->sequence,
      (iree_hal_buffer_binding_table_t){
          .count = context->batch->binding_count,
          .bindings = context->batch->binding_table,
      });
}

static iree_status_t loom_run_hal_testbench_scenario_product_execute(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  loom_run_hal_testbench_scenario_product_t* product =
      (loom_run_hal_testbench_scenario_product_t*)user_data;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_scenario_product_validate(product, invocation));
  if (call_count == 0) {
    return iree_ok_status();
  }

  loom_run_hal_testbench_scenario_batch_t batch = {0};
  iree_status_t status = loom_run_hal_testbench_scenario_batch_prepare(
      product, call_count, calls, &batch);
  loom_run_hal_testbench_scenario_batch_execution_t context = {
      .product = product,
      .batch = &batch,
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_scenario_batch_execute(&context);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_staging_readback(
        &product->provider.context->runtime, &batch.staging);
  }
  loom_run_hal_testbench_scenario_batch_deinitialize(&batch);
  return status;
}

static iree_status_t loom_run_hal_testbench_scenario_product_benchmark(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_benchmark_options_t* options,
    iree_allocator_t host_allocator, loom_run_benchmark_result_t* out_result) {
  loom_run_benchmark_result_initialize(out_result);
  loom_run_hal_testbench_scenario_product_t* product =
      (loom_run_hal_testbench_scenario_product_t*)user_data;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_scenario_product_validate(product, invocation));
  if (call_count != options->batch_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL scenario benchmark call count %zu does not match batch size %zu",
        call_count, options->batch_size);
  }

  loom_run_hal_testbench_scenario_batch_t batch = {0};
  iree_status_t status = loom_run_hal_testbench_scenario_batch_prepare(
      product, call_count, calls, &batch);
  loom_run_hal_testbench_scenario_batch_execution_t context = {
      .product = product,
      .batch = &batch,
  };
  if (iree_status_is_ok(status)) {
    status = loom_run_benchmark_run_batches(
        (loom_run_benchmark_batch_callback_t){
            .fn = loom_run_hal_testbench_scenario_batch_execute,
            .user_data = &context,
        },
        options, host_allocator, out_result);
  }
  loom_run_hal_testbench_scenario_batch_deinitialize(&batch);
  return status;
}

static iree_status_t loom_run_hal_testbench_scenario_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scenario subject has an invalid symbol ref");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "scenario subject has an invalid symbol name");
  }
  *out_name = loom_string_table_get(&module->strings, symbol->name_id);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_find_function(
    loom_module_t* module, iree_string_view_t name, loom_symbol_ref_t* out_ref,
    loom_func_like_t* out_function) {
  *out_ref = loom_symbol_ref_null();
  *out_function = (loom_func_like_t){0};
  const loom_string_id_t name_id = loom_module_lookup_string(module, name);
  if (name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "scenario function '@%.*s' was not cloned",
                            (int)name.size, name.data);
  }
  const loom_symbol_id_t symbol_id = loom_module_find_symbol(module, name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "scenario function '@%.*s' was not cloned",
                            (int)name.size, name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (!loom_func_def_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL scenario function '@%.*s' must be an ordinary func.def",
        (int)name.size, name.data);
  }
  *out_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
  *out_function = loom_func_like_cast(module, symbol->defining_op);
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_add_adapter_symbol(
    loom_module_t* module, loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  for (iree_host_size_t ordinal = module->symbols.count;; ++ordinal) {
    char name_buffer[64];
    const int name_length =
        iree_snprintf(name_buffer, sizeof(name_buffer),
                      "__loom_scenario_adapter_%zu", ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "scenario adapter symbol name is too long");
    }
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_intern_string(
        module,
        iree_make_string_view(name_buffer, (iree_host_size_t)name_length),
        &name_id));
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};
    return iree_ok_status();
  }
}

static iree_status_t loom_run_hal_testbench_scenario_remap_predicates(
    loom_module_t* module, const loom_value_id_t* source_arguments,
    const loom_value_id_t* adapter_arguments, uint16_t argument_count,
    loom_func_like_t adapter) {
  if (adapter.vtable->predicates_attr_index == LOOM_ATTR_INDEX_NONE) {
    return iree_ok_status();
  }
  const uint8_t attribute_index = adapter.vtable->predicates_attr_index;
  for (uint16_t i = 0; i < argument_count; ++i) {
    loom_attribute_t remapped =
        loom_op_const_attrs(adapter.op)[attribute_index];
    bool changed = false;
    IREE_RETURN_IF_ERROR(loom_module_replace_attribute_value_references(
        module, remapped, source_arguments[i], adapter_arguments[i], &remapped,
        &changed));
    if (changed) {
      IREE_RETURN_IF_ERROR(
          loom_op_set_attr(module, adapter.op, attribute_index, remapped));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_build_function_adapter(
    loom_run_hal_testbench_scenario_product_t* product,
    loom_run_session_t* session, const loom_run_module_t* source_run_module,
    const loom_testbench_invocation_plan_t* invocation) {
  if (invocation->workload_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL scenario function subject cannot have call parameters");
  }
  if (invocation->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL scenario function result transport is not implemented");
  }

  iree_string_view_t function_name = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_scenario_symbol_name_from_ref(
      source_run_module->module, invocation->callee_ref, &function_name));
  IREE_RETURN_IF_ERROR(loom_run_module_clone(session, source_run_module,
                                             (iree_string_view_list_t){0},
                                             &product->adapted_run_module));
  loom_module_t* module = product->adapted_run_module.module;

  loom_symbol_ref_t function_ref = loom_symbol_ref_null();
  loom_func_like_t function = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_scenario_find_function(
      module, function_name, &function_ref, &function));
  uint16_t argument_count = 0;
  const loom_value_id_t* function_arguments =
      loom_func_like_arg_ids(function, &argument_count);
  if (argument_count != invocation->input_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL scenario function '@%.*s' has %u arguments for %zu inputs",
        (int)function_name.size, function_name.data, (unsigned)argument_count,
        invocation->input_count);
  }
  if (function.op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL scenario function '@%.*s' result transport is not implemented",
        (int)function_name.size, function_name.data);
  }

  loom_type_t* argument_types = NULL;
  if (argument_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        &module->arena, argument_count, sizeof(*argument_types),
        (void**)&argument_types));
  }
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(module, function_arguments[i]);
    if (!loom_type_is_scalar(type) && !loom_type_is_buffer(type)) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "HAL scenario function '@%.*s' argument %u must be scalar or buffer",
          (int)function_name.size, function_name.data, (unsigned)i);
    }
    argument_types[i] = type;
  }

  loom_symbol_ref_t adapter_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_scenario_add_adapter_symbol(module, &adapter_ref));
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(function, &predicate_count);
  const loom_symbol_ref_t target = loom_func_like_target(function);
  loom_kernel_def_build_flags_t build_flags = 0;
  if (loom_symbol_ref_is_valid(target)) {
    build_flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_TARGET;
  }
  if (predicate_count != 0) {
    build_flags |= LOOM_KERNEL_DEF_BUILD_FLAG_HAS_PREDICATES;
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);
  loom_op_t* adapter_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_def_build(
      &builder, build_flags, /*retain=*/0, target, LOOM_STRING_ID_INVALID,
      /*export_linkage=*/0, adapter_ref, /*config_arg_types=*/NULL,
      /*config_arg_types_count=*/0, argument_types, argument_count, predicates,
      predicate_count, function.op->location, &adapter_op));
  module->symbols.entries[adapter_ref.symbol_id].flags |=
      LOOM_SYMBOL_FLAG_PUBLIC;

  const loom_func_like_t adapter = loom_func_like_cast(module, adapter_op);
  uint16_t adapter_argument_count = 0;
  const loom_value_id_t* adapter_arguments =
      loom_func_like_arg_ids(adapter, &adapter_argument_count);
  if (adapter_argument_count != argument_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "generated HAL scenario adapter argument count is inconsistent");
  }
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_scenario_remap_predicates(
      module, function_arguments, adapter_arguments, argument_count, adapter));

  loom_builder_ip_t module_ip = loom_builder_enter_region(
      &builder, adapter_op, loom_kernel_def_config(adapter_op));
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  IREE_RETURN_IF_ERROR(
      loom_module_intern_type(module, index_type, &index_type));
  loom_op_t* unit_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &builder, loom_attr_i64(1), index_type, function.op->location, &unit_op));
  const loom_value_id_t unit = loom_index_constant_result(unit_op);
  loom_op_t* launch_config_op = NULL;
  IREE_RETURN_IF_ERROR(loom_kernel_launch_config_build(
      &builder, /*build_flags=*/0, unit, unit, unit, unit, unit, unit,
      LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      function.op->location, &launch_config_op));
  loom_builder_restore(&builder, module_ip);

  module_ip = loom_builder_enter_region(&builder, adapter_op,
                                        loom_kernel_def_body(adapter_op));
  loom_op_t* call_op = NULL;
  IREE_RETURN_IF_ERROR(loom_func_call_build(
      &builder, /*build_flags=*/0, /*purity=*/0, /*temperature=*/0,
      /*inline_policy=*/0, function_ref, adapter_arguments, argument_count,
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, function.op->location, &call_op));
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_kernel_return_build(&builder, function.op->location, &return_op));
  loom_builder_restore(&builder, module_ip);

  product->adapted_invocation = (loom_testbench_invocation_plan_t){
      .kind = LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH,
      .module = module,
      .op = adapter_op,
      .callee_ref = adapter_ref,
      .provider_id = LOOM_STRING_ID_INVALID,
      .provider = iree_string_view_empty(),
      .attrs = loom_named_attr_slice_empty(),
      .execution_epoch = 0,
      .launch_schedule_depth = 0,
      .workload_value_ids = NULL,
      .workload_count = 0,
      .input_value_ids = adapter_arguments,
      .input_count = adapter_argument_count,
      .result_value_ids = NULL,
      .result_count = 0,
  };
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_scenario_product_prepare(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  (void)configuration;
  *out_product = (loom_testbench_prepared_product_t){0};
  loom_run_hal_testbench_scenario_profile_t* profile =
      (loom_run_hal_testbench_scenario_profile_t*)user_data;
  if (invocation->kind != LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH &&
      invocation->kind != LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "HAL scenario profile requires a kernel or "
                            "function subject");
  }
  if (invocation->result_count != 0) {
    return loom_run_hal_testbench_scenario_emit_unsupported_results(profile,
                                                                    invocation);
  }

  if (profile->provider_options.run_module == NULL ||
      profile->provider_options.run_module->module != invocation->module) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL scenario profile is not bound to the invocation module");
  }

  loom_run_hal_testbench_scenario_product_t* product = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*product),
                                             (void**)&product));
  *product = (loom_run_hal_testbench_scenario_product_t){
      .host_allocator = host_allocator,
      .subject = invocation,
  };
  loom_run_hal_testbench_actual_provider_options_t provider_options =
      profile->provider_options;
  iree_status_t status = iree_ok_status();
  if (invocation->kind == LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL) {
    status = loom_run_hal_testbench_scenario_build_function_adapter(
        product, provider_options.session, provider_options.run_module,
        invocation);
    if (iree_status_is_ok(status)) {
      provider_options.run_module = &product->adapted_run_module;
      provider_options.kernel_launch = &product->adapted_invocation;
    }
  } else {
    provider_options.kernel_launch = invocation;
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_scenario_product_destroy(product);
    return status;
  }
  loom_run_hal_testbench_actual_provider_initialize(&provider_options,
                                                    &product->provider);
  status = loom_run_hal_testbench_actual_provider_compile(&product->provider);
  if (iree_status_is_ok(status) && product->provider.compile_rejected) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL scenario product compilation rejected at '%.*s' with '%.*s'",
        (int)product->provider.compile_failure_stage.size,
        product->provider.compile_failure_stage.data,
        (int)product->provider.compile_failure_kind.size,
        product->provider.compile_failure_kind.data);
  }
  if (iree_status_is_ok(status)) {
    *out_product = (loom_testbench_prepared_product_t){
        .execute = loom_run_hal_testbench_scenario_product_execute,
        .benchmark = loom_run_hal_testbench_scenario_product_benchmark,
        .destroy = loom_run_hal_testbench_scenario_product_destroy,
        .user_data = product,
    };
  } else {
    loom_run_hal_testbench_scenario_product_destroy(product);
  }
  return status;
}

void loom_run_hal_testbench_scenario_profile_initialize(
    iree_string_view_t name,
    const loom_run_hal_testbench_actual_provider_options_t* provider_options,
    loom_run_hal_testbench_scenario_profile_t* out_profile) {
  *out_profile = (loom_run_hal_testbench_scenario_profile_t){
      .name = name,
      .provider_options = *provider_options,
  };
  out_profile->provider_options.kernel_launch = NULL;
}

loom_testbench_execution_profile_t
loom_run_hal_testbench_scenario_execution_profile(
    loom_run_hal_testbench_scenario_profile_t* profile) {
  return (loom_testbench_execution_profile_t){
      .name = profile->name,
      .prepare = loom_run_hal_testbench_scenario_product_prepare,
      .user_data = profile,
  };
}
