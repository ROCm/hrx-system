// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/testbench_actual.h"

#include <string.h>

#include "iree/tooling/device_util.h"
#include "loom/error/diagnostic.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ops/kernel/launch_config.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/special_values.h"
#include "loom/tooling/compile/pipeline.h"
#include "loom/tooling/config/config.h"
#include "loom/util/fact_table.h"

void loom_run_hal_testbench_context_initialize(
    const loom_run_hal_artifact_provider_registry_t* artifact_provider_registry,
    iree_allocator_t host_allocator,
    loom_run_hal_testbench_context_t* out_context) {
  *out_context = (loom_run_hal_testbench_context_t){
      .artifact_provider_registry = artifact_provider_registry,
      .host_allocator = iree_allocator_is_null(host_allocator)
                            ? iree_allocator_system()
                            : host_allocator,
      .device_event_sink = iree_hal_device_event_sink_stderr(),
  };
}

void loom_run_hal_testbench_context_set_device_event_sink(
    loom_run_hal_testbench_context_t* context,
    iree_hal_device_event_sink_t device_event_sink) {
  IREE_ASSERT(!context->runtime_initialized);
  context->device_event_sink = device_event_sink;
}

void loom_run_hal_testbench_context_set_runtime_sanitizer_options(
    loom_run_hal_testbench_context_t* context,
    const loom_sanitizer_options_t* sanitizer_options) {
  IREE_ASSERT(!context->runtime_initialized);
  context->runtime_sanitizer_options = *sanitizer_options;
  context->has_runtime_sanitizer_options = true;
}

void loom_run_hal_testbench_context_deinitialize(
    loom_run_hal_testbench_context_t* context) {
  if (context == NULL) {
    return;
  }
  if (context->runtime_initialized) {
    loom_run_hal_runtime_deinitialize(&context->runtime);
  }
  *context = (loom_run_hal_testbench_context_t){0};
}

iree_string_view_t loom_run_hal_testbench_device_uri_driver_name(
    iree_string_view_t device_uri) {
  iree_string_view_t driver_name = iree_string_view_empty();
  iree_string_view_split(device_uri, ':', &driver_name, NULL);
  return driver_name;
}

static iree_status_t loom_run_hal_testbench_context_select_artifact_provider(
    loom_run_hal_testbench_context_t* context) {
  if (context->artifact_provider != NULL) {
    return iree_ok_status();
  }
  const loom_run_hal_artifact_provider_registry_t* registry =
      context->artifact_provider_registry;
  if (registry == NULL || registry->provider_count == 0) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "HAL actual invocations require a linked HAL artifact provider");
  }

  const iree_string_view_list_t device_uris = iree_hal_device_flag_list();
  if (device_uris.count > 1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "HAL actual invocations support exactly one "
                            "--device= URI; got %" PRIhsz,
                            device_uris.count);
  }
  if (device_uris.count == 1) {
    const iree_string_view_t driver_name =
        loom_run_hal_testbench_device_uri_driver_name(device_uris.values[0]);
    for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
      const loom_run_hal_artifact_provider_t* artifact_provider =
          registry->providers[i];
      if (iree_string_view_equal(artifact_provider->hal_driver_name,
                                 driver_name)) {
        context->artifact_provider = artifact_provider;
        return iree_ok_status();
      }
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--device=%.*s selects HAL driver `%.*s`, but no linked Loom HAL "
        "artifact provider can emit for that driver",
        (int)device_uris.values[0].size, device_uris.values[0].data,
        (int)driver_name.size, driver_name.data);
  }

  iree_string_builder_t provider_names;
  iree_string_builder_initialize(context->host_allocator, &provider_names);
  iree_status_t status = loom_run_hal_artifact_provider_registry_format_names(
      registry, &provider_names);
  if (iree_status_is_ok(status)) {
    const iree_string_view_t provider_names_view =
        iree_string_builder_view(&provider_names);
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL actual invocations require an explicit --device= URI to select a "
        "HAL driver; linked Loom HAL artifact providers: %.*s",
        (int)provider_names_view.size, provider_names_view.data);
  }
  iree_string_builder_deinitialize(&provider_names);
  return status;
}

iree_status_t loom_run_hal_testbench_context_ensure_runtime(
    loom_run_hal_testbench_context_t* context) {
  if (context->runtime_initialized) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_select_artifact_provider(context));
  loom_run_hal_runtime_options_t runtime_options;
  loom_run_hal_runtime_options_initialize(
      context->artifact_provider->hal_driver_name, &runtime_options);
  runtime_options.event_sink = context->device_event_sink;
  loom_target_pipeline_options_t runtime_target_pipeline_options =
      context->artifact_provider->default_pipeline_options;
  if (context->has_runtime_sanitizer_options) {
    runtime_target_pipeline_options.sanitizer =
        context->runtime_sanitizer_options;
  }
  runtime_options.runtime_features |=
      loom_run_hal_runtime_features_from_sanitizer_options(
          &runtime_target_pipeline_options.sanitizer);
  iree_status_t status = loom_run_hal_runtime_initialize(
      &runtime_options, context->host_allocator, &context->runtime);
  IREE_RETURN_IF_ERROR(status);
  context->runtime_initialized = true;
  return iree_ok_status();
}

iree_hal_buffer_params_t loom_run_hal_testbench_host_visible_buffer_params(
    void) {
  return (iree_hal_buffer_params_t){
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER |
               IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type =
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 0,
  };
}

static iree_status_t loom_run_hal_testbench_validate_actual_invocation(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation) {
  if (invocation->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL actual invocations support in-place HAL buffer arguments only; "
        "actual invocation in `%.*s` has %" PRIhsz " results",
        (int)case_plan->name.size, case_plan->name.data,
        invocation->result_count);
  }
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_select_actual_invocation(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t** out_invocation) {
  *out_invocation = NULL;
  iree_host_size_t actual_invocation_count = 0;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    *out_invocation = invocation;
    ++actual_invocation_count;
  }
  if (actual_invocation_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL actual invocations require exactly one actual invocation in "
        "check.case `%.*s`; found %" PRIhsz,
        (int)case_plan->name.size, case_plan->name.data,
        actual_invocation_count);
  }
  return loom_run_hal_testbench_validate_actual_invocation(case_plan,
                                                           *out_invocation);
}

iree_status_t loom_run_hal_testbench_count_actual_invocations(
    const loom_testbench_case_plan_t* case_plan,
    iree_host_size_t* out_actual_invocation_count) {
  *out_actual_invocation_count = 0;
  for (iree_host_size_t i = 0; i < case_plan->invocation_count; ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_run_hal_testbench_validate_actual_invocation(
        case_plan, invocation));
    ++*out_actual_invocation_count;
  }
  return iree_ok_status();
}

void loom_run_hal_testbench_actual_provider_initialize(
    const loom_run_hal_testbench_actual_provider_options_t* options,
    loom_run_hal_testbench_actual_provider_t* out_provider) {
  *out_provider = (loom_run_hal_testbench_actual_provider_t){
      .context = options->context,
      .session = options->session,
      .target_environment = options->target_environment,
      .filename = options->filename,
      .source = options->source,
      .pipeline = options->pipeline,
      .sanitizer = options->sanitizer,
      .config_set = options->config_set,
      .test_module = options->test_module,
      .actual_invocation = options->actual_invocation,
      .case_plan = options->case_plan,
      .sample_constant_ordinal = options->sample_constant_ordinal,
      .has_sample_constant_ordinal = options->has_sample_constant_ordinal,
      .diagnostic_sink = options->diagnostic_sink,
      .max_errors = options->max_errors,
      .report = options->report,
      .artifact_flags = options->artifact_flags,
      .artifact_manifest = options->artifact_manifest,
  };
  loom_run_hal_invocation_options_initialize(&out_provider->invocation_options);
}

void loom_run_hal_testbench_actual_provider_deinitialize(
    loom_run_hal_testbench_actual_provider_t* provider) {
  if (provider == NULL) {
    return;
  }
  if (provider->prepared_candidate_initialized) {
    loom_run_hal_prepared_candidate_deinitialize(&provider->prepared_candidate);
  }
  if (provider->candidate_initialized) {
    loom_run_hal_candidate_deinitialize(&provider->candidate);
  }
  if (provider->compile_device_target_initialized &&
      provider->context != NULL &&
      provider->context->artifact_provider != NULL &&
      provider->context->artifact_provider->deinitialize_device_target !=
          NULL) {
    provider->context->artifact_provider->deinitialize_device_target(
        provider->context->artifact_provider, &provider->compile_device_target,
        provider->context->host_allocator);
  }
  if (provider->compile_module_initialized) {
    loom_run_module_deinitialize(&provider->compile_module);
  }
  *provider = (loom_run_hal_testbench_actual_provider_t){0};
}

static iree_status_t loom_run_hal_testbench_module_symbol_name_from_ref(
    const loom_module_t* module, loom_symbol_ref_t ref,
    iree_string_view_t* out_name) {
  *out_name = iree_string_view_empty();
  if (ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u is outside the module symbol table",
                            (unsigned)ref.symbol_id);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol ref %u has an invalid name",
                            (unsigned)ref.symbol_id);
  }
  *out_name = module->strings.entries[symbol->name_id];
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_resolve_compile_func(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t entry_symbol, loom_func_like_t* out_func) {
  *out_func = (loom_func_like_t){0};
  loom_module_t* module = provider->compile_module.module;
  const loom_string_id_t entry_name_id =
      loom_module_lookup_string(module, entry_symbol);
  if (entry_name_id == LOOM_STRING_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in compile "
                            "module string table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const uint16_t symbol_id = loom_module_find_symbol(module, entry_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID ||
      symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "entry symbol '@%.*s' was not found in compile "
                            "module symbol table",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  loom_func_like_t func = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(func)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "entry symbol '@%.*s' does not define a function",
                            (int)entry_symbol.size, entry_symbol.data);
  }
  *out_func = func;
  return iree_ok_status();
}

static iree_string_view_t loom_run_hal_testbench_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = module->values.entries[value_id].name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[name_id];
}

static bool loom_run_hal_testbench_find_case_constant_value_for_name(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    iree_string_view_t name, loom_value_id_t* out_value_id) {
  if (iree_string_view_is_empty(name)) {
    *out_value_id = LOOM_VALUE_ID_INVALID;
    return false;
  }
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[i];
    if (iree_string_view_equal(parameter->name, name)) {
      *out_value_id = parameter->value_id;
      return true;
    }
    if (iree_string_view_equal(
            loom_run_hal_testbench_value_name(module, parameter->value_id),
            name)) {
      *out_value_id = parameter->value_id;
      return true;
    }
  }
  for (iree_host_size_t i = 0; i < case_plan->value_source_count; ++i) {
    const loom_testbench_value_source_plan_t* source =
        &case_plan->value_sources[i];
    if (source->kind == LOOM_TESTBENCH_VALUE_SOURCE_LITERAL &&
        iree_string_view_equal(
            loom_run_hal_testbench_value_name(module, source->value_id),
            name)) {
      *out_value_id = source->value_id;
      return true;
    }
  }
  *out_value_id = LOOM_VALUE_ID_INVALID;
  return false;
}

static bool loom_run_hal_testbench_value_facts_from_constant_attr(
    loom_attribute_t attr, loom_value_facts_t* out_facts) {
  switch (attr.kind) {
    case LOOM_ATTR_I64:
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(attr));
      return true;
    case LOOM_ATTR_F64:
      *out_facts = loom_value_facts_exact_f64(loom_attr_as_f64(attr));
      return true;
    case LOOM_ATTR_BOOL:
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_bool(attr) ? 1 : 0);
      return true;
    default:
      *out_facts = loom_value_facts_unknown();
      return false;
  }
}

typedef enum loom_run_hal_testbench_case_constant_kind_e {
  LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_NONE = 0,
  LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_LITERAL = 1,
  LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_SAMPLE_PARAMETER = 2,
} loom_run_hal_testbench_case_constant_kind_t;

static iree_status_t loom_run_hal_testbench_resolve_case_constant_facts(
    const loom_run_hal_testbench_actual_provider_t* provider,
    loom_value_id_t value_id, loom_value_facts_t* out_facts,
    loom_run_hal_testbench_case_constant_kind_t* out_kind) {
  *out_facts = loom_value_facts_unknown();
  *out_kind = LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_NONE;
  const loom_testbench_case_plan_t* case_plan = provider->case_plan;
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const loom_testbench_parameter_plan_t* parameter =
        &case_plan->parameters[i];
    if (parameter->value_id != value_id) {
      continue;
    }
    if (!provider->has_sample_constant_ordinal) {
      return iree_ok_status();
    }
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_parameter_ordinal(
            case_plan, provider->sample_constant_ordinal, i);
    loom_attribute_t sample_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        parameter, parameter_sample_ordinal, &sample_value));
    if (loom_run_hal_testbench_value_facts_from_constant_attr(sample_value,
                                                              out_facts)) {
      *out_kind = LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_SAMPLE_PARAMETER;
    }
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < case_plan->value_source_count; ++i) {
    const loom_testbench_value_source_plan_t* source =
        &case_plan->value_sources[i];
    if (source->value_id != value_id ||
        source->kind != LOOM_TESTBENCH_VALUE_SOURCE_LITERAL) {
      continue;
    }
    if (loom_run_hal_testbench_value_facts_from_constant_attr(
            source->literal.value, out_facts)) {
      *out_kind = LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_LITERAL;
    }
    return iree_ok_status();
  }
  return iree_ok_status();
}

static iree_status_t
loom_run_hal_testbench_apply_case_constant_to_func_region_argument(
    loom_run_hal_testbench_actual_provider_t* provider, loom_func_like_t func,
    uint8_t region_index, uint16_t argument_index,
    loom_value_id_t case_value_id) {
  loom_region_t* region = loom_func_like_region(func, region_index);
  if (region == NULL || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  if (argument_index >= entry_block->arg_count) {
    return iree_ok_status();
  }
  const loom_value_id_t argument_id =
      loom_block_arg_id(entry_block, argument_index);
  loom_value_facts_t facts = loom_value_facts_unknown();
  loom_run_hal_testbench_case_constant_kind_t constant_kind =
      LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_NONE;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_case_constant_facts(
      provider, case_value_id, &facts, &constant_kind));
  if (constant_kind == LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_NONE) {
    return iree_ok_status();
  }
  loom_module_t* module = provider->compile_module.module;
  const loom_type_t argument_type = loom_module_value_type(module, argument_id);
  if (!loom_value_facts_can_materialize_constant(facts, argument_type)) {
    return iree_ok_status();
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, entry_block, &builder);
  if (entry_block->first_op != NULL) {
    loom_builder_set_before(&builder, entry_block->first_op);
  }
  const loom_location_id_t location = entry_block->first_op != NULL
                                          ? entry_block->first_op->location
                                          : func.op->location;
  loom_value_id_t replacement_id = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_constant_build(&builder, facts, argument_type,
                                           location, &replacement_id));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(module, argument_id, replacement_id));
  if (constant_kind == LOOM_RUN_HAL_TESTBENCH_CASE_CONSTANT_SAMPLE_PARAMETER) {
    ++provider->sample_constant_argument_count;
  }
  return iree_ok_status();
}

static iree_status_t
loom_run_hal_testbench_apply_case_constants_to_named_region_args(
    loom_run_hal_testbench_actual_provider_t* provider, loom_func_like_t func,
    uint8_t region_index) {
  loom_region_t* region = loom_func_like_region(func, region_index);
  if (region == NULL || region->block_count == 0) {
    return iree_ok_status();
  }
  loom_block_t* entry_block = loom_region_entry_block(region);
  for (uint16_t argument_index = 0; argument_index < entry_block->arg_count;
       ++argument_index) {
    const loom_value_id_t argument_id =
        loom_block_arg_id(entry_block, argument_index);
    const iree_string_view_t argument_name = loom_run_hal_testbench_value_name(
        provider->compile_module.module, argument_id);
    loom_value_id_t case_value_id = LOOM_VALUE_ID_INVALID;
    if (!loom_run_hal_testbench_find_case_constant_value_for_name(
            provider->test_module, provider->case_plan, argument_name,
            &case_value_id)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_run_hal_testbench_apply_case_constant_to_func_region_argument(
            provider, func, region_index, argument_index, case_value_id));
  }
  return iree_ok_status();
}

static iree_status_t
loom_run_hal_testbench_apply_case_constants_to_kernel_config(
    loom_run_hal_testbench_actual_provider_t* provider, loom_func_like_t func) {
  if (!loom_kernel_def_isa(func.op)) {
    return iree_ok_status();
  }
  enum {
    LOOM_RUN_HAL_TESTBENCH_KERNEL_CONFIG_REGION_INDEX = 0,
  };
  return loom_run_hal_testbench_apply_case_constants_to_named_region_args(
      provider, func, LOOM_RUN_HAL_TESTBENCH_KERNEL_CONFIG_REGION_INDEX);
}

static iree_status_t loom_run_hal_testbench_apply_case_constants(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t entry_symbol) {
  if (provider->case_plan == NULL) {
    return iree_ok_status();
  }

  loom_func_like_t func = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_compile_func(
      provider, entry_symbol, &func));
  loom_module_t* module = provider->compile_module.module;
  for (iree_host_size_t input_index = 0;
       input_index < provider->actual_invocation->input_count; ++input_index) {
    if (input_index > UINT16_MAX) {
      continue;
    }
    const uint8_t region_count = loom_func_like_region_count(func);
    for (uint8_t region_index = 0; region_index < region_count;
         ++region_index) {
      if (!loom_func_like_region_is_body(func, region_index) &&
          !loom_func_like_region_projects_args(module, func, region_index)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_run_hal_testbench_apply_case_constant_to_func_region_argument(
              provider, func, region_index, (uint16_t)input_index,
              provider->actual_invocation->input_value_ids[input_index]));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_apply_case_constants_to_kernel_config(provider,
                                                                   func));
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_materialize_config_set(
    loom_run_hal_testbench_actual_provider_t* provider) {
  loom_tooling_config_materialize_options_t options = {0};
  loom_tooling_config_materialize_options_initialize(&options);
  options.config_set = provider->config_set;
  return loom_tooling_config_materialize_module(
      provider->compile_module.module, &options,
      loom_run_session_block_pool(provider->session), NULL);
}

static void loom_run_hal_testbench_record_compile_rejection(
    loom_run_hal_testbench_actual_provider_t* provider,
    iree_string_view_t stage, iree_string_view_t kind,
    iree_string_view_t message) {
  provider->compile_rejected = true;
  provider->compile_failure_stage = stage;
  provider->compile_failure_kind = kind;
  provider->compile_failure_message = message;
}

static iree_status_t loom_run_hal_testbench_forward_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_run_hal_testbench_actual_provider_t* provider =
      (loom_run_hal_testbench_actual_provider_t*)user_data;
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      ++provider->diagnostic_error_count;
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      ++provider->diagnostic_warning_count;
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      ++provider->diagnostic_remark_count;
      break;
    default:
      break;
  }
  const loom_diagnostic_sink_t sink =
      provider->diagnostic_sink.fn != NULL
          ? provider->diagnostic_sink
          : (loom_diagnostic_sink_t){.fn = loom_diagnostic_stderr_sink};
  return loom_diagnostic_emit(&sink, diagnostic);
}

static loom_diagnostic_sink_t loom_run_hal_testbench_counting_diagnostic_sink(
    loom_run_hal_testbench_actual_provider_t* provider) {
  return (loom_diagnostic_sink_t){
      .fn = loom_run_hal_testbench_forward_diagnostic,
      .user_data = provider,
  };
}

static uint32_t loom_run_hal_testbench_max_errors(
    const loom_run_hal_testbench_actual_provider_t* provider) {
  return provider->max_errors == 0 ? 20 : provider->max_errors;
}

static iree_status_t
loom_run_hal_testbench_resolve_static_workgroup_count_from_facts(
    loom_module_t* module, loom_func_like_t func,
    loom_target_dispatch_workgroup_count_t* out_workgroup_count,
    bool* out_resolved) {
  *out_workgroup_count = (loom_target_dispatch_workgroup_count_t){0};
  *out_resolved = false;
  if (!loom_kernel_def_isa(func.op)) {
    return iree_ok_status();
  }
  loom_value_fact_table_t facts = {0};
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(&facts, &module->arena,
                                                        module->values.count));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(&facts, module, func));
  *out_resolved = loom_kernel_def_static_workgroup_count_from_facts(
      module, func.op, &facts, out_workgroup_count);
  return iree_ok_status();
}

iree_status_t loom_run_hal_testbench_actual_provider_compile(
    loom_run_hal_testbench_actual_provider_t* provider) {
  if (provider->prepared_candidate_initialized || provider->compile_rejected) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_context_ensure_runtime(provider->context));
  if (!provider->compile_device_target_initialized) {
    const loom_run_hal_artifact_provider_t* artifact_provider =
        provider->context->artifact_provider;
    if (artifact_provider->select_device_target == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HAL artifact provider '%.*s' is missing required device target "
          "selection hook",
          (int)artifact_provider->name.size, artifact_provider->name.data);
    }
    IREE_RETURN_IF_ERROR(artifact_provider->select_device_target(
        artifact_provider, &provider->context->runtime,
        provider->context->host_allocator, &provider->compile_device_target));
    provider->compile_device_target_initialized = true;
  }

  iree_string_view_t entry_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_module_symbol_name_from_ref(
      provider->test_module, provider->actual_invocation->callee_ref,
      &entry_symbol));

  loom_run_module_parse_options_t parse_options = {0};
  loom_run_module_parse_options_initialize(&parse_options);
  parse_options.filename = provider->filename;
  parse_options.source = provider->source;
  parse_options.diagnostic_sink =
      loom_run_hal_testbench_counting_diagnostic_sink(provider);
  parse_options.max_errors = loom_run_hal_testbench_max_errors(provider);
  IREE_RETURN_IF_ERROR(loom_run_module_parse(provider->session, &parse_options,
                                             &provider->compile_module));
  provider->compile_module_initialized = true;
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_materialize_config_set(provider));
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_apply_case_constants(provider, entry_symbol));

  loom_func_like_t entry_func = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_testbench_resolve_compile_func(
      provider, entry_symbol, &entry_func));
  provider->invocation_options.function_name = entry_symbol;
  loom_target_dispatch_workgroup_count_t workgroup_count = {0};
  bool workgroup_count_resolved = false;
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_resolve_static_workgroup_count_from_facts(
          provider->compile_module.module, entry_func, &workgroup_count,
          &workgroup_count_resolved));
  if (!workgroup_count_resolved) {
    loom_run_hal_testbench_record_compile_rejection(
        provider, IREE_SV("compile"), IREE_SV("unresolved_workgroup_count"),
        IREE_SV("HAL actual invocation requires a statically resolved "
                "workgroup count after sample constants are applied; use "
                "--sample-compilation=per_sample or make launch geometry "
                "static for once-compiled candidates"));
    return iree_ok_status();
  }
  provider->invocation_options.workgroup_count[0] = workgroup_count.x;
  provider->invocation_options.workgroup_count[1] = workgroup_count.y;
  provider->invocation_options.workgroup_count[2] = workgroup_count.z;

  if (provider->target_environment == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual invocations require a target environment");
  }

  const loom_diagnostic_sink_t diagnostic_sink =
      loom_run_hal_testbench_counting_diagnostic_sink(provider);
  const uint32_t max_errors = loom_run_hal_testbench_max_errors(provider);

  loom_compile_pipeline_options_t pipeline_options = {0};
  loom_compile_pipeline_options_initialize(&pipeline_options);
  pipeline_options.pipeline = provider->pipeline;
  pipeline_options.target_pipeline_options =
      provider->context->artifact_provider->default_pipeline_options;
  pipeline_options.target_pipeline_options.sanitizer = provider->sanitizer;
  pipeline_options.target_environment = provider->target_environment;
  pipeline_options.target_selection = (loom_target_selection_t){
      .bundle = provider->compile_device_target.target_bundle,
      .data = provider->compile_device_target.data,
  };
  pipeline_options.low_descriptor_registry =
      loom_run_session_low_descriptor_registry(provider->session);
  pipeline_options.diagnostic_sink = diagnostic_sink;
  pipeline_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  pipeline_options.report = provider->report;
  const iree_host_size_t compile_error_count = provider->diagnostic_error_count;
  iree_status_t status = loom_compile_run_pipeline(
      provider->compile_module.module, &pipeline_options,
      loom_run_session_block_pool(provider->session), &provider->pass_result);
  if (!iree_status_is_ok(status)) {
    if (provider->diagnostic_error_count != compile_error_count) {
      iree_status_free(status);
      loom_run_hal_testbench_record_compile_rejection(
          provider, IREE_SV("compile"), IREE_SV("pass_diagnostics"),
          iree_string_view_empty());
      return iree_ok_status();
    }
    return status;
  }
  if (provider->pass_result.error_count != 0) {
    loom_run_hal_testbench_record_compile_rejection(
        provider, IREE_SV("compile"), IREE_SV("pass_diagnostics"),
        iree_string_view_empty());
    return iree_ok_status();
  }

  loom_run_candidate_compile_options_t compile_options = {0};
  loom_run_candidate_compile_options_initialize(&compile_options);
  compile_options.module_name = IREE_SV("loom");
  compile_options.compile_root_symbol = entry_symbol;
  compile_options.target_pipeline_options =
      pipeline_options.target_pipeline_options;
  compile_options.diagnostic_sink = diagnostic_sink;
  compile_options.source_resolver =
      loom_run_module_source_resolver(&provider->compile_module);
  compile_options.max_errors = max_errors;
  compile_options.report = provider->report;
  compile_options.artifact_flags = provider->artifact_flags;
  compile_options.artifact_manifest = provider->artifact_manifest;

  provider->candidate_initialized = true;
  const iree_host_size_t emit_error_count = provider->diagnostic_error_count;
  status = loom_run_hal_candidate_compile(
      provider->context->artifact_provider, &provider->context->runtime,
      &provider->compile_module, &compile_options,
      provider->context->host_allocator, &provider->candidate);
  provider->compile_report_available = true;
  if (!iree_status_is_ok(status)) {
    if (provider->diagnostic_error_count != emit_error_count) {
      iree_status_free(status);
      loom_run_hal_testbench_record_compile_rejection(
          provider, IREE_SV("emit"), IREE_SV("emit_diagnostics"),
          iree_string_view_empty());
      return iree_ok_status();
    }
    return status;
  }
  if (!provider->candidate.compiled) {
    if (provider->diagnostic_error_count != emit_error_count) {
      loom_run_hal_testbench_record_compile_rejection(
          provider, IREE_SV("emit"), IREE_SV("emit_diagnostics"),
          iree_string_view_empty());
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL artifact provider '%.*s' did not emit an artifact or diagnostics",
        (int)provider->context->artifact_provider->name.size,
        provider->context->artifact_provider->name.data);
  }

  IREE_RETURN_IF_ERROR(loom_run_hal_prepared_candidate_prepare(
      &provider->context->runtime, &provider->candidate.artifact,
      &provider->prepared_candidate));
  provider->prepared_candidate_initialized = true;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_invocation_options_push_constant(
    const loom_testbench_value_t* value, loom_type_t source_type,
    loom_run_hal_invocation_options_t* options) {
  if (!loom_testbench_value_is_scalar(value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch constant input must be a scalar "
                            "value");
  }
  if (!loom_type_is_scalar(source_type)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL dispatch constant input must have a scalar "
                            "source type");
  }
  const loom_scalar_type_t source_scalar_type =
      loom_type_element_type(source_type);
  if (source_scalar_type == LOOM_SCALAR_TYPE_INDEX) {
    int64_t integer_value = 0;
    IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &integer_value));
    if (integer_value < INT32_MIN || integer_value > INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "HAL dispatch %s constant value %" PRId64
                              " does not fit the 32-bit direct-constant ABI",
                              loom_scalar_type_name(source_scalar_type),
                              integer_value);
    }
    if (options->constant_count + 1 > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch constant count exceeds capacity "
          "%" PRIhsz,
          (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
    }
    options->constants[options->constant_count++] =
        (uint32_t)(int32_t)integer_value;
    return iree_ok_status();
  }
  if (source_scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    int64_t integer_value = 0;
    IREE_RETURN_IF_ERROR(loom_testbench_value_as_i64(value, &integer_value));
    if (options->constant_count + 2 > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "HAL dispatch constant count exceeds capacity "
          "%" PRIhsz,
          (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
    }
    const uint64_t raw_value = (uint64_t)integer_value;
    options->constants[options->constant_count++] = (uint32_t)raw_value;
    options->constants[options->constant_count++] = (uint32_t)(raw_value >> 32);
    return iree_ok_status();
  }

  uint32_t words[2] = {0};
  iree_host_size_t word_count = 0;
  IREE_RETURN_IF_ERROR(iree_tooling_value_write_abi_words(
      &value->scalar, IREE_ARRAYSIZE(words), words, &word_count));
  if (options->constant_count + word_count > LOOM_RUN_HAL_MAX_CONSTANT_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL dispatch constant count exceeds capacity "
                            "%" PRIhsz,
                            (iree_host_size_t)LOOM_RUN_HAL_MAX_CONSTANT_COUNT);
  }
  for (iree_host_size_t i = 0; i < word_count; ++i) {
    options->constants[options->constant_count++] = words[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_append_buffer_binding(
    const loom_testbench_value_t* input,
    loom_run_hal_binding_list_t* bindings) {
  if (!loom_testbench_value_is_buffer(input) ||
      input->buffer.kind == IREE_TOOLING_BUFFER_BINDING_KIND_NONE ||
      input->buffer.buffer == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input buffer binding is invalid");
  }
  if (bindings->count >= bindings->capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "HAL invocation binding count exceeds capacity "
                            "%" PRIhsz,
                            bindings->capacity);
  }
  iree_tooling_buffer_binding_t* binding = &bindings->values[bindings->count];
  *binding = input->buffer;
  iree_hal_buffer_retain(binding->buffer);
  iree_hal_buffer_view_retain(binding->buffer_view);
  ++bindings->count;
  return iree_ok_status();
}

static iree_status_t loom_run_hal_testbench_input_append(
    loom_run_hal_binding_list_t* bindings, const loom_testbench_value_t* input,
    loom_type_t input_type, loom_run_hal_invocation_options_t* options) {
  if (loom_testbench_value_is_buffer(input)) {
    return loom_run_hal_testbench_append_buffer_binding(input, bindings);
  }
  if (loom_testbench_value_is_scalar(input)) {
    return loom_run_hal_testbench_invocation_options_push_constant(
        input, input_type, options);
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "HAL invocation input must be a buffer binding or "
                          "a scalar value");
}

iree_status_t loom_run_hal_testbench_invocation_inputs_from_values(
    const loom_testbench_value_t* inputs, const loom_type_t* input_types,
    iree_host_size_t input_count, loom_run_hal_invocation_options_t* options,
    iree_allocator_t allocator, loom_run_hal_binding_list_t* out_bindings) {
  if (input_count != 0 && (inputs == NULL || input_types == NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL invocation input values and types are "
                            "required when input count is non-zero");
  }
  loom_run_hal_binding_list_initialize(out_bindings);
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      input_count, allocator, out_bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    status = loom_run_hal_testbench_input_append(out_bindings, &inputs[i],
                                                 input_types[i], options);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_bindings);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_actual_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  (void)out_results;
  loom_run_hal_testbench_actual_provider_t* provider =
      (loom_run_hal_testbench_actual_provider_t*)user_data;
  if (invocation != provider->actual_invocation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual provider received an unexpected invocation");
  }
  if (result_count != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "HAL actual invocations support in-place HAL buffer arguments only");
  }
  if (input_count != invocation->input_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HAL actual invocation input count mismatch");
  }
  IREE_RETURN_IF_ERROR(
      loom_run_hal_testbench_actual_provider_compile(provider));
  if (provider->compile_rejected) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "HAL actual invocation compile was rejected before dispatch");
  }

  loom_run_hal_invocation_options_t invocation_options =
      provider->invocation_options;
  loom_run_hal_binding_list_t bindings = {0};
  IREE_RETURN_IF_ERROR(loom_run_hal_binding_list_initialize_capacity(
      input_count, provider->context->host_allocator, &bindings));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < input_count;
       ++i) {
    const loom_value_id_t input_value_id =
        provider->actual_invocation->input_value_ids[i];
    const loom_type_t input_type =
        loom_module_value_type(provider->test_module, input_value_id);
    status = loom_run_hal_testbench_input_append(
        &bindings, &inputs[i], input_type, &invocation_options);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(
          status, "preparing HAL actual input %" PRIhsz " for value ID %u", i,
          (unsigned)input_value_id);
    }
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(&bindings);
    return status;
  }

  loom_run_hal_invocation_plan_t plan = {0};
  loom_run_hal_iteration_t iteration = {0};
  status = loom_run_hal_invocation_plan_prepare_from_lists(
      &invocation_options, &bindings, /*expected_bindings=*/NULL,
      /*max_output_element_count=*/0, provider->context->host_allocator, &plan);
  loom_run_hal_binding_list_deinitialize(&bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_dispatch_plan(
        &provider->context->runtime, &provider->prepared_candidate, &plan,
        provider->context->host_allocator, &iteration);
  }
  loom_run_hal_iteration_deinitialize(&iteration);
  loom_run_hal_invocation_plan_deinitialize(&plan);
  return status;
}

iree_status_t loom_run_hal_testbench_actual_sequence_initialize(
    const loom_run_hal_testbench_actual_sequence_options_t* options,
    loom_run_hal_testbench_actual_sequence_t* out_sequence) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_sequence);
  *out_sequence = (loom_run_hal_testbench_actual_sequence_t){
      .host_allocator = options->context->host_allocator,
  };

  iree_host_size_t actual_invocation_count = 0;
  iree_status_t status = loom_run_hal_testbench_count_actual_invocations(
      options->case_plan, &actual_invocation_count);
  if (iree_status_is_ok(status) && actual_invocation_count == 0) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL actual sequence requires at least one actual invocation in "
        "check.case `%.*s`",
        (int)options->case_plan->name.size, options->case_plan->name.data);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        out_sequence->host_allocator, actual_invocation_count,
        sizeof(*out_sequence->providers), (void**)&out_sequence->providers);
  }
  if (iree_status_is_ok(status)) {
    memset(out_sequence->providers, 0,
           actual_invocation_count * sizeof(*out_sequence->providers));
    out_sequence->provider_count = actual_invocation_count;
  }

  iree_host_size_t provider_index = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < options->case_plan->invocation_count;
       ++i) {
    const loom_testbench_invocation_plan_t* invocation =
        &options->case_plan->invocations[i];
    if (invocation->kind != LOOM_TESTBENCH_INVOCATION_ACTUAL) {
      continue;
    }
    const loom_run_hal_testbench_actual_provider_options_t provider_options = {
        .context = options->context,
        .session = options->session,
        .target_environment = options->target_environment,
        .filename = options->filename,
        .source = options->source,
        .pipeline = options->pipeline,
        .sanitizer = options->sanitizer,
        .config_set = options->config_set,
        .test_module = options->test_module,
        .actual_invocation = invocation,
        .case_plan = options->case_plan,
        .sample_constant_ordinal = options->sample_constant_ordinal,
        .has_sample_constant_ordinal = options->has_sample_constant_ordinal,
        .diagnostic_sink = options->diagnostic_sink,
        .max_errors = options->max_errors,
        .artifact_flags = options->artifact_flags,
        .artifact_manifest = options->artifact_manifest,
    };
    loom_run_hal_testbench_actual_provider_initialize(
        &provider_options, &out_sequence->providers[provider_index++]);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_testbench_actual_sequence_deinitialize(out_sequence);
  }
  return status;
}

void loom_run_hal_testbench_actual_sequence_deinitialize(
    loom_run_hal_testbench_actual_sequence_t* sequence) {
  if (sequence == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    loom_run_hal_testbench_actual_provider_deinitialize(
        &sequence->providers[i]);
  }
  iree_allocator_free(sequence->host_allocator, sequence->providers);
  *sequence = (loom_run_hal_testbench_actual_sequence_t){0};
}

iree_status_t loom_run_hal_testbench_actual_sequence_invoke(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t input_count, const loom_testbench_value_t* inputs,
    iree_host_size_t result_count, loom_testbench_value_t* out_results) {
  loom_run_hal_testbench_actual_sequence_t* sequence =
      (loom_run_hal_testbench_actual_sequence_t*)user_data;
  for (iree_host_size_t i = 0; i < sequence->provider_count; ++i) {
    loom_run_hal_testbench_actual_provider_t* provider =
        &sequence->providers[i];
    if (provider->actual_invocation == invocation) {
      return loom_run_hal_testbench_actual_invoke(
          provider, invocation, input_count, inputs, result_count, out_results);
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "HAL actual sequence received an unexpected invocation");
}

iree_status_t loom_run_hal_testbench_create_invocation_inputs_from_table(
    const loom_testbench_value_table_t* table,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_run_hal_invocation_options_t* base_options,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings) {
  *out_options = *base_options;
  loom_run_hal_binding_list_initialize(out_bindings);
  iree_status_t status = loom_run_hal_binding_list_initialize_capacity(
      invocation->input_count, allocator, out_bindings);
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < invocation->input_count; ++i) {
    loom_testbench_value_t value = {0};
    status = loom_testbench_value_table_lookup_retain(
        table, invocation->input_value_ids[i], &value);
    if (iree_status_is_ok(status)) {
      const loom_type_t input_type =
          loom_module_value_type(table->module, invocation->input_value_ids[i]);
      status = loom_run_hal_testbench_input_append(out_bindings, &value,
                                                   input_type, out_options);
    }
    loom_testbench_value_deinitialize(&value);
  }
  if (!iree_status_is_ok(status)) {
    loom_run_hal_binding_list_deinitialize(out_bindings);
  }
  return status;
}

iree_status_t loom_run_hal_testbench_create_invocation_inputs_for_sample(
    const loom_module_t* module,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t sample_ordinal,
    const loom_run_hal_invocation_options_t* base_options,
    iree_allocator_t allocator, loom_run_hal_invocation_options_t* out_options,
    loom_run_hal_binding_list_t* out_bindings) {
  loom_testbench_value_table_t table = {0};
  iree_status_t status = loom_testbench_value_table_initialize(
      module, case_plan, allocator, &table);
  if (iree_status_is_ok(status)) {
    status = loom_testbench_materialize_case_sample(
        materializer_options, case_plan, sample_ordinal, &table);
  }
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_testbench_create_invocation_inputs_from_table(
        &table, invocation, base_options, allocator, out_options, out_bindings);
  }
  loom_testbench_value_table_deinitialize(&table);
  return status;
}

iree_status_t loom_run_hal_testbench_prepare_invocation_plan_for_sample(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_value_materializer_options_t* materializer_options,
    const loom_testbench_invocation_plan_t* invocation,
    const loom_run_hal_invocation_options_t* base_options,
    iree_host_size_t sample_ordinal, iree_allocator_t allocator,
    loom_run_hal_invocation_plan_t* out_plan) {
  loom_run_hal_invocation_options_t invocation_options = {0};
  loom_run_hal_binding_list_t bindings = {0};
  iree_status_t status =
      loom_run_hal_testbench_create_invocation_inputs_for_sample(
          module_plan->module, materializer_options, case_plan, invocation,
          sample_ordinal, base_options, allocator, &invocation_options,
          &bindings);
  if (iree_status_is_ok(status)) {
    status = loom_run_hal_invocation_plan_prepare_from_lists(
        &invocation_options, &bindings, /*expected_bindings=*/NULL,
        /*max_output_element_count=*/0, allocator, out_plan);
  }
  loom_run_hal_binding_list_deinitialize(&bindings);
  return status;
}
