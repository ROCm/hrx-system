// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "program_plan.h"

#include <string.h>

#include "context.h"
#include "diagnostic.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/compile_report.h"
#include "loomc/iree.h"
#include "module.h"
#include "option_chain.h"
#include "program_provider.h"
#include "result.h"
#include "target.h"
#include "workspace.h"

typedef struct loomc_program_plan_root_storage_t {
  // Owned canonical public root name.
  loomc_string_view_t name;

  // Owned complete transitive unit closure.
  loomc_program_plan_unit_t* required_units;

  // Number of entries in |required_units|.
  loomc_host_size_t required_unit_count;
} loomc_program_plan_root_storage_t;

struct loomc_program_plan_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release this plan.
  loomc_allocator_t allocator;

  // Owned selected roots in plan-local token order.
  loomc_program_plan_root_storage_t* roots;

  // Number of entries in |roots|.
  loomc_host_size_t root_count;

  // Number of independently compilable units in token order.
  loomc_host_size_t unit_count;

  // Provider-owned immutable plan representation.
  loomc_program_plan_storage_t* storage;
};

typedef struct loomc_program_plan_resolved_options_t {
  // Compile-report descriptor found in the option chain, or NULL.
  const loomc_compile_report_options_t* compile_report;
} loomc_program_plan_resolved_options_t;

static bool loomc_program_plan_string_view_is_valid(loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_string_view_t loomc_program_plan_normalize_root_name(
    loomc_string_view_t name) {
  if (name.size != 0 && name.data[0] == '@') {
    return loomc_make_string_view(name.data + 1, name.size - 1);
  }
  return name;
}

static loomc_status_t loomc_program_plan_validate_prepare_options(
    const loomc_program_plan_options_t* options,
    loomc_program_plan_resolved_options_t* out_options) {
  *out_options = (loomc_program_plan_resolved_options_t){0};
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan options structure_size is too small");
  }
  loomc_option_chain_t option_chain = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      options->next, LOOMC_OPTION_CHAIN_ALLOW_COMPILE_REPORT, &option_chain));
  out_options->compile_report = option_chain.compile_report;
  return loomc_ok_status();
}

static loomc_status_t loomc_program_plan_validate_root_names(
    const loomc_string_view_t* root_names, loomc_host_size_t root_count) {
  if (root_count == 0 || root_names == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "at least one program root name is required");
  }
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    if (!loomc_program_plan_string_view_is_valid(root_names[i]) ||
        loomc_string_view_is_empty(
            loomc_program_plan_normalize_root_name(root_names[i]))) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "program root names must not be empty");
    }
  }
  return loomc_ok_status();
}

static iree_status_t loomc_program_plan_write_report_root(
    const loomc_program_plan_root_storage_t* root,
    loomc_compile_report_mode_t mode, loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("name"), iree_string_view_from_loomc(root->name)));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("required_unit_count"), root->required_unit_count));
  if (mode == LOOMC_COMPILE_REPORT_MODE_DETAILS) {
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("required_units")));
    loom_json_array_writer_t units;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(object.stream, &units));
    for (loomc_host_size_t i = 0; i < root->required_unit_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_json_array_write_uint64_element(
          &units, root->required_units[i].value));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&units));
  }
  return loom_json_object_end(&object);
}

static iree_status_t loomc_program_plan_write_report(
    const loomc_program_plan_t* program_plan,
    const loomc_compile_report_options_t* options,
    loom_output_stream_t* stream) {
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("kind"), IREE_SV("loom.program_plan")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
      &object, IREE_SV("schema_version"), 0));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("mode"),
      iree_string_view_from_loomc(
          loomc_compile_report_mode_name(options->mode))));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("root_count"), program_plan->root_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("unit_count"), program_plan->unit_count));
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(&object, IREE_SV("roots")));
  loom_json_array_writer_t roots;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(object.stream, &roots));
  for (loomc_host_size_t i = 0; i < program_plan->root_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&roots));
    IREE_RETURN_IF_ERROR(loomc_program_plan_write_report_root(
        &program_plan->roots[i], options->mode, roots.stream));
  }
  IREE_RETURN_IF_ERROR(loom_json_array_end(&roots));
  return loom_json_object_end(&object);
}

static loomc_status_t loomc_program_plan_add_report_artifact(
    loomc_result_t* result, const loomc_program_plan_t* program_plan,
    const loomc_compile_report_options_t* options) {
  if (options == NULL || options->mode == LOOMC_COMPILE_REPORT_MODE_NONE) {
    return loomc_ok_status();
  }
  const loomc_allocator_t allocator = loomc_result_allocator(result);
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_from_loomc(allocator),
                                 &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loomc_status_t status = loomc_status_from_iree(
      loomc_program_plan_write_report(program_plan, options, &stream));

  char* report_storage = NULL;
  if (loomc_status_is_ok(status)) {
    const iree_host_size_t report_length = iree_string_builder_size(&builder);
    report_storage = iree_string_builder_take_storage(&builder);
    const loomc_string_view_t identifier =
        loomc_string_view_is_empty(options->identifier)
            ? loomc_make_cstring_view("program-plan.compile-report.json")
            : options->identifier;
    status = loomc_result_add_artifact_take_contents(
        result, LOOMC_ARTIFACT_KIND_REPORT,
        loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
        identifier, loomc_make_byte_span(report_storage, report_length));
  }
  if (loomc_status_is_ok(status)) report_storage = NULL;
  loomc_allocator_free(allocator, report_storage);
  iree_string_builder_deinitialize(&builder);
  return status;
}

static void loomc_program_plan_destroy(loomc_program_plan_t* program_plan) {
  const loomc_allocator_t allocator = program_plan->allocator;
  program_plan->storage->operations->destroy(program_plan->storage, allocator);
  loomc_allocator_free(allocator, program_plan);
}

loomc_status_t loomc_program_plan_create(
    const loomc_program_plan_create_params_t* params,
    loomc_allocator_t allocator, loomc_program_plan_t** out_program_plan) {
  *out_program_plan = NULL;

  iree_host_size_t required_unit_storage_count = 0;
  iree_host_size_t name_storage_size = 0;
  for (loomc_host_size_t i = 0; i < params->root_count; ++i) {
    if (!iree_host_size_checked_add(required_unit_storage_count,
                                    params->roots[i].required_unit_count,
                                    &required_unit_storage_count) ||
        !iree_host_size_checked_add(name_storage_size,
                                    params->roots[i].name.size,
                                    &name_storage_size)) {
      return loomc_make_status(LOOMC_STATUS_OUT_OF_RANGE,
                               "program plan metadata storage overflow");
    }
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t roots_offset = 0;
  iree_host_size_t required_units_offset = 0;
  iree_host_size_t name_storage_offset = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(loomc_program_plan_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          params->root_count, loomc_program_plan_root_storage_t,
          iree_alignof(loomc_program_plan_root_storage_t), &roots_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          required_unit_storage_count, loomc_program_plan_unit_t,
          iree_alignof(loomc_program_plan_unit_t), &required_units_offset),
      IREE_STRUCT_FIELD(name_storage_size, char, &name_storage_offset))));

  loomc_program_plan_t* program_plan = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      iree_allocator_malloc_array(iree_allocator_from_loomc(allocator),
                                  total_size, 1, (void**)&program_plan)));
  iree_atomic_ref_count_init(&program_plan->ref_count);
  program_plan->allocator = allocator;
  program_plan->roots =
      (loomc_program_plan_root_storage_t*)((uint8_t*)program_plan +
                                           roots_offset);
  program_plan->root_count = params->root_count;
  program_plan->unit_count = params->unit_count;
  program_plan->storage = params->storage;

  loomc_program_plan_unit_t* required_unit_storage =
      (loomc_program_plan_unit_t*)((uint8_t*)program_plan +
                                   required_units_offset);
  char* name_storage = (char*)program_plan + name_storage_offset;
  for (loomc_host_size_t i = 0; i < params->root_count; ++i) {
    const loomc_program_plan_root_create_params_t* source = &params->roots[i];
    loomc_program_plan_root_storage_t* target = &program_plan->roots[i];
    target->name = loomc_make_string_view(name_storage, source->name.size);
    memcpy(name_storage, source->name.data, source->name.size);
    name_storage += source->name.size;
    target->required_units = required_unit_storage;
    target->required_unit_count = source->required_unit_count;
    for (loomc_host_size_t j = 0; j < source->required_unit_count; ++j) {
      required_unit_storage[j] = source->required_units[j];
    }
    required_unit_storage += source->required_unit_count;
  }

  *out_program_plan = program_plan;
  return loomc_ok_status();
}

const loomc_program_plan_storage_t* loomc_program_plan_storage(
    const loomc_program_plan_t* program_plan) {
  return program_plan ? program_plan->storage : NULL;
}

loomc_status_t loomc_program_plan_prepare(
    loomc_workspace_t* workspace, const loomc_module_t* module,
    const loomc_string_view_t* root_names, loomc_host_size_t root_count,
    const loomc_program_plan_options_t* options, loomc_allocator_t allocator,
    loomc_program_plan_t** out_program_plan, loomc_result_t** out_result) {
  if (out_program_plan == NULL || out_result == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "out_program_plan and out_result must not be NULL");
  }
  *out_program_plan = NULL;
  *out_result = NULL;
  if (workspace == NULL || module == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "workspace and module must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(
      loomc_program_plan_validate_root_names(root_names, root_count));
  loomc_program_plan_resolved_options_t resolved_options = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_program_plan_validate_prepare_options(options, &resolved_options));

  loomc_context_t* context = loomc_module_context(module);
  loomc_target_environment_t* target_environment =
      loomc_context_target_environment(context);
  const loomc_program_provider_set_t* provider_set =
      loomc_target_environment_program_provider_set(target_environment);
  if (provider_set == NULL || provider_set->count == 0) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "target environment does not provide program planning");
  }

  loomc_result_t* result = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_result_create(LOOMC_RESULT_STATE_SUCCEEDED, allocator, &result));
  loomc_module_t* sealed_module = NULL;
  loomc_status_t status =
      loomc_module_clone(module, workspace, allocator, &sealed_module);

  iree_arena_allocator_t selection_arena;
  bool selection_arena_initialized = false;
  loomc_program_provider_selection_t selection = {0};
  if (loomc_status_is_ok(status)) {
    iree_arena_initialize(loomc_workspace_block_pool(workspace),
                          &selection_arena);
    selection_arena_initialized = true;
    status = loomc_program_provider_select_roots(
        provider_set, loomc_module_const_loom_module(sealed_module), root_names,
        root_count, &selection_arena, &selection);
  }

  loomc_program_plan_t* program_plan = NULL;
  if (loomc_status_is_ok(status)) {
    status = selection.provider->prepare(
        workspace, sealed_module, selection.root_ops, selection.root_count,
        options, result, allocator, &program_plan);
  }
  if (loomc_status_is_ok(status) && program_plan != NULL &&
      loomc_result_succeeded(result)) {
    status = loomc_program_plan_add_report_artifact(
        result, program_plan, resolved_options.compile_report);
  }

  if (selection_arena_initialized) {
    iree_arena_deinitialize(&selection_arena);
  }
  loomc_module_release(sealed_module);
  if (loomc_status_is_ok(status)) {
    *out_program_plan = program_plan;
    *out_result = result;
    program_plan = NULL;
    result = NULL;
  }
  loomc_program_plan_release(program_plan);
  loomc_result_release(result);
  return status;
}

void loomc_program_plan_retain(loomc_program_plan_t* program_plan) {
  if (program_plan == NULL) return;
  iree_atomic_ref_count_inc(&program_plan->ref_count);
}

void loomc_program_plan_release(loomc_program_plan_t* program_plan) {
  if (program_plan == NULL) return;
  if (iree_atomic_ref_count_dec(&program_plan->ref_count) == 1) {
    loomc_program_plan_destroy(program_plan);
  }
}

loomc_host_size_t loomc_program_plan_root_count(
    const loomc_program_plan_t* program_plan) {
  return program_plan ? program_plan->root_count : 0;
}

loomc_program_plan_root_t loomc_program_plan_root_at(
    const loomc_program_plan_t* program_plan, loomc_host_size_t index) {
  if (program_plan == NULL || index >= program_plan->root_count) {
    return loomc_program_plan_root_invalid();
  }
  const loomc_program_plan_root_t root = {index};
  return root;
}

loomc_status_t loomc_program_plan_lookup_root(
    const loomc_program_plan_t* program_plan, loomc_string_view_t name,
    loomc_program_plan_root_t* out_root) {
  if (program_plan == NULL || out_root == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program_plan and out_root must not be NULL");
  }
  *out_root = loomc_program_plan_root_invalid();
  if (!loomc_program_plan_string_view_is_valid(name) ||
      loomc_string_view_is_empty(
          loomc_program_plan_normalize_root_name(name))) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "root name must not be empty");
  }
  name = loomc_program_plan_normalize_root_name(name);
  for (loomc_host_size_t i = 0; i < program_plan->root_count; ++i) {
    if (loomc_string_view_equal(program_plan->roots[i].name, name)) {
      *out_root = loomc_program_plan_root_at(program_plan, i);
      return loomc_ok_status();
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "program plan root was not found");
}

loomc_status_t loomc_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_program_plan_root_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  *out_info = (loomc_program_plan_root_info_t){0};
  if (program_plan == NULL || root.value >= program_plan->root_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan root token is out of range");
  }
  const loomc_program_plan_root_storage_t* storage =
      &program_plan->roots[root.value];
  *out_info = (loomc_program_plan_root_info_t){
      .name = storage->name,
      .required_units = storage->required_units,
      .required_unit_count = storage->required_unit_count,
  };
  return loomc_ok_status();
}

loomc_host_size_t loomc_program_plan_unit_count(
    const loomc_program_plan_t* program_plan) {
  return program_plan ? program_plan->unit_count : 0;
}

loomc_program_plan_unit_t loomc_program_plan_unit_at(
    const loomc_program_plan_t* program_plan, loomc_host_size_t index) {
  if (program_plan == NULL || index >= program_plan->unit_count) {
    return loomc_program_plan_unit_invalid();
  }
  const loomc_program_plan_unit_t unit = {index};
  return unit;
}

loomc_status_t loomc_program_plan_unit_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    loomc_program_plan_unit_info_t* out_info) {
  if (out_info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  *out_info = (loomc_program_plan_unit_info_t){0};
  if (program_plan == NULL || unit.value >= program_plan->unit_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "program plan unit token is out of range");
  }
  out_info->module = program_plan->storage->operations->unit_module(
      program_plan->storage, unit.value);
  return loomc_ok_status();
}

static loomc_status_t loomc_program_plan_validate_compile_options(
    const loomc_program_plan_unit_compile_options_t* options) {
  if (options == NULL) return loomc_ok_status();
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit compile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program plan unit compile options structure_size is too small");
  }
  loomc_option_chain_t option_chain = {0};
  return loomc_option_chain_resolve(
      options->next, LOOMC_OPTION_CHAIN_ALLOW_COMPILE_REPORT, &option_chain);
}

loomc_status_t loomc_program_plan_compile_unit(
    const loomc_program_plan_t* program_plan, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_program_plan_unit_t unit,
    const loomc_pass_program_t* pass_program,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_result_t** out_result) {
  if (out_result == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_result must not be NULL");
  }
  *out_result = NULL;
  if (program_plan == NULL || compiler == NULL || workspace == NULL ||
      unit.value >= program_plan->unit_count || pass_program == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "program_plan, compiler, workspace, pass_program, and a valid unit "
        "token are required");
  }
  LOOMC_RETURN_IF_ERROR(loomc_program_plan_validate_compile_options(options));
  return program_plan->storage->operations->compile_unit(
      program_plan->storage, compiler, workspace, unit.value, pass_program,
      options, allocator, out_result);
}
