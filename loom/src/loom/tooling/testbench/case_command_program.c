// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/case_command_program.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/ir/type_refs.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/command/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scalar/ops.h"
#include "loom/rewrite/remap.h"

typedef struct loom_testbench_case_scalar_t {
  // Source value receiving the concrete scalar.
  loom_value_id_t value_id;
  // Concrete sampled or literal payload.
  loom_attribute_t value;
  // Source location used by the generated constant.
  loom_location_id_t location;
  // True when a launch operand or dependent type consumes this scalar.
  bool required;
} loom_testbench_case_scalar_t;

typedef struct loom_testbench_case_command_schedule_frame_t {
  // Insertion point to restore after completing the schedule body.
  loom_builder_ip_t parent_ip;
} loom_testbench_case_command_schedule_frame_t;

typedef struct loom_testbench_case_command_program_build_t {
  // Module receiving the generated command program.
  loom_module_t* module;
  // Testbench plan owning case storage identities.
  const loom_testbench_module_plan_t* module_plan;
  // Planned case being projected.
  const loom_testbench_case_plan_t* case_plan;
  // Concrete case sample being projected.
  loom_testbench_case_sample_t sample;
  // Transient storage returned before this build completes.
  iree_arena_allocator_t* scratch_arena;
  // Concrete host-known scalar recipes.
  loom_testbench_case_scalar_t* scalars;
  // Number of entries in |scalars|.
  iree_host_size_t scalar_count;
  // Authored storage roots in first-launch-use order.
  loom_value_id_t* binding_value_ids;
  // Number of entries in |binding_value_ids|.
  iree_host_size_t binding_count;
  // Allocated entries in |binding_value_ids|.
  iree_host_size_t binding_capacity;
  // Source-to-generated value correspondence used by dependent types.
  loom_ir_remap_t remap;
  // Builder appending to the generated command program.
  loom_builder_t builder;
  // Generated command program binding arguments.
  const loom_value_id_t* wrapper_arguments;
  // Cached byte offsets used by buffer.view operations.
  iree_device_size_t* cached_byte_offsets;
  // Generated offset constants parallel to |cached_byte_offsets|.
  loom_value_id_t* cached_offset_values;
  // Number of initialized offset cache entries.
  iree_host_size_t cached_offset_count;
  // Allocated entries in |cached_byte_offsets|.
  iree_host_size_t cached_byte_offset_capacity;
  // Allocated entries in |cached_offset_values|.
  iree_host_size_t cached_offset_value_capacity;
  // Active command schedule builder frames.
  loom_testbench_case_command_schedule_frame_t* schedule_frames;
  // Number of active entries in |schedule_frames|.
  iree_host_size_t schedule_frame_count;
  // Number of allocated entries in |schedule_frames|.
  iree_host_size_t schedule_frame_capacity;
} loom_testbench_case_command_program_build_t;

static bool loom_testbench_case_is_buffer_value(loom_type_t type) {
  return loom_type_is_buffer(type) || loom_type_is_tensor(type) ||
         loom_type_is_view(type);
}

static iree_status_t loom_testbench_case_command_program_validate_plan(
    const loom_module_t* module,
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan) {
  IREE_ASSERT(module_plan->module == module &&
              module_plan->value_storage_identity_count ==
                  module->values.count);
  IREE_ASSERT(loom_symbol_ref_is_valid(case_plan->ref) &&
              case_plan->ref.module_id == 0 &&
              case_plan->ref.symbol_id < module->symbols.count &&
              module->symbols.entries[case_plan->ref.symbol_id].defining_op ==
                  case_plan->op);
  if (case_plan->issue_count != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "case '@%.*s' has planning issues",
                            (int)case_plan->name.size, case_plan->name.data);
  }
  if (case_plan->kernel_launch_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "case '@%.*s' has no kernel launches",
                            (int)case_plan->name.size, case_plan->name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_initialize_tables(
    loom_testbench_case_command_program_build_t* build) {
  iree_host_size_t scalar_capacity = 0;
  if (!iree_host_size_checked_add(build->case_plan->parameter_count,
                                  build->case_plan->value_source_count,
                                  &scalar_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "case scalar recipe count exceeds host limits");
  }
  if (scalar_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, scalar_capacity, sizeof(*build->scalars),
        (void**)&build->scalars));
  }
  const loom_ir_remap_options_t remap_options = {0};
  return loom_ir_remap_initialize(build->module, build->module,
                                  build->scratch_arena, &remap_options,
                                  &build->remap);
}

static iree_status_t loom_testbench_case_command_program_collect_scalars(
    loom_testbench_case_command_program_build_t* build) {
  loom_testbench_case_sample_t remaining = build->sample;
  for (iree_host_size_t i = 0; i < build->case_plan->parameter_count; ++i) {
    const loom_testbench_parameter_plan_t* parameter =
        &build->case_plan->parameters[i];
    const iree_host_size_t parameter_sample_ordinal =
        loom_testbench_case_sample_next(&remaining, parameter->sample_count);
    loom_testbench_case_scalar_t* scalar =
        &build->scalars[build->scalar_count++];
    scalar->value_id = parameter->value_id;
    IREE_RETURN_IF_ERROR(loom_testbench_parameter_sample_value(
        parameter, parameter_sample_ordinal, &scalar->value));
    scalar->location = parameter->op->location;
    scalar->required = false;
  }

  for (iree_host_size_t i = 0; i < build->case_plan->value_source_count; ++i) {
    const loom_testbench_value_source_plan_t* source =
        &build->case_plan->value_sources[i];
    if (source->kind != LOOM_TESTBENCH_VALUE_SOURCE_LITERAL ||
        !loom_type_is_scalar(
            loom_module_value_type(build->module, source->value_id))) {
      continue;
    }
    build->scalars[build->scalar_count++] = (loom_testbench_case_scalar_t){
        .value_id = source->value_id,
        .value = source->literal.value,
        .location = source->op->location,
        .required = false,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_require_scalar(
    loom_testbench_case_command_program_build_t* build,
    loom_value_id_t value_id, iree_host_size_t launch_index,
    iree_string_view_t group_name, iree_host_size_t operand_index) {
  for (iree_host_size_t i = 0; i < build->scalar_count; ++i) {
    if (build->scalars[i].value_id == value_id) {
      build->scalars[i].required = true;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "case '@%.*s' selected sweep sample %" PRIhsz " kernel launch %" PRIhsz
      " %.*s %" PRIhsz " value %u is not known without execution",
      (int)build->case_plan->name.size, build->case_plan->name.data,
      build->sample.ordinal, launch_index, (int)group_name.size,
      group_name.data, operand_index, (unsigned)value_id);
}

typedef struct loom_testbench_case_type_scalar_requirement_t {
  loom_testbench_case_command_program_build_t* build;
  iree_host_size_t launch_index;
  iree_host_size_t argument_index;
} loom_testbench_case_type_scalar_requirement_t;

static iree_status_t loom_testbench_case_command_program_require_type_scalar(
    loom_value_id_t value_id, void* user_data) {
  loom_testbench_case_type_scalar_requirement_t* requirement = user_data;
  return loom_testbench_case_command_program_require_scalar(
      requirement->build, value_id, requirement->launch_index,
      IREE_SV("argument type dependency"), requirement->argument_index);
}

static iree_status_t loom_testbench_case_command_program_add_binding(
    loom_testbench_case_command_program_build_t* build,
    loom_value_id_t value_id,
    const loom_testbench_storage_identity_t* storage_identity,
    iree_host_size_t launch_index, iree_host_size_t argument_index) {
  if (storage_identity->root_value_id == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "case '@%.*s' selected sweep sample %" PRIhsz
                            " kernel launch %" PRIhsz " argument %" PRIhsz
                            " value %u has no storage provenance",
                            (int)build->case_plan->name.size,
                            build->case_plan->name.data, build->sample.ordinal,
                            launch_index, argument_index, (unsigned)value_id);
  }
  for (iree_host_size_t i = 0; i < build->binding_count; ++i) {
    if (build->binding_value_ids[i] == storage_identity->root_value_id) {
      return iree_ok_status();
    }
  }
  if (build->binding_count >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "case command program has too many bindings");
  }
  const iree_host_size_t required_count = build->binding_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      build->scratch_arena, build->binding_count, required_count,
      sizeof(*build->binding_value_ids), &build->binding_capacity,
      (void**)&build->binding_value_ids));
  build->binding_value_ids[build->binding_count++] =
      storage_identity->root_value_id;
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_preflight_launches(
    loom_testbench_case_command_program_build_t* build) {
  for (iree_host_size_t launch_index = 0;
       launch_index < build->case_plan->kernel_launch_count; ++launch_index) {
    const loom_testbench_invocation_plan_t* invocation =
        build->case_plan->kernel_launches[launch_index];
    IREE_ASSERT_NE(invocation->execution_epoch,
                   LOOM_TESTBENCH_EXECUTION_EPOCH_INVALID);

    for (iree_host_size_t workload_index = 0;
         workload_index < invocation->workload_count; ++workload_index) {
      const loom_value_id_t value_id =
          invocation->workload_value_ids[workload_index];
      IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_require_scalar(
          build, value_id, launch_index, IREE_SV("workload"), workload_index));
    }
    for (iree_host_size_t argument_index = 0;
         argument_index < invocation->input_count; ++argument_index) {
      const loom_value_id_t value_id =
          invocation->input_value_ids[argument_index];
      const loom_type_t type = loom_module_value_type(build->module, value_id);
      if (loom_type_is_scalar(type)) {
        IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_require_scalar(
            build, value_id, launch_index, IREE_SV("argument"),
            argument_index));
        continue;
      }
      if (!loom_testbench_case_is_buffer_value(type)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "case '@%.*s' kernel launch %" PRIhsz " argument %" PRIhsz
            " type is not representable in a command program",
            (int)build->case_plan->name.size, build->case_plan->name.data,
            launch_index, argument_index);
      }
      const loom_testbench_storage_identity_t* storage_identity =
          &build->module_plan->value_storage_identities[value_id];
      if (loom_type_is_buffer(type) && storage_identity->byte_offset != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "case '@%.*s' kernel launch %" PRIhsz " argument %" PRIhsz
            " is an opaque buffer alias at a nonzero byte offset",
            (int)build->case_plan->name.size, build->case_plan->name.data,
            launch_index, argument_index);
      }
      IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_add_binding(
          build, value_id, storage_identity, launch_index, argument_index));
      loom_testbench_case_type_scalar_requirement_t requirement = {
          .build = build,
          .launch_index = launch_index,
          .argument_index = argument_index,
      };
      IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
          build->module, type,
          loom_testbench_case_command_program_require_type_scalar,
          &requirement));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_add_symbol(
    loom_module_t* module, loom_symbol_ref_t* out_ref) {
  *out_ref = loom_symbol_ref_null();
  for (iree_host_size_t ordinal = module->symbols.count;; ++ordinal) {
    char name_buffer[64];
    const int name_length =
        iree_snprintf(name_buffer, sizeof(name_buffer),
                      "__loom_case_sample_%" PRIhsz, ordinal);
    if (name_length < 0 ||
        (iree_host_size_t)name_length >= sizeof(name_buffer)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "case command program symbol is too long");
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

static iree_status_t loom_testbench_case_command_program_build_constant(
    loom_testbench_case_command_program_build_t* build, loom_value_id_t source,
    loom_attribute_t value, loom_location_id_t location) {
  const loom_type_t type = loom_module_value_type(build->module, source);
  if (!loom_type_is_scalar(type)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "case scalar constant source %u is not scalar",
                            (unsigned)source);
  }
  loom_op_t* constant_op = NULL;
  loom_value_id_t target = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    IREE_RETURN_IF_ERROR(loom_index_constant_build(&build->builder, value, type,
                                                   location, &constant_op));
    target = loom_index_constant_result(constant_op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
        &build->builder, value, type, location, &constant_op));
    target = loom_scalar_constant_result(constant_op);
  }
  return loom_ir_remap_map_value(&build->remap, source, target);
}

static iree_status_t loom_testbench_case_command_program_build_scalars(
    loom_testbench_case_command_program_build_t* build) {
  for (iree_host_size_t i = 0; i < build->scalar_count; ++i) {
    const loom_testbench_case_scalar_t* scalar = &build->scalars[i];
    if (!scalar->required) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_build_constant(
        build, scalar->value_id, scalar->value, scalar->location));
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_offset_constant(
    loom_testbench_case_command_program_build_t* build,
    iree_device_size_t byte_offset, loom_location_id_t location,
    loom_value_id_t* out_value) {
  for (iree_host_size_t i = 0; i < build->cached_offset_count; ++i) {
    if (build->cached_byte_offsets[i] == byte_offset) {
      *out_value = build->cached_offset_values[i];
      return iree_ok_status();
    }
  }
  if (byte_offset > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "case tensor view offset exceeds i64");
  }
  const iree_host_size_t required_count = build->cached_offset_count + 1;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      build->scratch_arena, build->cached_offset_count, required_count,
      sizeof(*build->cached_byte_offsets), &build->cached_byte_offset_capacity,
      (void**)&build->cached_byte_offsets));
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      build->scratch_arena, build->cached_offset_count, required_count,
      sizeof(*build->cached_offset_values),
      &build->cached_offset_value_capacity,
      (void**)&build->cached_offset_values));
  loom_op_t* constant_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &build->builder, loom_attr_i64((int64_t)byte_offset),
      loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), location, &constant_op));
  const loom_value_id_t value = loom_index_constant_result(constant_op);
  build->cached_byte_offsets[build->cached_offset_count] = byte_offset;
  build->cached_offset_values[build->cached_offset_count] = value;
  ++build->cached_offset_count;
  *out_value = value;
  return iree_ok_status();
}

static iree_host_size_t loom_testbench_case_command_program_find_binding(
    const loom_testbench_case_command_program_build_t* build,
    loom_value_id_t root_value_id) {
  for (iree_host_size_t i = 0; i < build->binding_count; ++i) {
    if (build->binding_value_ids[i] == root_value_id) {
      return i;
    }
  }
  IREE_ASSERT_UNREACHABLE("preflighted storage root has no command binding");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_testbench_case_command_program_materialize_buffer(
    loom_testbench_case_command_program_build_t* build,
    loom_value_id_t source_value,
    const loom_testbench_storage_identity_t* storage_identity,
    loom_location_id_t location) {
  loom_value_id_t existing_value = LOOM_VALUE_ID_INVALID;
  if (loom_ir_remap_try_lookup_value(&build->remap, source_value,
                                     &existing_value)) {
    return iree_ok_status();
  }
  const iree_host_size_t binding_index =
      loom_testbench_case_command_program_find_binding(
          build, storage_identity->root_value_id);
  const loom_value_id_t root_argument = build->wrapper_arguments[binding_index];
  loom_type_t source_type = loom_module_value_type(build->module, source_value);
  if (loom_type_is_buffer(source_type)) {
    IREE_ASSERT_EQ(storage_identity->byte_offset, 0u);
    return loom_ir_remap_map_value(&build->remap, source_value, root_argument);
  }

  iree_status_t status =
      loom_ir_remap_type(&build->remap, source_type, &source_type);
  if (!iree_status_is_ok(status)) {
    return iree_status_annotate_f(
        status,
        "case '@%.*s' selected sweep sample %" PRIhsz
        " cannot resolve the type of shaped value %u",
        (int)build->case_plan->name.size, build->case_plan->name.data,
        build->sample.ordinal, (unsigned)source_value);
  }
  const uint8_t alignment = loom_type_is_view(source_type)
                                ? loom_type_view_alignment_override(source_type)
                                : 0;
  source_type.header = loom_type_make_header(
      LOOM_TYPE_VIEW, loom_type_element_type(source_type),
      loom_type_rank(source_type), loom_type_flags(source_type));
  source_type = loom_type_view_with_alignment(source_type, alignment);
  loom_value_id_t byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_offset_constant(
      build, storage_identity->byte_offset, location, &byte_offset));
  loom_op_t* view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(&build->builder, root_argument,
                                              byte_offset, source_type,
                                              location, &view_op));
  return loom_ir_remap_map_value(&build->remap, source_value,
                                 loom_buffer_view_result(view_op));
}

static iree_status_t loom_testbench_case_command_program_prepare_values(
    loom_testbench_case_command_program_build_t* build) {
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_build_scalars(build));
  for (iree_host_size_t launch_index = 0;
       launch_index < build->case_plan->kernel_launch_count; ++launch_index) {
    const loom_testbench_invocation_plan_t* invocation =
        build->case_plan->kernel_launches[launch_index];
    for (iree_host_size_t argument_index = 0;
         argument_index < invocation->input_count; ++argument_index) {
      const loom_value_id_t value_id =
          invocation->input_value_ids[argument_index];
      if (!loom_type_is_scalar(
              loom_module_value_type(build->module, value_id))) {
        IREE_RETURN_IF_ERROR(
            loom_testbench_case_command_program_materialize_buffer(
                build, value_id,
                &build->module_plan->value_storage_identities[value_id],
                invocation->op->location));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_resolve_operands(
    loom_testbench_case_command_program_build_t* build,
    iree_host_size_t launch_index, iree_string_view_t group_name,
    const loom_value_id_t* source_values, iree_host_size_t value_count,
    loom_value_id_t** out_values) {
  *out_values = NULL;
  if (value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(build->scratch_arena, value_count,
                                sizeof(**out_values), (void**)out_values));
  for (iree_host_size_t i = 0; i < value_count; ++i) {
    if (!loom_ir_remap_try_lookup_value(&build->remap, source_values[i],
                                        &(*out_values)[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "case '@%.*s' selected sweep sample %" PRIhsz
          " kernel launch %" PRIhsz " %.*s %" PRIhsz
          " value %u is not known without execution",
          (int)build->case_plan->name.size, build->case_plan->name.data,
          build->sample.ordinal, launch_index, (int)group_name.size,
          group_name.data, i, (unsigned)source_values[i]);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_build_launch(
    loom_testbench_case_command_program_build_t* build,
    iree_host_size_t launch_index) {
  const loom_testbench_invocation_plan_t* invocation =
      build->case_plan->kernel_launches[launch_index];
  loom_value_id_t* workloads = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_resolve_operands(
      build, launch_index, IREE_SV("workload"), invocation->workload_value_ids,
      invocation->workload_count, &workloads));
  loom_value_id_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_resolve_operands(
      build, launch_index, IREE_SV("argument"), invocation->input_value_ids,
      invocation->input_count, &arguments));
  loom_op_t* launch_op = NULL;
  return loom_kernel_launch_build(&build->builder, invocation->callee_ref,
                                  workloads, invocation->workload_count,
                                  arguments, invocation->input_count,
                                  invocation->op->location, &launch_op);
}

static iree_status_t loom_testbench_case_command_program_build_schedule(
    loom_testbench_case_command_program_build_t* build) {
  iree_host_size_t launch_index = 0;
  for (iree_host_size_t event_index = 0;
       event_index < build->case_plan->launch_schedule_event_count;
       ++event_index) {
    const loom_testbench_launch_schedule_event_t event =
        build->case_plan->launch_schedule_events[event_index];
    if (event.kind == LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_LAUNCH) {
      IREE_ASSERT_LT(launch_index, build->case_plan->kernel_launch_count);
      IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_build_launch(
          build, launch_index++));
      continue;
    }
    if (event.kind == LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_END) {
      IREE_ASSERT_NE(build->schedule_frame_count, 0u);
      loom_op_t* yield_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_command_yield_build(&build->builder, event.location, &yield_op));
      const loom_builder_ip_t parent_ip =
          build->schedule_frames[--build->schedule_frame_count].parent_ip;
      loom_builder_restore(&build->builder, parent_ip);
      continue;
    }

    const iree_host_size_t required_count = build->schedule_frame_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        build->scratch_arena, build->schedule_frame_count, required_count,
        sizeof(*build->schedule_frames), &build->schedule_frame_capacity,
        (void**)&build->schedule_frames));
    loom_op_t* schedule_op = NULL;
    loom_region_t* schedule_body = NULL;
    if (event.kind == LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_CONCURRENT_BEGIN) {
      IREE_RETURN_IF_ERROR(loom_command_concurrent_build(
          &build->builder, event.location, &schedule_op));
      schedule_body = loom_command_concurrent_body(schedule_op);
    } else {
      IREE_ASSERT_EQ(event.kind,
                     LOOM_TESTBENCH_LAUNCH_SCHEDULE_EVENT_SERIAL_BEGIN);
      IREE_RETURN_IF_ERROR(loom_command_serial_build(
          &build->builder, event.location, &schedule_op));
      schedule_body = loom_command_serial_body(schedule_op);
    }
    build->schedule_frames[build->schedule_frame_count++].parent_ip =
        loom_builder_enter_region(&build->builder, schedule_op, schedule_body);
  }
  IREE_ASSERT_EQ(build->schedule_frame_count, 0u);
  IREE_ASSERT_EQ(launch_index, build->case_plan->kernel_launch_count);
  return iree_ok_status();
}

static iree_status_t loom_testbench_case_command_program_build_impl(
    loom_testbench_case_command_program_build_t* build,
    loom_testbench_case_command_program_t* out_program) {
  IREE_RETURN_IF_ERROR(loom_testbench_case_command_program_validate_plan(
      build->module, build->module_plan, build->case_plan));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_initialize_tables(build));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_collect_scalars(build));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_preflight_launches(build));

  loom_type_t* binding_types = NULL;
  if (build->binding_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        build->scratch_arena, build->binding_count, sizeof(*binding_types),
        (void**)&binding_types));
    for (iree_host_size_t i = 0; i < build->binding_count; ++i) {
      binding_types[i] = loom_type_buffer();
    }
  }
  loom_symbol_ref_t root_ref = loom_symbol_ref_null();
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_add_symbol(build->module, &root_ref));

  loom_builder_initialize(build->module, &build->module->arena,
                          loom_module_block(build->module), &build->builder);
  loom_op_t* program_op = NULL;
  IREE_RETURN_IF_ERROR(loom_command_def_build(
      &build->builder, /*build_flags=*/0, /*visibility=*/0, /*retain=*/0,
      loom_symbol_ref_null(), root_ref,
      /*specializations_types=*/NULL,
      /*specializations_types_count=*/0, binding_types, build->binding_count,
      /*predicates=*/NULL, /*predicates_count=*/0,
      build->case_plan->op->location, &program_op));
  const loom_func_like_t program =
      loom_func_like_cast(build->module, program_op);
  uint16_t wrapper_argument_count = 0;
  build->wrapper_arguments =
      loom_func_like_arg_ids(program, &wrapper_argument_count);
  IREE_ASSERT_EQ(wrapper_argument_count, build->binding_count);

  const loom_builder_ip_t module_ip = loom_builder_enter_region(
      &build->builder, program_op, loom_command_def_body(program_op));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_prepare_values(build));
  IREE_RETURN_IF_ERROR(
      loom_testbench_case_command_program_build_schedule(build));
  loom_op_t* return_op = NULL;
  IREE_RETURN_IF_ERROR(loom_command_return_build(
      &build->builder, build->case_plan->op->location, &return_op));
  loom_builder_restore(&build->builder, module_ip);

  loom_value_id_t* retained_binding_value_ids = NULL;
  if (build->binding_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(&build->module->arena, build->binding_count,
                                  sizeof(*retained_binding_value_ids),
                                  (void**)&retained_binding_value_ids));
    memcpy(retained_binding_value_ids, build->binding_value_ids,
           build->binding_count * sizeof(*retained_binding_value_ids));
  }
  *out_program = (loom_testbench_case_command_program_t){
      .root_ref = root_ref,
      .binding_value_ids = retained_binding_value_ids,
      .binding_count = build->binding_count,
  };
  return iree_ok_status();
}

iree_status_t loom_testbench_case_command_program_build(
    loom_module_t* module, const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_case_plan_t* case_plan,
    loom_testbench_case_sample_t sample,
    loom_testbench_case_command_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(module_plan);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(out_program);
  *out_program = (loom_testbench_case_command_program_t){
      .root_ref = loom_symbol_ref_null(),
  };

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  loom_testbench_case_command_program_build_t build = {
      .module = module,
      .module_plan = module_plan,
      .case_plan = case_plan,
      .sample = sample,
      .scratch_arena = &scratch_arena,
  };
  const iree_status_t status =
      loom_testbench_case_command_program_build_impl(&build, out_program);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
