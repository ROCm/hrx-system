// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_record.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/packet.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/emit/native/amdgpu/preflight.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/launch.h"

#define LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT 2u

typedef struct loom_amdgpu_kernel_record_hidden_user_sgprs_t {
  // Descriptor flags requested by hidden user-SGPR live-ins.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Number of enabled hidden user SGPRs in AMDHSA order.
  uint32_t user_sgpr_count;
} loom_amdgpu_kernel_record_hidden_user_sgprs_t;

typedef struct loom_amdgpu_kernel_record_workitem_id_t {
  // Predicate identifying the ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // Physical VGPR base required by the AMDGPU kernel ABI.
  uint32_t expected_location_base;
  // Kernel descriptor flag that requests this initialized system VGPR.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flag;
  // Diagnostic label for this workitem-id dimension.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_packed_workitem_id_t {
  // Predicate identifying the packed ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // Kernel descriptor flags that request the packed logical dimensions.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Diagnostic label for the packed live-in source.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_packed_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_workgroup_id_t {
  // Predicate identifying the ABI live-in value.
  bool (*is_live_in)(const loom_module_t* module, loom_value_id_t value_id);
  // Kernel descriptor flag that requests this initialized system SGPR.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flag;
  // Diagnostic label for this workgroup-id dimension.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_workgroup_id_t;

static bool loom_amdgpu_kernel_record_symbol_ref_equal(loom_symbol_ref_t lhs,
                                                       loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static iree_status_t loom_amdgpu_kernel_record_concat3(
    iree_string_view_t a, iree_string_view_t b, iree_string_view_t c,
    iree_string_view_t* out_value, iree_arena_allocator_t* arena) {
  *out_value = iree_string_view_empty();
  iree_host_size_t prefix_length = 0;
  iree_host_size_t length = 0;
  if (!iree_host_size_checked_add(a.size, b.size, &prefix_length) ||
      !iree_host_size_checked_add(prefix_length, c.size, &length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU kernel record string length overflows");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, length, (void**)&data));
  char* cursor = data;
  if (!iree_string_view_is_empty(a)) {
    memcpy(cursor, a.data, a.size);
    cursor += a.size;
  }
  if (!iree_string_view_is_empty(b)) {
    memcpy(cursor, b.data, b.size);
    cursor += b.size;
  }
  if (!iree_string_view_is_empty(c)) {
    memcpy(cursor, c.data, c.size);
  }
  *out_value = iree_make_string_view(data, length);
  return iree_ok_status();
}

static bool loom_amdgpu_kernel_record_symbol_start_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_' ||
         c == '$';
}

static bool loom_amdgpu_kernel_record_symbol_continue_char(char c) {
  return loom_amdgpu_kernel_record_symbol_start_char(c) ||
         (c >= '0' && c <= '9') || c == '.';
}

static iree_status_t loom_amdgpu_kernel_record_validate_symbol(
    iree_string_view_t symbol) {
  if (iree_string_view_is_empty(symbol)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission symbol is required");
  }
  if (!loom_amdgpu_kernel_record_symbol_start_char(symbol.data[0])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission symbol '%.*s' has an invalid first character",
        (int)symbol.size, symbol.data);
  }
  for (iree_host_size_t i = 1; i < symbol.size; ++i) {
    if (!loom_amdgpu_kernel_record_symbol_continue_char(symbol.data[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel emission symbol '%.*s' contains an invalid character",
          (int)symbol.size, symbol.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_symbol_name(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_string_view_t* out_symbol) {
  *out_symbol = iree_string_view_empty();
  loom_symbol_ref_t symbol_ref = loom_low_function_callee(function_op);
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission function symbol is invalid");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function symbol has no "
                            "module string");
  }
  *out_symbol = module->strings.entries[symbol->name_id];
  return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_export_symbol(
    const loom_low_resolved_target_t* target, const loom_module_t* module,
    const loom_op_t* function_op, iree_string_view_t* out_symbol) {
  if (!iree_string_view_is_empty(
          target->bundle_storage.export_plan.export_symbol)) {
    *out_symbol = target->bundle_storage.export_plan.export_symbol;
    return loom_amdgpu_kernel_record_validate_symbol(*out_symbol);
  }
  return loom_amdgpu_kernel_record_symbol_name(module, function_op, out_symbol);
}

static iree_status_t loom_amdgpu_kernel_record_validate_target(
    const loom_low_resolved_target_t* target) {
  const loom_target_snapshot_t* snapshot = &target->bundle_storage.snapshot;
  const loom_target_export_plan_t* export_plan =
      &target->bundle_storage.export_plan;
  if (target->descriptor_set == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission target bundle is required");
  }
  if (snapshot->codegen_format != LOOM_TARGET_CODEGEN_FORMAT_LOW_NATIVE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires low_native codegen snapshots");
  }
  if (snapshot->artifact_format != LOOM_TARGET_ARTIFACT_FORMAT_ELF) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires ELF artifacts");
  }
  if (export_plan->abi_kind != LOOM_TARGET_ABI_HAL_KERNEL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires a HAL kernel ABI");
  }
  if (export_plan->linkage != LOOM_TARGET_LINKAGE_DEFAULT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires default linkage");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_validate_function_shape(
    const loom_op_t* function_op) {
  if (!loom_low_function_def_isa(function_op)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU kernel emission requires low.func.def or low.kernel.def");
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU kernel emission function body is required");
  }
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  if (entry_block->arg_count != 0 || function_op->result_count != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires ABI-lowered kernels with no low "
        "function arguments or results");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_segment_usage(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_segment_sizes_t* out_usage) {
  *out_usage = (loom_amdgpu_storage_layout_segment_sizes_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_collect_segment_sizes(
      module, function_op, out_usage));
  if (out_usage->group_segment_fixed_size > UINT32_MAX ||
      out_usage->private_segment_fixed_size > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU kernel emission fixed segment sizes exceed metadata limits");
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_validate_hidden_ptr_assignment(
    const loom_low_allocation_assignment_t* assignment,
    iree_string_view_t label, uint32_t expected_location_base) {
  if (assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
      assignment->location_base != expected_location_base ||
      assignment->location_count !=
          LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
        "s[%" PRIu32 ":%" PRIu32 "]",
        (int)label.size, label.data, expected_location_base,
        expected_location_base +
            LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT - 1);
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_hidden_user_sgprs(
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    loom_amdgpu_kernel_record_hidden_user_sgprs_t* out_hidden_user_sgprs) {
  *out_hidden_user_sgprs = (loom_amdgpu_kernel_record_hidden_user_sgprs_t){0};

  const loom_low_allocation_assignment_t* dispatch_ptr_assignment = NULL;
  const loom_low_allocation_assignment_t* kernarg_ptr_assignment = NULL;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    if (loom_amdgpu_hal_kernel_abi_is_dispatch_ptr_live_in(
            allocation->module, assignment->value_id)) {
      if (dispatch_ptr_assignment != NULL) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate dispatch pointer live-ins");
      }
      dispatch_ptr_assignment = assignment;
      continue;
    }
    if (!loom_amdgpu_hal_kernel_abi_is_kernarg_segment_ptr_live_in(
            allocation->module, assignment->value_id)) {
      continue;
    }
    if (kernarg_ptr_assignment != NULL) {
      return iree_make_status(
          IREE_STATUS_ALREADY_EXISTS,
          "AMDGPU kernel emission found duplicate kernarg segment pointer "
          "live-ins");
    }
    kernarg_ptr_assignment = assignment;
  }

  if (abi_layout->uses_kernarg_segment_ptr && kernarg_ptr_assignment == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires a kernarg segment pointer live-in "
        "when the HAL ABI layout uses kernargs");
  }

  uint32_t user_sgpr_count = 0;
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags = 0;
  if (dispatch_ptr_assignment != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_kernel_record_validate_hidden_ptr_assignment(
            dispatch_ptr_assignment, IREE_SV("dispatch pointer"),
            user_sgpr_count));
    user_sgpr_count += LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT;
    descriptor_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR;
  }
  if (kernarg_ptr_assignment != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_kernel_record_validate_hidden_ptr_assignment(
            kernarg_ptr_assignment, IREE_SV("kernarg segment pointer"),
            user_sgpr_count));
    user_sgpr_count += LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT;
    descriptor_flags |=
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  }

  *out_hidden_user_sgprs = (loom_amdgpu_kernel_record_hidden_user_sgprs_t){
      .descriptor_flags = descriptor_flags,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_descriptor_flags(
    const loom_low_allocation_table_t* allocation, uint32_t system_sgpr_base,
    bool target_has_packed_workitem_id,
    loom_amdgpu_kernel_descriptor_flags_t* out_flags) {
  *out_flags = 0;
  static const loom_amdgpu_kernel_record_workgroup_id_t workgroup_ids[] = {
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_x_live_in,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X,
          .label = IREE_SVL("workgroup_id.x"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_y_live_in,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y,
          .label = IREE_SVL("workgroup_id.y"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workgroup_id_z_live_in,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z,
          .label = IREE_SVL("workgroup_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_workitem_id_t workitem_ids[] = {
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_x_live_in,
          .expected_location_base = 0,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X,
          .label = IREE_SVL("workitem_id.x"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_y_live_in,
          .expected_location_base = 1,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
          .label = IREE_SVL("workitem_id.y"),
      },
      {
          .is_live_in = loom_amdgpu_hal_kernel_abi_is_workitem_id_z_live_in,
          .expected_location_base = 2,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
          .label = IREE_SVL("workitem_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_packed_workitem_id_t
      packed_workitem_ids[] = {
          {
              .is_live_in =
                  loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xy_live_in,
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
              .label = IREE_SVL("packed workitem_id.x/y"),
          },
          {
              .is_live_in =
                  loom_amdgpu_hal_kernel_abi_is_workitem_id_packed_xyz_live_in,
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
              .label = IREE_SVL("packed workitem_id.x/y/z"),
          },
      };
  bool found_workgroup_ids[IREE_ARRAYSIZE(workgroup_ids)] = {0};
  const loom_low_allocation_assignment_t*
      workgroup_id_assignments[IREE_ARRAYSIZE(workgroup_ids)] = {0};
  bool found_workitem_ids[IREE_ARRAYSIZE(workitem_ids)] = {0};
  bool found_packed_workitem_id = false;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(workgroup_ids); ++j) {
      if (!workgroup_ids[j].is_live_in(allocation->module,
                                       assignment->value_id)) {
        continue;
      }
      if (found_workgroup_ids[j]) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)workgroup_ids[j].label.size, workgroup_ids[j].label.data);
      }
      found_workgroup_ids[j] = true;
      workgroup_id_assignments[j] = assignment;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(packed_workitem_ids); ++j) {
      if (!packed_workitem_ids[j].is_live_in(allocation->module,
                                             assignment->value_id)) {
        continue;
      }
      if (!target_has_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission target does not support %.*s live-ins",
            (int)packed_workitem_ids[j].label.size,
            packed_workitem_ids[j].label.data);
      }
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate packed workitem-id "
            "live-ins");
      }
      for (iree_host_size_t k = 0; k < IREE_ARRAYSIZE(found_workitem_ids);
           ++k) {
        if (found_workitem_ids[k]) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "AMDGPU kernel emission cannot mix packed and unpacked "
              "workitem-id live-ins");
        }
      }
      found_packed_workitem_id = true;
      if (assignment->location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          assignment->location_base != 0 || assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v0",
            (int)packed_workitem_ids[j].label.size,
            packed_workitem_ids[j].label.data);
      }
      *out_flags |= packed_workitem_ids[j].descriptor_flags;
    }
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(workitem_ids); ++j) {
      if (!workitem_ids[j].is_live_in(allocation->module,
                                      assignment->value_id)) {
        continue;
      }
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel emission cannot mix packed and unpacked "
            "workitem-id live-ins");
      }
      if (found_workitem_ids[j]) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)workitem_ids[j].label.size, workitem_ids[j].label.data);
      }
      found_workitem_ids[j] = true;
      if (assignment->location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          assignment->location_base != workitem_ids[j].expected_location_base ||
          assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v%" PRIu32,
            (int)workitem_ids[j].label.size, workitem_ids[j].label.data,
            workitem_ids[j].expected_location_base);
      }
      *out_flags |= workitem_ids[j].descriptor_flag;
    }
  }
  if (target_has_packed_workitem_id &&
      (found_workitem_ids[1] || found_workitem_ids[2])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires packed workitem-id live-ins when "
        "workitem_id.y/z are used on this target");
  }
  uint32_t workgroup_id_sgpr = system_sgpr_base;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workgroup_ids); ++i) {
    if (!found_workgroup_ids[i]) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        workgroup_id_assignments[i];
    if (assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->location_base != workgroup_id_sgpr ||
        assignment->location_count != 1) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
          "s%" PRIu32,
          (int)workgroup_ids[i].label.size, workgroup_ids[i].label.data,
          workgroup_id_sgpr);
    }
    *out_flags |= workgroup_ids[i].descriptor_flag;
    ++workgroup_id_sgpr;
  }
  if (iree_any_bit_set(
          *out_flags,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y;
  } else if (iree_any_bit_set(
                 *out_flags,
                 LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y)) {
    *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_build_metadata_arguments(
    const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout,
    const loom_amdgpu_metadata_argument_t** out_arguments,
    iree_arena_allocator_t* arena) {
  *out_arguments = NULL;
  const iree_host_size_t argument_count =
      abi_layout->resource_count + abi_layout->direct_arg_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*arguments), (void**)&arguments));
  for (iree_host_size_t i = 0; i < abi_layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &abi_layout->resources[i];
    arguments[i] = (loom_amdgpu_metadata_argument_t){
        .name = resource->name,
        .offset = resource->kernarg_offset,
        .size = resource->kernarg_size,
        .alignment = resource->kernarg_alignment,
        .kind = LOOM_AMDGPU_METADATA_ARGUMENT_GLOBAL_BUFFER,
        .address_space = IREE_SV("global"),
    };
  }
  for (iree_host_size_t i = 0; i < abi_layout->direct_arg_count; ++i) {
    const loom_amdgpu_hal_kernarg_direct_arg_t* direct_arg =
        &abi_layout->direct_args[i];
    arguments[abi_layout->resource_count + i] =
        (loom_amdgpu_metadata_argument_t){
            .name = direct_arg->name,
            .offset = direct_arg->kernarg_offset,
            .size = direct_arg->kernarg_size,
            .alignment = direct_arg->kernarg_alignment,
            .kind = LOOM_AMDGPU_METADATA_ARGUMENT_BY_VALUE,
        };
  }
  *out_arguments = arguments;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_max_flat_workgroup_size(
    const loom_target_snapshot_t* snapshot,
    const loom_target_hal_kernel_abi_t* hal_kernel,
    uint32_t* out_max_flat_workgroup_size) {
  const loom_target_workgroup_size_t* required =
      &hal_kernel->required_workgroup_size;
  if (loom_target_workgroup_size_is_concrete(required)) {
    if (!loom_target_workgroup_size_flat_product_u32(
            required, out_max_flat_workgroup_size)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "validated HAL kernel workgroup size overflows "
                              "uint32_t");
    }
    return iree_ok_status();
  }
  *out_max_flat_workgroup_size = hal_kernel->flat_workgroup_size_max != 0
                                     ? hal_kernel->flat_workgroup_size_max
                                     : snapshot->max_flat_workgroup_size;
  return iree_ok_status();
}

iree_status_t loom_amdgpu_kernel_record_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_kernel_record_options_t* options,
    loom_amdgpu_kernel_record_t* out_record,
    iree_arena_allocator_t* scratch_arena) {
  *out_record = (loom_amdgpu_kernel_record_t){0};
  loom_amdgpu_native_preflight_t derived_preflight = {0};
  const loom_amdgpu_native_preflight_t* preflight =
      options ? options->preflight : NULL;
  if (preflight != NULL) {
    if (preflight->schedule != schedule ||
        preflight->allocation != allocation) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel record preflight does not match the scheduled "
          "function");
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_native_preflight_analyze(
        schedule, allocation, /*options=*/NULL, &derived_preflight));
    preflight = &derived_preflight;
  }
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_target(&schedule->target));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_record_validate_function_shape(schedule->function_op));

  iree_string_view_t symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_export_symbol(
      &schedule->target, schedule->module, schedule->function_op, &symbol));

  loom_amdgpu_hal_kernel_abi_layout_t derived_abi_layout = {0};
  const loom_amdgpu_hal_kernel_abi_layout_t* abi_layout =
      options ? options->abi_layout : NULL;
  if (abi_layout != NULL) {
    if (abi_layout->function_op != schedule->function_op) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU kernel record ABI layout does not belong to the scheduled "
          "function");
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_layout_from_low(
        schedule->module, schedule->function_op, &derived_abi_layout,
        scratch_arena));
    abi_layout = &derived_abi_layout;
  }

  loom_amdgpu_storage_layout_segment_sizes_t segment_usage = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_segment_usage(
      schedule->module, schedule->function_op, &segment_usage));
  loom_amdgpu_kernel_record_hidden_user_sgprs_t hidden_user_sgprs = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_hidden_user_sgprs(
      allocation, abi_layout, &hidden_user_sgprs));

  const loom_target_hal_kernel_abi_t* hal_kernel =
      &schedule->target.bundle_storage.export_plan.hal_kernel;
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_resolved_target(schedule->module,
                                                        &schedule->target);
  if (processor == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU kernel emission requires an AMDGPU "
                            "processor target record");
  }
  const uint32_t wavefront_size =
      schedule->target.bundle_storage.snapshot.subgroup_size;
  IREE_ASSERT(
      loom_amdgpu_processor_supports_wavefront_size(processor, wavefront_size));

  const uint32_t user_sgpr_count = hidden_user_sgprs.user_sgpr_count;
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags =
      hidden_user_sgprs.descriptor_flags;
  loom_amdgpu_kernel_descriptor_flags_t topology_descriptor_flags = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_collect_descriptor_flags(
      allocation, user_sgpr_count,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          processor, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID),
      &topology_descriptor_flags));
  descriptor_flags |= topology_descriptor_flags;
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor_flags, &system_vgpr_workitem_id));

  const uint32_t next_free_sgpr = preflight->next_free_sgpr > user_sgpr_count
                                      ? preflight->next_free_sgpr
                                      : user_sgpr_count;

  iree_string_view_t target_id = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_amdhsa_target_id_format(
      processor, iree_string_view_empty(), scratch_arena, &target_id));
  iree_string_view_t descriptor_symbol = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_concat3(
      symbol, IREE_SV(".kd"), iree_string_view_empty(), &descriptor_symbol,
      scratch_arena));

  const loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_build_metadata_arguments(
      abi_layout, &arguments, scratch_arena));
  const bool has_required_workgroup_size =
      loom_target_workgroup_size_is_concrete(
          &hal_kernel->required_workgroup_size);
  uint32_t max_flat_workgroup_size = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_max_flat_workgroup_size(
      &schedule->target.bundle_storage.snapshot, hal_kernel,
      &max_flat_workgroup_size));

  *out_record = (loom_amdgpu_kernel_record_t){
      .symbol = symbol,
      .descriptor_symbol = descriptor_symbol,
      .target_id = target_id,
      .processor = processor,
      .abi_layout = *abi_layout,
      .metadata =
          {
              .name = symbol,
              .descriptor_symbol = descriptor_symbol,
              .kernarg_segment_size = abi_layout->kernarg_segment_size,
              .kernarg_segment_alignment =
                  abi_layout->kernarg_segment_alignment,
              .wavefront_size = wavefront_size,
              .group_segment_fixed_size =
                  (uint32_t)segment_usage.group_segment_fixed_size,
              .private_segment_fixed_size =
                  (uint32_t)segment_usage.private_segment_fixed_size,
              .sgpr_count = next_free_sgpr,
              .vgpr_count = preflight->next_free_vgpr,
              .max_flat_workgroup_size = max_flat_workgroup_size,
              .required_workgroup_size = hal_kernel->required_workgroup_size,
              .has_required_workgroup_size = has_required_workgroup_size,
              .arguments = arguments,
              .argument_count =
                  abi_layout->resource_count + abi_layout->direct_arg_count,
          },
      .descriptor_flags = descriptor_flags,
      .system_vgpr_workitem_id = system_vgpr_workitem_id,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}
