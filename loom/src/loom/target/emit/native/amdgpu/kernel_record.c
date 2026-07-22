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
#include "loom/ops/low/kernel.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/target/emit/native/amdgpu/kernel_entry.h"
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
  // Physical VGPR base required by the AMDGPU kernel ABI.
  uint32_t expected_location_base;
  // Kernel descriptor flag that requests this initialized system VGPR.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flag;
  // Diagnostic label for this workitem-id dimension.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_packed_workitem_id_t {
  // Kernel descriptor flags that request the packed logical dimensions.
  loom_amdgpu_kernel_descriptor_flags_t descriptor_flags;
  // Diagnostic label for the packed live-in source.
  iree_string_view_t label;
} loom_amdgpu_kernel_record_packed_workitem_id_t;

typedef struct loom_amdgpu_kernel_record_workgroup_id_t {
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

static bool loom_amdgpu_kernel_record_workgroup_dimension(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    uint32_t* out_dimension) {
  if (source_kind < LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X ||
      source_kind > LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_Z) {
    return false;
  }
  *out_dimension = (uint32_t)(source_kind -
                              LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKGROUP_ID_X);
  return true;
}

static bool loom_amdgpu_kernel_record_workitem_dimension(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind,
    uint32_t* out_dimension) {
  if (source_kind < LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X ||
      source_kind > LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_Z) {
    return false;
  }
  *out_dimension =
      (uint32_t)(source_kind - LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_X);
  return true;
}

static bool loom_amdgpu_kernel_record_packed_workitem_index(
    loom_amdgpu_hal_kernel_abi_source_kind_t source_kind, uint32_t* out_index) {
  switch (source_kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XY:
      *out_index = 0;
      return true;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_WORKITEM_ID_PACKED_XYZ:
      *out_index = 1;
      return true;
    default:
      return false;
  }
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

static iree_status_t loom_amdgpu_kernel_record_validate_hidden_sgpr_pair(
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
  const loom_low_allocation_assignment_t* dispatch_id_assignment = NULL;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        loom_amdgpu_hal_kernel_abi_live_in_source_kind(allocation->module,
                                                       assignment->value_id);
    switch (source_kind) {
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_PTR:
        if (dispatch_ptr_assignment != NULL) {
          return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "AMDGPU kernel emission found duplicate "
                                  "dispatch pointer live-ins");
        }
        dispatch_ptr_assignment = assignment;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_KERNARG_SEGMENT_PTR:
        if (kernarg_ptr_assignment != NULL) {
          return iree_make_status(
              IREE_STATUS_ALREADY_EXISTS,
              "AMDGPU kernel emission found duplicate kernarg segment pointer "
              "live-ins");
        }
        kernarg_ptr_assignment = assignment;
        break;
      case LOOM_AMDGPU_HAL_KERNEL_ABI_SOURCE_DISPATCH_ID:
        if (dispatch_id_assignment != NULL) {
          return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "AMDGPU kernel emission found duplicate "
                                  "dispatch ID live-ins");
        }
        dispatch_id_assignment = assignment;
        break;
      default:
        break;
    }
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
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_validate_hidden_sgpr_pair(
        dispatch_ptr_assignment, IREE_SV("dispatch pointer"), user_sgpr_count));
    user_sgpr_count += LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT;
    descriptor_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_PTR;
  }
  if (kernarg_ptr_assignment != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_validate_hidden_sgpr_pair(
        kernarg_ptr_assignment, IREE_SV("kernarg segment pointer"),
        user_sgpr_count));
    user_sgpr_count += LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT;
    descriptor_flags |=
        LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_KERNARG_SEGMENT_PTR;
  }
  if (dispatch_id_assignment != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_kernel_record_validate_hidden_sgpr_pair(
        dispatch_id_assignment, IREE_SV("dispatch ID"), user_sgpr_count));
    user_sgpr_count += LOOM_AMDGPU_KERNEL_RECORD_HIDDEN_PTR_USER_SGPR_COUNT;
    descriptor_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_DISPATCH_ID;
  }

  *out_hidden_user_sgprs = (loom_amdgpu_kernel_record_hidden_user_sgprs_t){
      .descriptor_flags = descriptor_flags,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_kernel_record_collect_descriptor_flags(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation, uint32_t system_sgpr_base,
    bool target_has_packed_workitem_id, bool target_has_cluster_launch_state,
    loom_amdgpu_kernel_descriptor_flags_t* out_flags) {
  *out_flags = 0;
  static const loom_amdgpu_kernel_record_workgroup_id_t workgroup_ids[] = {
      {
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X,
          .label = IREE_SVL("workgroup_id.x"),
      },
      {
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y,
          .label = IREE_SVL("workgroup_id.y"),
      },
      {
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z,
          .label = IREE_SVL("workgroup_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_workitem_id_t workitem_ids[] = {
      {
          .expected_location_base = 0,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X,
          .label = IREE_SVL("workitem_id.x"),
      },
      {
          .expected_location_base = 1,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
          .label = IREE_SVL("workitem_id.y"),
      },
      {
          .expected_location_base = 2,
          .descriptor_flag =
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
          .label = IREE_SVL("workitem_id.z"),
      },
  };
  static const loom_amdgpu_kernel_record_packed_workitem_id_t
      packed_workitem_ids[] = {
          {
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y,
              .label = IREE_SVL("packed workitem_id.x/y"),
          },
          {
              .descriptor_flags =
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_X |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Y |
                  LOOM_AMDGPU_KERNEL_DESCRIPTOR_SYSTEM_VGPR_WORKITEM_ID_Z,
              .label = IREE_SVL("packed workitem_id.x/y/z"),
          },
      };
  static const loom_amdgpu_kernel_descriptor_flags_t
      launch_workgroup_id_descriptor_flags[] = {
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y,
          LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z,
      };

  if (target_has_cluster_launch_state) {
    loom_target_workgroup_cluster_size_t cluster_size = {0};
    if (loom_low_kernel_def_static_workgroup_cluster_size(schedule->function_op,
                                                          &cluster_size)) {
      // A clustered dispatch requires the x enable bit even when the kernel
      // does not consume a coordinate. This is the launch-state enable for
      // TTMP6/TTMP7/TTMP9 on gfx1250.
      *out_flags |= LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X;
    }

    const loom_low_descriptor_t* flat_cluster_workgroup_id_descriptor =
        loom_amdgpu_descriptor_ref_descriptor(
            schedule->target.descriptor_set,
            LOOM_AMDGPU_DESCRIPTOR_REF_S_GETREG_B32_CLUSTER_WORKGROUP_FLAT_ID);
    if (flat_cluster_workgroup_id_descriptor != NULL) {
      for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
        if (schedule->nodes[i].descriptor ==
            flat_cluster_workgroup_id_descriptor) {
          *out_flags |=
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_X |
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Y |
              LOOM_AMDGPU_KERNEL_DESCRIPTOR_ENABLE_SGPR_WORKGROUP_ID_Z;
          break;
        }
      }
    }
  }

  uint32_t found_workgroup_mask = 0;
  const loom_low_allocation_assignment_t*
      workgroup_id_assignments[IREE_ARRAYSIZE(workgroup_ids)] = {0};
  uint32_t found_workitem_mask = 0;
  bool found_packed_workitem_id = false;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->value_class.type_kind != LOOM_TYPE_REGISTER) {
      continue;
    }
    const loom_amdgpu_hal_kernel_abi_source_kind_t source_kind =
        loom_amdgpu_hal_kernel_abi_live_in_source_kind(allocation->module,
                                                       assignment->value_id);
    const loom_amdgpu_hal_kernel_abi_launch_workgroup_id_flags_t launch_flags =
        loom_amdgpu_hal_kernel_abi_source_launch_workgroup_id_flags(
            source_kind);
    if (launch_flags != 0) {
      for (uint32_t dimension = 0;
           dimension < IREE_ARRAYSIZE(launch_workgroup_id_descriptor_flags);
           ++dimension) {
        if (iree_any_bit_set(launch_flags, 1u << dimension)) {
          *out_flags |= launch_workgroup_id_descriptor_flags[dimension];
        }
      }
      continue;
    }
    uint32_t workgroup_dimension = 0;
    if (loom_amdgpu_kernel_record_workgroup_dimension(source_kind,
                                                      &workgroup_dimension)) {
      const loom_amdgpu_kernel_record_workgroup_id_t* row =
          &workgroup_ids[workgroup_dimension];
      const uint32_t workgroup_bit = 1u << workgroup_dimension;
      if (iree_any_bit_set(found_workgroup_mask, workgroup_bit)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)row->label.size, row->label.data);
      }
      found_workgroup_mask |= workgroup_bit;
      workgroup_id_assignments[workgroup_dimension] = assignment;
      continue;
    }
    uint32_t packed_workitem_index = 0;
    if (loom_amdgpu_kernel_record_packed_workitem_index(
            source_kind, &packed_workitem_index)) {
      const loom_amdgpu_kernel_record_packed_workitem_id_t* row =
          &packed_workitem_ids[packed_workitem_index];
      if (!target_has_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission target does not support %.*s live-ins",
            (int)row->label.size, row->label.data);
      }
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate packed workitem-id "
            "live-ins");
      }
      if (found_workitem_mask != 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel emission cannot mix packed and unpacked "
            "workitem-id live-ins");
      }
      found_packed_workitem_id = true;
      if (assignment->location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          assignment->location_base != 0 || assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v0",
            (int)row->label.size, row->label.data);
      }
      *out_flags |= row->descriptor_flags;
      continue;
    }
    uint32_t workitem_dimension = 0;
    if (loom_amdgpu_kernel_record_workitem_dimension(source_kind,
                                                     &workitem_dimension)) {
      const loom_amdgpu_kernel_record_workitem_id_t* row =
          &workitem_ids[workitem_dimension];
      if (found_packed_workitem_id) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU kernel emission cannot mix packed and unpacked "
            "workitem-id live-ins");
      }
      const uint32_t workitem_bit = 1u << workitem_dimension;
      if (iree_any_bit_set(found_workitem_mask, workitem_bit)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "AMDGPU kernel emission found duplicate %.*s live-ins",
            (int)row->label.size, row->label.data);
      }
      found_workitem_mask |= workitem_bit;
      if (assignment->location_kind !=
              LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
          assignment->location_base != row->expected_location_base ||
          assignment->location_count != 1) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "AMDGPU kernel emission requires the %.*s live-in to be fixed to "
            "v%" PRIu32,
            (int)row->label.size, row->label.data, row->expected_location_base);
      }
      *out_flags |= row->descriptor_flag;
    }
  }
  if (target_has_packed_workitem_id &&
      iree_any_bit_set(found_workitem_mask, (1u << 1) | (1u << 2))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU kernel emission requires packed workitem-id live-ins when "
        "workitem_id.y/z are used on this target");
  }
  uint32_t workgroup_id_sgpr = system_sgpr_base;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workgroup_ids); ++i) {
    if (!iree_any_bit_set(found_workgroup_mask, 1u << i)) {
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
  const iree_host_size_t argument_count = abi_layout->parameter_count;
  if (argument_count == 0) {
    return iree_ok_status();
  }
  loom_amdgpu_metadata_argument_t* arguments = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, argument_count, sizeof(*arguments), (void**)&arguments));
  for (iree_host_size_t i = 0; i < abi_layout->resource_count; ++i) {
    const loom_amdgpu_hal_kernarg_resource_t* resource =
        &abi_layout->resources[i];
    IREE_ASSERT_LT(resource->parameter_index, argument_count);
    arguments[resource->parameter_index] = (loom_amdgpu_metadata_argument_t){
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
    IREE_ASSERT_LT(direct_arg->parameter_index, argument_count);
    arguments[direct_arg->parameter_index] = (loom_amdgpu_metadata_argument_t){
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
    const bool has_flat_workgroup_size =
        loom_target_workgroup_size_flat_product_u32(
            required, out_max_flat_workgroup_size);
    IREE_ASSERT(has_flat_workgroup_size,
                "validated HAL kernel workgroup size fits uint32_t");
    (void)has_flat_workgroup_size;
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
      loom_amdgpu_target_processor_from_resolved_target(&schedule->target);
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
      schedule, allocation, user_sgpr_count,
      loom_amdgpu_processor_kernel_descriptor_has_flags(
          processor, LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID),
      loom_amdgpu_processor_info_has_flags(
          processor, LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE),
      &topology_descriptor_flags));
  descriptor_flags |= topology_descriptor_flags;
  uint32_t system_vgpr_workitem_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_kernel_descriptor_workitem_id_mode_from_flags(
          descriptor_flags, &system_vgpr_workitem_id));

  const loom_amdgpu_kernel_entry_envelope_t* entry_envelope =
      loom_amdgpu_kernel_entry_envelope_for_processor(processor);
  uint32_t next_free_sgpr = preflight->next_free_sgpr > user_sgpr_count
                                ? preflight->next_free_sgpr
                                : user_sgpr_count;
  if (entry_envelope->minimum_sgpr_count > next_free_sgpr) {
    next_free_sgpr = entry_envelope->minimum_sgpr_count;
  }
  uint32_t next_free_vgpr = preflight->next_free_vgpr;
  if (entry_envelope->minimum_vgpr_count > next_free_vgpr) {
    next_free_vgpr = entry_envelope->minimum_vgpr_count;
  }

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
  loom_target_workgroup_cluster_size_t workgroup_cluster_size = {0};
  const bool has_workgroup_cluster_size =
      loom_low_kernel_def_static_workgroup_cluster_size(
          schedule->function_op, &workgroup_cluster_size);
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
              .vgpr_count = next_free_vgpr,
              .max_flat_workgroup_size = max_flat_workgroup_size,
              .required_workgroup_size = hal_kernel->required_workgroup_size,
              .has_required_workgroup_size = has_required_workgroup_size,
              .workgroup_cluster_size = workgroup_cluster_size,
              .has_workgroup_cluster_size = has_workgroup_cluster_size,
              .gfx1250_revision =
                  loom_amdgpu_target_record_effective_gfx1250_revision(
                      schedule->target.target_op),
              .arguments = arguments,
              .argument_count = abi_layout->parameter_count,
          },
      .descriptor_flags = descriptor_flags,
      .system_vgpr_workitem_id = system_vgpr_workitem_id,
      .user_sgpr_count = user_sgpr_count,
  };
  return iree_ok_status();
}
