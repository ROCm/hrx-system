// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/vm/constants/materialization.h"

#include "iree/vm/bytecode/wire/core/constant.h"
#include "iree/vm/bytecode/wire/module_format.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/vm/descriptors.h"
#include "loom/target/arch/vm/lower/constants.h"
#include "loom/target/arch/vm/ops/ops.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/walk.h"

static const loom_pass_info_t
    loom_vm_materialize_constant_pool_pass_info_storage = {
        .name = IREE_SVL("vm-materialize-constant-pool"),
        .description =
            IREE_SVL("Materialize profitable Core VM constant pools."),
        .kind = LOOM_PASS_MODULE,
};

const loom_pass_info_t* loom_vm_materialize_constant_pool_pass_info(void) {
  return &loom_vm_materialize_constant_pool_pass_info_storage;
}

typedef struct loom_vm_constant_occurrence_t {
  // Inline low.const operation defining the value.
  loom_op_t* op;
  // Complete 64-bit value-register pattern defined by |op|.
  uint64_t bits;
  // Inline descriptor ordinal selecting the value width.
  uint16_t inline_descriptor_ordinal;
  // Selected constant-pool ordinal, or UINT32_MAX when left inline.
  uint32_t pool_ordinal;
} loom_vm_constant_occurrence_t;

static bool loom_vm_constant_occurrence_less(
    const loom_vm_constant_occurrence_t* lhs,
    const loom_vm_constant_occurrence_t* rhs) {
  return lhs->bits < rhs->bits;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_vm_constant_occurrence_sort,
                          loom_vm_constant_occurrence_t,
                          loom_vm_constant_occurrence_less)

typedef struct loom_vm_constant_scan_t {
  // Module owning all scanned operations.
  const loom_module_t* module;
  // Exact occurrence storage, or NULL during the counting scan.
  loom_vm_constant_occurrence_t* occurrences;
  // Number of pool-eligible occurrences seen.
  iree_host_size_t occurrence_count;
} loom_vm_constant_scan_t;

static iree_status_t loom_vm_constant_scan_op(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_low_const_isa(op)) return iree_ok_status();

  const uint32_t descriptor_ordinal = loom_low_const_descriptor(op);
  if (descriptor_ordinal == VM_CORE_DESCRIPTOR_REF_CONSTANT_POOL_LOAD_I32 ||
      descriptor_ordinal == VM_CORE_DESCRIPTOR_REF_CONSTANT_POOL_LOAD_I64) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "VM constant-pool materialization requires inline constants");
  }

  loom_vm_inline_constant_t constant = {0};
  loom_vm_constant_scan_t* scan = (loom_vm_constant_scan_t*)user_data;
  if (!loom_vm_inline_constant_try_decode(scan->module, op, &constant) ||
      (constant.descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_I32 &&
       constant.descriptor_ordinal != VM_CORE_DESCRIPTOR_REF_CONSTANT_I64)) {
    return iree_ok_status();
  }
  if (scan->occurrences != NULL) {
    scan->occurrences[scan->occurrence_count] = (loom_vm_constant_occurrence_t){
        .op = op,
        .bits = constant.bits,
        .inline_descriptor_ordinal = constant.descriptor_ordinal,
        .pool_ordinal = UINT32_MAX,
    };
  }
  ++scan->occurrence_count;
  return iree_ok_status();
}

static bool loom_vm_constant_function_uses_core(const loom_module_t* module,
                                                const loom_op_t* op) {
  if (!loom_low_func_def_isa(op)) return false;
  const loom_string_id_t descriptor_set_id =
      loom_low_func_def_descriptor_set(op);
  return descriptor_set_id < module->strings.count &&
         iree_string_view_equal(module->strings.entries[descriptor_set_id],
                                IREE_SV("vm.core"));
}

static iree_status_t loom_vm_constant_scan_module(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_vm_constant_scan_t* scan) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(loom_module_block(module), op) {
    if (!loom_vm_constant_function_uses_core(module, op)) continue;
    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    IREE_RETURN_IF_ERROR(loom_walk_function(
        module, loom_func_like_cast(module, op), LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_vm_constant_scan_op, scan}, arena,
        &walk_result));
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_constant_verify_unmaterialized(
    const loom_module_t* module) {
  const loom_op_t* op = NULL;
  loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
    if (loom_vm_constant_isa(op)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "VM constant-pool materialization has already run");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vm_constant_downstream_alignment(
    const loom_module_t* module, uint64_t* out_alignment) {
  uint64_t alignment = IREE_VM_BYTECODE_SECTION_MIN_ALIGNMENT;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(loom_region_const_entry_block(module->body), op) {
    if (!loom_global_rodata_def_isa(op)) continue;
    const loom_attribute_t alignment_attr =
        loom_op_const_attrs(op)[loom_global_rodata_def_alignment_ATTR_INDEX];
    if (loom_attr_is_absent(alignment_attr)) continue;
    const int64_t value = loom_attr_as_i64(alignment_attr);
    if (value <= 0 || value > UINT32_MAX ||
        !iree_is_power_of_two_uint64((uint64_t)value)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "VM rodata alignment must be a positive power-of-two u32");
    }
    alignment = iree_max(alignment, (uint64_t)value);
  }
  *out_alignment = alignment;
  return iree_ok_status();
}

static iree_host_size_t loom_vm_constant_inline_saving(
    uint16_t descriptor_ordinal) {
  switch (descriptor_ordinal) {
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I32:
      return sizeof(iree_vm_isa_constant_i32_record_t) -
             sizeof(iree_vm_isa_constant_pool_load_i32_record_t);
    case VM_CORE_DESCRIPTOR_REF_CONSTANT_I64:
      return sizeof(iree_vm_isa_constant_i64_record_t) -
             sizeof(iree_vm_isa_constant_pool_load_i64_record_t);
    default:
      IREE_ASSERT_UNREACHABLE("pool-eligible VM constant descriptor");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static uint32_t loom_vm_constant_select_groups(
    loom_vm_constant_occurrence_t* occurrences,
    iree_host_size_t occurrence_count, uint64_t downstream_alignment) {
  uint32_t pool_count = 0;
  uint64_t instruction_savings = 0;
  iree_host_size_t group_begin = 0;
  while (group_begin < occurrence_count && pool_count < 65536u) {
    iree_host_size_t group_end = group_begin + 1;
    while (group_end < occurrence_count &&
           occurrences[group_end].bits == occurrences[group_begin].bits) {
      ++group_end;
    }
    uint64_t group_savings = 0;
    for (iree_host_size_t i = group_begin; i < group_end; ++i) {
      group_savings += loom_vm_constant_inline_saving(
          occurrences[i].inline_descriptor_ordinal);
    }
    if (group_savings > sizeof(iree_vm_bytecode_v0_constant_cell_t)) {
      const uint32_t pool_ordinal = pool_count++;
      for (iree_host_size_t i = group_begin; i < group_end; ++i) {
        occurrences[i].pool_ordinal = pool_ordinal;
      }
      instruction_savings += group_savings;
    }
    group_begin = group_end;
  }

  // The new section adds one directory row and one cell per selected group.
  // Instruction savings change later offsets in four-byte steps. Sections are
  // at least eight-byte aligned and rodata may require stronger alignment, so
  // reserving alignment-4 bytes covers the worst possible padding increase.
  // Branch relaxation caused by shorter records can only improve the result.
  const uint64_t image_cost =
      sizeof(iree_vm_bytecode_v0_section_directory_row_t) +
      (uint64_t)pool_count * sizeof(iree_vm_bytecode_v0_constant_cell_t) +
      downstream_alignment - 4u;
  return instruction_savings > image_cost ? pool_count : 0;
}

static iree_status_t loom_vm_constant_build_pool_load(
    loom_rewriter_t* rewriter, loom_vm_constant_occurrence_t* occurrence) {
  const uint16_t descriptor_ordinal =
      occurrence->inline_descriptor_ordinal ==
              VM_CORE_DESCRIPTOR_REF_CONSTANT_I32
          ? VM_CORE_DESCRIPTOR_REF_CONSTANT_POOL_LOAD_I32
          : VM_CORE_DESCRIPTOR_REF_CONSTANT_POOL_LOAD_I64;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_vm_core_descriptor_set();
  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(descriptor_set, descriptor_ordinal);
  IREE_ASSERT(descriptor != NULL);
  IREE_ASSERT_EQ(descriptor->immediate_count, 1u);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[descriptor->immediate_start];

  loom_named_attr_t pool_attr = {
      .value = loom_attr_i64(occurrence->pool_ordinal),
  };
  const iree_string_view_t immediate_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      &rewriter->builder, immediate_name, &pool_attr.name_id));

  const loom_value_id_t old_result = loom_low_const_result(occurrence->op);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, old_result);
  loom_builder_set_before(&rewriter->builder, occurrence->op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* replacement_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &rewriter->builder, descriptor_set, descriptor,
      loom_make_named_attr_slice(&pool_attr, 1), result_type,
      occurrence->op->location, &replacement_op));
  const loom_value_id_t replacement_result =
      loom_low_const_result(replacement_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, occurrence->op, &replacement_result, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, occurrence->op,
                                                  &replacement_result, 1);
}

static iree_status_t loom_vm_constant_materialize_selected(
    loom_rewriter_t* rewriter, loom_vm_constant_occurrence_t* occurrences,
    iree_host_size_t occurrence_count, uint32_t pool_count) {
  loom_builder_set_block(&rewriter->builder,
                         loom_module_block(rewriter->module));
  uint32_t built_pool_count = 0;
  for (iree_host_size_t i = 0; i < occurrence_count; ++i) {
    loom_vm_constant_occurrence_t* occurrence = &occurrences[i];
    if (occurrence->pool_ordinal == UINT32_MAX) continue;
    if (i == 0 || occurrences[i - 1].pool_ordinal != occurrence->pool_ordinal) {
      loom_op_t* pool_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vm_constant_build(
          &rewriter->builder, occurrence->pool_ordinal,
          (int64_t)occurrence->bits, occurrence->op->location, &pool_op));
      ++built_pool_count;
    }
  }
  IREE_ASSERT_EQ(built_pool_count, pool_count);

  for (iree_host_size_t i = 0; i < occurrence_count; ++i) {
    if (occurrences[i].pool_ordinal == UINT32_MAX) continue;
    IREE_RETURN_IF_ERROR(
        loom_vm_constant_build_pool_load(rewriter, &occurrences[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_vm_materialize_constant_pool_run(loom_pass_t* pass,
                                                    loom_module_t* module) {
  IREE_RETURN_IF_ERROR(loom_vm_constant_verify_unmaterialized(module));

  loom_vm_constant_scan_t count_scan = {
      .module = module,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_constant_scan_module(module, pass->arena, &count_scan));
  if (count_scan.occurrence_count == 0) return iree_ok_status();

  loom_vm_constant_occurrence_t* occurrences = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(pass->arena, count_scan.occurrence_count,
                                sizeof(*occurrences), (void**)&occurrences));
  loom_vm_constant_scan_t collect_scan = {
      .module = module,
      .occurrences = occurrences,
  };
  IREE_RETURN_IF_ERROR(
      loom_vm_constant_scan_module(module, pass->arena, &collect_scan));
  IREE_ASSERT_EQ(collect_scan.occurrence_count, count_scan.occurrence_count);
  loom_vm_constant_occurrence_sort(occurrences, collect_scan.occurrence_count);

  uint64_t downstream_alignment = 0;
  IREE_RETURN_IF_ERROR(
      loom_vm_constant_downstream_alignment(module, &downstream_alignment));
  const uint32_t pool_count = loom_vm_constant_select_groups(
      occurrences, collect_scan.occurrence_count, downstream_alignment);
  if (pool_count == 0) return iree_ok_status();

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  const iree_status_t status = loom_vm_constant_materialize_selected(
      &rewriter, occurrences, collect_scan.occurrence_count, pool_count);
  if (iree_status_is_ok(status)) loom_pass_mark_changed(pass);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
