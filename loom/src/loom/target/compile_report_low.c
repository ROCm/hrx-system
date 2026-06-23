// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/compile_report_low.h"

#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/packet.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/low/ops.h"
#include "loom/target/registers.h"

static bool loom_target_compile_report_descriptor_semantic_tag_is(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor, iree_string_view_t tag) {
  if (descriptor_set == NULL || descriptor == NULL) {
    return false;
  }
  if (descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return false;
  }
  const iree_string_view_t semantic_tag = loom_low_descriptor_set_string(
      descriptor_set, descriptor->semantic_tag_string_offset);
  return iree_string_view_equal(semantic_tag, tag);
}

static iree_string_view_t loom_target_compile_report_descriptor_semantic_tag(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->semantic_tag_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->semantic_tag_string_offset);
}

static iree_string_view_t loom_target_compile_report_descriptor_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->key_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        descriptor->key_string_offset);
}

static bool loom_target_compile_report_string_contains(
    iree_string_view_t value, iree_string_view_t needle) {
  return iree_string_view_find(value, needle, /*pos=*/0) !=
         IREE_STRING_VIEW_NPOS;
}

static const loom_low_schedule_class_t*
loom_target_compile_report_descriptor_schedule_class(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return NULL;
  }
  return &descriptor_set->schedule_classes[descriptor->schedule_class_id];
}

static iree_string_view_t loom_target_compile_report_schedule_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  if (!iree_string_view_is_empty(node->schedule_class_name)) {
    return node->schedule_class_name;
  }
  const loom_low_schedule_class_t* schedule_class =
      loom_target_compile_report_descriptor_schedule_class(descriptor_set,
                                                           node->descriptor);
  if (schedule_class == NULL ||
      schedule_class->name_string_offset == LOOM_LOW_STRING_OFFSET_NONE) {
    return iree_string_view_empty();
  }
  return loom_low_descriptor_set_string(descriptor_set,
                                        schedule_class->name_string_offset);
}

static bool loom_target_compile_report_schedule_class_uses_resource_kind(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_class_t* schedule_class,
    loom_low_resource_kind_t kind) {
  if (descriptor_set == NULL || schedule_class == NULL) {
    return false;
  }
  const uint32_t end = (uint32_t)schedule_class->issue_use_start +
                       schedule_class->issue_use_count;
  if (end > descriptor_set->issue_use_count) {
    return false;
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &descriptor_set->issue_uses[schedule_class->issue_use_start + i];
    if (issue_use->resource_id >= descriptor_set->resource_count) {
      continue;
    }
    if (descriptor_set->resources[issue_use->resource_id].kind == kind) {
      return true;
    }
  }
  return false;
}

static bool loom_target_compile_report_low_branch_falls_through(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node, const loom_block_t* dest) {
  const uint32_t dest_block_index = loom_low_packet_block_index(schedule, dest);
  return dest_block_index != LOOM_LOW_PACKET_INDEX_NONE &&
         dest_block_index == node->block_index + 1;
}

static uint64_t
loom_target_compile_report_low_structural_control_transfer_count(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  if (node->op == NULL) {
    return 0;
  }
  if (loom_low_return_isa(node->op)) {
    return 1;
  }
  if (loom_low_br_isa(node->op)) {
    return loom_target_compile_report_low_branch_falls_through(
               schedule, node, loom_low_br_dest(node->op))
               ? 0
               : 1;
  }
  if (!loom_low_cond_br_isa(node->op)) {
    return 0;
  }

  const loom_block_t* true_dest = loom_low_cond_br_true_dest(node->op);
  const loom_block_t* false_dest = loom_low_cond_br_false_dest(node->op);
  const bool true_fallthrough =
      loom_target_compile_report_low_branch_falls_through(schedule, node,
                                                          true_dest);
  if (true_dest == false_dest) {
    return true_fallthrough ? 0 : 1;
  }
  const bool false_fallthrough =
      loom_target_compile_report_low_branch_falls_through(schedule, node,
                                                          false_dest);
  return true_fallthrough || false_fallthrough ? 1 : 2;
}

typedef struct loom_target_compile_report_low_node_features_t {
  // Node contributes scalar ALU work.
  bool scalar_alu;
  // Node contributes vector ALU work.
  bool vector_alu;
  // Node contributes matrix or tensor-core-like work.
  bool matrix;
  // Node contributes MFMA-family work.
  bool mfma;
  // Node contributes WMMA-family work.
  bool wmma;
  // Node contributes dot-product work.
  bool dot;
  // Node contributes global-memory work.
  bool global_memory;
  // Node contributes AMDGPU global_load-family work.
  bool global_load;
  // Node contributes AMDGPU global_store-family work.
  bool global_store;
  // Node contributes AMDGPU buffer_load-family work.
  bool buffer_load;
  // Node contributes AMDGPU buffer_store-family work.
  bool buffer_store;
  // Node contributes AMDGPU flat-memory-family work.
  bool flat_memory;
  // Node contributes local or workgroup-memory work.
  bool local_memory;
  // Node contributes scalar-memory work.
  bool scalar_memory;
  // Node contributes memory work without a more specific memory family.
  bool generic_memory;
  // Node contributes atomic-memory work.
  bool atomic;
  // Node contributes branch work.
  bool branch;
  // Node contributes barrier or synchronization work.
  bool barrier;
  // Node contributes control-flow work.
  bool control;
  // Node contributes numeric conversion work.
  bool conversion;
  // Node contributes cache or prefetch work.
  bool cache;
  // Node contributes register-move or repair work.
  bool register_move;
} loom_target_compile_report_low_node_features_t;

static loom_target_compile_report_low_node_features_t
loom_target_compile_report_classify_low_node_features(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  loom_target_compile_report_low_node_features_t features = {0};
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
      node->descriptor == NULL) {
    return features;
  }
  const loom_low_descriptor_t* descriptor = node->descriptor;
  const loom_low_schedule_class_t* schedule_class =
      loom_target_compile_report_descriptor_schedule_class(descriptor_set,
                                                           descriptor);
  const iree_string_view_t schedule_class_name =
      loom_target_compile_report_schedule_class_name(descriptor_set, node);
  const iree_string_view_t semantic_tag =
      loom_target_compile_report_descriptor_semantic_tag(descriptor_set,
                                                         descriptor);
  const iree_string_view_t descriptor_key =
      loom_target_compile_report_descriptor_key(descriptor_set, descriptor);

  features.scalar_alu =
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.salu")) ||
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_SCALAR_ALU);
  features.vector_alu =
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.valu")) ||
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_VECTOR_ALU);
  features.matrix =
      iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.")) ||
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.mfma")) ||
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.wmma")) ||
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_MATRIX);
  features.mfma =
      iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.mfma.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.smfmac.")) ||
      iree_string_view_starts_with(schedule_class_name, IREE_SV("amdgpu.mfma"));
  features.wmma =
      iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.wmma.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("matrix.swmmac.")) ||
      iree_string_view_starts_with(schedule_class_name, IREE_SV("amdgpu.wmma"));
  features.dot = iree_string_view_starts_with(semantic_tag, IREE_SV("dot."));
  features.cache =
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.cache."));
  features.global_memory =
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.vmem")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.global."));
  features.global_load = iree_string_view_starts_with(
      descriptor_key, IREE_SV("amdgpu.global_load_"));
  features.global_store = iree_string_view_starts_with(
      descriptor_key, IREE_SV("amdgpu.global_store_"));
  features.buffer_load = iree_string_view_starts_with(
      descriptor_key, IREE_SV("amdgpu.buffer_load_"));
  features.buffer_store = iree_string_view_starts_with(
      descriptor_key, IREE_SV("amdgpu.buffer_store_"));
  features.flat_memory = iree_string_view_starts_with(
                             descriptor_key, IREE_SV("amdgpu.flat_load_")) ||
                         iree_string_view_starts_with(
                             descriptor_key, IREE_SV("amdgpu.flat_store_"));
  features.local_memory =
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.lds")) ||
      iree_string_view_equal(schedule_class_name,
                             IREE_SV("amdgpu.vmem.load.lds")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.workgroup."));
  features.scalar_memory =
      iree_string_view_starts_with(schedule_class_name, IREE_SV("amdgpu.smem"));
  features.generic_memory =
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.generic.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.load.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.store.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("memory.hal."));
  const bool memory_resource =
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_LOAD) ||
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_STORE);
  if (memory_resource && !features.global_memory && !features.local_memory &&
      !features.scalar_memory && !features.generic_memory) {
    features.generic_memory = true;
  }
  features.atomic = loom_target_compile_report_string_contains(
      semantic_tag, IREE_SV(".atomic."));
  features.branch =
      iree_string_view_starts_with(semantic_tag, IREE_SV("control.branch")) ||
      iree_string_view_starts_with(semantic_tag,
                                   IREE_SV("control.cond_branch")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("control.return")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("control.call"));
  features.barrier =
      iree_string_view_starts_with(semantic_tag, IREE_SV("control.barrier")) ||
      iree_string_view_starts_with(schedule_class_name,
                                   IREE_SV("amdgpu.barrier"));
  features.control =
      iree_string_view_starts_with(semantic_tag, IREE_SV("control.")) ||
      (schedule_class != NULL &&
       iree_all_bits_set(schedule_class->flags,
                         LOOM_LOW_SCHEDULE_CLASS_FLAG_CONTROL)) ||
      loom_target_compile_report_schedule_class_uses_resource_kind(
          descriptor_set, schedule_class, LOOM_LOW_RESOURCE_KIND_CONTROL);
  features.conversion =
      iree_string_view_starts_with(semantic_tag, IREE_SV("convert."));
  features.register_move =
      iree_string_view_starts_with(semantic_tag, IREE_SV("register.copy.")) ||
      iree_string_view_starts_with(semantic_tag, IREE_SV("integer.move."));
  return features;
}

static bool loom_target_compile_report_low_node_features_are_known(
    const loom_target_compile_report_low_node_features_t* features) {
  return features->scalar_alu || features->vector_alu || features->matrix ||
         features->mfma || features->wmma || features->dot ||
         features->global_memory || features->global_load ||
         features->global_store || features->buffer_load ||
         features->buffer_store || features->flat_memory ||
         features->local_memory || features->scalar_memory ||
         features->generic_memory || features->atomic || features->branch ||
         features->barrier || features->control || features->conversion ||
         features->cache || features->register_move;
}

static void loom_target_compile_report_accumulate_low_node_static_mix(
    const loom_low_schedule_table_t* schedule,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_static_instruction_mix_t* mix) {
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_TERMINATOR) {
    const uint64_t control_transfer_count =
        loom_target_compile_report_low_structural_control_transfer_count(
            schedule, node);
    mix->branch_count += control_transfer_count;
    mix->control_count += control_transfer_count;
    return;
  }
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
      node->descriptor == NULL) {
    return;
  }
  ++mix->descriptor_count;
  const loom_target_compile_report_low_node_features_t features =
      loom_target_compile_report_classify_low_node_features(descriptor_set,
                                                            node);
  mix->scalar_alu_count += features.scalar_alu ? 1 : 0;
  mix->vector_alu_count += features.vector_alu ? 1 : 0;
  mix->matrix_count += features.matrix ? 1 : 0;
  mix->mfma_count += features.mfma ? 1 : 0;
  mix->wmma_count += features.wmma ? 1 : 0;
  mix->dot_count += features.dot ? 1 : 0;
  mix->global_memory_count += features.global_memory ? 1 : 0;
  mix->global_load_count += features.global_load ? 1 : 0;
  mix->global_store_count += features.global_store ? 1 : 0;
  mix->buffer_load_count += features.buffer_load ? 1 : 0;
  mix->buffer_store_count += features.buffer_store ? 1 : 0;
  mix->flat_memory_count += features.flat_memory ? 1 : 0;
  mix->local_memory_count += features.local_memory ? 1 : 0;
  mix->scalar_memory_count += features.scalar_memory ? 1 : 0;
  mix->generic_memory_count += features.generic_memory ? 1 : 0;
  mix->atomic_count += features.atomic ? 1 : 0;
  mix->branch_count += features.branch ? 1 : 0;
  mix->barrier_count += features.barrier ? 1 : 0;
  mix->control_count += features.control ? 1 : 0;
  mix->conversion_count += features.conversion ? 1 : 0;
  mix->cache_count += features.cache ? 1 : 0;
  mix->register_move_count += features.register_move ? 1 : 0;
  mix->unknown_count +=
      loom_target_compile_report_low_node_features_are_known(&features) ? 0 : 1;
}

static void loom_target_compile_report_accumulate_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source) {
  target->descriptor_count += source->descriptor_count;
  target->unknown_count += source->unknown_count;
  target->scalar_alu_count += source->scalar_alu_count;
  target->vector_alu_count += source->vector_alu_count;
  target->matrix_count += source->matrix_count;
  target->mfma_count += source->mfma_count;
  target->wmma_count += source->wmma_count;
  target->dot_count += source->dot_count;
  target->global_memory_count += source->global_memory_count;
  target->global_load_count += source->global_load_count;
  target->global_store_count += source->global_store_count;
  target->buffer_load_count += source->buffer_load_count;
  target->buffer_store_count += source->buffer_store_count;
  target->flat_memory_count += source->flat_memory_count;
  target->local_memory_count += source->local_memory_count;
  target->scalar_memory_count += source->scalar_memory_count;
  target->generic_memory_count += source->generic_memory_count;
  target->atomic_count += source->atomic_count;
  target->branch_count += source->branch_count;
  target->barrier_count += source->barrier_count;
  target->control_count += source->control_count;
  target->conversion_count += source->conversion_count;
  target->cache_count += source->cache_count;
  target->register_move_count += source->register_move_count;
}

static void loom_target_compile_report_record_low_static_instruction_mix(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  loom_target_compile_report_static_instruction_mix_t mix = {0};
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    loom_target_compile_report_accumulate_low_node_static_mix(
        &frame->schedule, descriptor_set, node, &mix);
  }
  loom_target_compile_report_record_static_instruction_mix(report, &mix);
}

static iree_string_view_t
loom_target_compile_report_low_frame_emitted_function_name(
    const loom_low_emission_frame_t* frame) {
  const iree_string_view_t export_symbol =
      frame->target.bundle_storage.export_plan.export_symbol;
  if (!iree_string_view_is_empty(export_symbol)) {
    return export_symbol;
  }
  return loom_low_diagnostic_function_name(frame->module, frame->function_op);
}

static void loom_target_compile_report_record_low_frame_identity(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  report->function_name =
      loom_target_compile_report_low_frame_emitted_function_name(frame);
  report->lowered_symbol =
      loom_low_diagnostic_function_name(frame->module, frame->function_op);
  report->target_bundle_name = frame->target.bundle_storage.bundle.name;
  report->target_snapshot_name = frame->target.bundle_storage.snapshot.name;
  report->target_export_name = frame->target.bundle_storage.export_plan.name;
  report->target_export_symbol =
      frame->target.bundle_storage.export_plan.export_symbol;
  report->target_config_name = frame->target.bundle_storage.config.name;
}

static iree_string_view_t loom_target_compile_report_module_string(
    const loom_module_t* module, loom_string_id_t string_id,
    iree_string_view_t fallback) {
  if (module == NULL || string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return fallback;
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_target_compile_report_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return IREE_SV("<unknown>");
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_target_compile_report_module_string(module, value->name_id,
                                                  IREE_SV("<unnamed>"));
}

static iree_string_view_t loom_target_compile_report_block_name(
    const loom_module_t* module, const loom_block_t* block) {
  if (block == NULL) {
    return IREE_SV("<anonymous>");
  }
  return loom_target_compile_report_module_string(module, block->label_id,
                                                  IREE_SV("<anonymous>"));
}

static iree_string_view_t loom_target_compile_report_value_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_liveness_value_class_t value_class) {
  return loom_low_diagnostic_value_class_name(descriptor_set, value_class);
}

static iree_string_view_t loom_target_compile_report_op_name(
    const loom_module_t* module, const loom_op_t* op) {
  if (module == NULL || op == NULL) {
    return IREE_SV("<unknown>");
  }
  return loom_op_name(module, op);
}

typedef struct loom_target_compile_report_pressure_origin_info_t {
  // Structured pressure origin kind.
  loom_target_compile_report_pressure_origin_kind_t kind;
  // Defining operation mnemonic when available.
  iree_string_view_t operation_name;
  // Descriptor semantic tag for descriptor-backed origins.
  iree_string_view_t semantic_tag;
} loom_target_compile_report_pressure_origin_info_t;

static loom_target_compile_report_pressure_origin_kind_t
loom_target_compile_report_pressure_origin_from_low_node_features(
    const loom_target_compile_report_low_node_features_t* features) {
  if (features->dot) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_DOT;
  }
  if (features->matrix) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_MATRIX;
  }
  if (features->local_memory) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_LOCAL_MEMORY;
  }
  if (features->global_memory) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GLOBAL_MEMORY;
  }
  if (features->scalar_memory) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_MEMORY;
  }
  if (features->generic_memory) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_GENERIC_MEMORY;
  }
  if (features->register_move) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_REGISTER_MOVE;
  }
  if (features->conversion) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONVERSION;
  }
  if (features->barrier) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BARRIER;
  }
  if (features->control || features->branch) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL;
  }
  if (features->cache) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CACHE;
  }
  if (features->scalar_alu) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SCALAR_ALU;
  }
  if (features->vector_alu) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_VECTOR_ALU;
  }
  return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN;
}

static loom_target_compile_report_pressure_origin_kind_t
loom_target_compile_report_pressure_origin_from_low_op(const loom_op_t* op) {
  if (op == NULL) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN;
  }
  if (loom_low_const_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONSTANT;
  }
  if (loom_low_copy_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_COPY;
  }
  if (loom_low_slice_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SLICE;
  }
  if (loom_low_concat_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONCAT;
  }
  if (loom_low_storage_reserve_isa(op) || loom_low_storage_view_isa(op) ||
      loom_low_storage_address_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_STORAGE;
  }
  if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_SPILL_RELOAD;
  }
  if (loom_low_br_isa(op) || loom_low_cond_br_isa(op) ||
      loom_low_return_isa(op)) {
    return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_CONTROL;
  }
  return LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION;
}

static loom_target_compile_report_pressure_origin_info_t
loom_target_compile_report_pressure_origin_from_node(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_schedule_node_t* node) {
  loom_target_compile_report_pressure_origin_info_t info = {
      .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN,
      .operation_name = loom_target_compile_report_op_name(module, node->op),
  };
  if (node->kind == LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR &&
      node->descriptor != NULL) {
    const loom_target_compile_report_low_node_features_t features =
        loom_target_compile_report_classify_low_node_features(descriptor_set,
                                                              node);
    info.kind =
        loom_target_compile_report_pressure_origin_from_low_node_features(
            &features);
    if (info.kind == LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
      info.kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_OPERATION;
    }
    info.semantic_tag = loom_target_compile_report_descriptor_semantic_tag(
        descriptor_set, node->descriptor);
    return info;
  }
  info.kind = loom_target_compile_report_pressure_origin_from_low_op(node->op);
  return info;
}

static loom_target_compile_report_pressure_origin_info_t
loom_target_compile_report_pressure_origin_from_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return (loom_target_compile_report_pressure_origin_info_t){
        .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN,
    };
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return (loom_target_compile_report_pressure_origin_info_t){
        .kind = LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_BLOCK_ARGUMENT,
        .operation_name = IREE_SV("<block-argument>"),
    };
  }
  const loom_op_t* op = loom_value_def_op(value);
  return (loom_target_compile_report_pressure_origin_info_t){
      .kind = loom_target_compile_report_pressure_origin_from_low_op(op),
      .operation_name = loom_target_compile_report_op_name(module, op),
  };
}

static iree_status_t loom_target_compile_report_build_pressure_origin_infos(
    const loom_low_schedule_table_t* schedule,
    loom_target_compile_report_pressure_origin_info_t* origin_infos,
    iree_host_size_t origin_info_count) {
  if (schedule == NULL || origin_infos == NULL) {
    return iree_ok_status();
  }
  const loom_module_t* module = schedule->module;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  for (iree_host_size_t i = 0; i < schedule->node_count; ++i) {
    const loom_low_schedule_node_t* node = &schedule->nodes[i];
    if (node->result_count == 0) {
      continue;
    }
    const loom_target_compile_report_pressure_origin_info_t info =
        loom_target_compile_report_pressure_origin_from_node(
            module, descriptor_set, node);
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(node);
    for (uint16_t j = 0; j < node->result_count; ++j) {
      const loom_value_ordinal_t result_ordinal = result_ordinals[j];
      if (result_ordinal >= origin_info_count) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "low schedule result ordinal exceeds liveness value domain");
      }
      origin_infos[result_ordinal] = info;
    }
  }
  return iree_ok_status();
}

static bool loom_target_compile_report_schedule_band_matches(
    const loom_target_compile_report_schedule_band_row_t* row,
    const loom_target_compile_report_pressure_origin_info_t* info) {
  if (row->origin_kind != info->kind) {
    return false;
  }
  if (!iree_string_view_is_empty(row->semantic_tag) ||
      !iree_string_view_is_empty(info->semantic_tag)) {
    return iree_string_view_equal(row->semantic_tag, info->semantic_tag);
  }
  return iree_string_view_equal(row->origin_operation_name,
                                info->operation_name);
}

static void loom_target_compile_report_add_schedule_band_node_results(
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_schedule_band_row_t* row) {
  if (liveness == NULL || liveness->value_ids == NULL) {
    return;
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[i];
    if (result_ordinal == LOOM_VALUE_ORDINAL_INVALID ||
        result_ordinal >= liveness->value_count) {
      continue;
    }
    const loom_value_id_t value_id = liveness->value_ids[result_ordinal];
    if (iree_string_view_is_empty(row->sample_value_name)) {
      row->sample_value_name =
          loom_target_compile_report_value_name(module, value_id);
    }
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, result_ordinal);
    if (interval != NULL) {
      row->result_unit_count += interval->unit_count;
    }
    ++row->result_value_count;
  }
}

static void loom_target_compile_report_accumulate_schedule_band_node(
    const loom_low_schedule_table_t* schedule,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node,
    loom_target_compile_report_schedule_band_row_t* row) {
  ++row->node_count;
  loom_target_compile_report_accumulate_low_node_static_mix(
      schedule, schedule->target.descriptor_set, node,
      &row->static_instruction_mix);
  loom_target_compile_report_add_schedule_band_node_results(
      schedule->module, liveness, node, row);
}

static bool loom_target_compile_report_schedule_band_summary_matches(
    const loom_target_compile_report_schedule_band_summary_row_t* summary,
    const loom_target_compile_report_schedule_band_row_t* band) {
  return summary->block_index == band->block_index &&
         iree_string_view_equal(summary->block_name, band->block_name) &&
         summary->origin_kind == band->origin_kind &&
         iree_string_view_equal(summary->origin_operation_name,
                                band->origin_operation_name) &&
         iree_string_view_equal(summary->semantic_tag, band->semantic_tag);
}

static void loom_target_compile_report_accumulate_schedule_band_summary(
    loom_target_compile_report_schedule_band_summary_row_t* summary,
    const loom_target_compile_report_schedule_band_row_t* band) {
  ++summary->band_count;
  summary->node_count += band->node_count;
  summary->max_band_node_count =
      iree_max(summary->max_band_node_count, band->node_count);
  if (iree_string_view_is_empty(summary->sample_value_name)) {
    summary->sample_value_name = band->sample_value_name;
  }
  loom_target_compile_report_accumulate_static_mix(
      &summary->static_instruction_mix, &band->static_instruction_mix);
  summary->result_value_count += band->result_value_count;
  summary->result_unit_count += band->result_unit_count;
}

static iree_status_t loom_target_compile_report_add_schedule_band_summary(
    loom_target_compile_report_schedule_band_summary_row_t* summaries,
    iree_host_size_t summary_capacity, iree_host_size_t* summary_count,
    const loom_target_compile_report_schedule_band_row_t* band) {
  for (iree_host_size_t i = 0; i < *summary_count; ++i) {
    if (loom_target_compile_report_schedule_band_summary_matches(&summaries[i],
                                                                 band)) {
      loom_target_compile_report_accumulate_schedule_band_summary(&summaries[i],
                                                                  band);
      return iree_ok_status();
    }
  }
  if (*summary_count >= summary_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low schedule band summary capacity exceeded");
  }
  loom_target_compile_report_schedule_band_summary_row_t* summary =
      &summaries[(*summary_count)++];
  *summary = (loom_target_compile_report_schedule_band_summary_row_t){
      .function_name = band->function_name,
      .block_name = band->block_name,
      .block_index = band->block_index,
      .first_packet_index = band->first_packet_index,
      .origin_kind = band->origin_kind,
      .origin_operation_name = band->origin_operation_name,
      .semantic_tag = band->semantic_tag,
      .sample_value_name = band->sample_value_name,
  };
  loom_target_compile_report_accumulate_schedule_band_summary(summary, band);
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_schedule_band(
    loom_target_compile_report_t* report,
    loom_target_compile_report_schedule_band_summary_row_t* summaries,
    iree_host_size_t summary_capacity, iree_host_size_t* summary_count,
    const loom_target_compile_report_schedule_band_row_t* band) {
  IREE_RETURN_IF_ERROR(
      loom_target_compile_report_record_schedule_band_row(report, band));
  return loom_target_compile_report_add_schedule_band_summary(
      summaries, summary_capacity, summary_count, band);
}

static iree_status_t loom_target_compile_report_record_schedule_band_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_table_t* schedule) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS)) {
    return iree_ok_status();
  }
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SCHEDULE_BAND_ROWS;
  if (schedule == NULL || schedule->scheduled_node_count == 0 ||
      schedule->block_count == 0 || iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  if (schedule->scheduled_node_indices == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule band rows require scheduled nodes");
  }
  const loom_module_t* module = schedule->module;
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  iree_host_size_t summary_bytes = 0;
  if (!iree_host_size_checked_mul(
          schedule->scheduled_node_count,
          sizeof(loom_target_compile_report_schedule_band_summary_row_t),
          &summary_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low schedule band summary scratch is too large");
  }
  loom_target_compile_report_schedule_band_summary_row_t* summaries = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(report->allocator, summary_bytes,
                                             (void**)&summaries));
  memset(summaries, 0, summary_bytes);
  iree_status_t status = iree_ok_status();
  iree_host_size_t summary_count = 0;
  for (iree_host_size_t block_index = 0;
       iree_status_is_ok(status) && block_index < schedule->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    bool has_band = false;
    loom_target_compile_report_schedule_band_row_t band = {0};
    for (uint32_t i = 0;
         iree_status_is_ok(status) && i < block->scheduled_node_count; ++i) {
      const iree_host_size_t packet_index =
          (iree_host_size_t)block->scheduled_node_start + (iree_host_size_t)i;
      if (packet_index >= schedule->scheduled_node_count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low schedule band packet index is invalid");
        break;
      }
      const uint32_t node_index =
          schedule->scheduled_node_indices[packet_index];
      if (node_index >= schedule->node_count) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low schedule band node index is invalid");
        break;
      }
      const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
      const loom_target_compile_report_pressure_origin_info_t info =
          loom_target_compile_report_pressure_origin_from_node(
              module, descriptor_set, node);
      if (!has_band ||
          !loom_target_compile_report_schedule_band_matches(&band, &info)) {
        if (has_band) {
          status = loom_target_compile_report_record_schedule_band(
              report, summaries, schedule->scheduled_node_count, &summary_count,
              &band);
          if (!iree_status_is_ok(status)) {
            break;
          }
        }
        band = (loom_target_compile_report_schedule_band_row_t){
            .function_name = report->function_name,
            .block_name =
                loom_target_compile_report_block_name(module, node->block),
            .block_index = (uint32_t)block_index,
            .first_packet_index = (uint64_t)packet_index,
            .first_scheduled_ordinal = node->scheduled_ordinal,
            .origin_kind = info.kind,
            .origin_operation_name = info.operation_name,
            .semantic_tag = info.semantic_tag,
        };
        has_band = true;
      }
      loom_target_compile_report_accumulate_schedule_band_node(
          schedule, liveness, node, &band);
    }
    if (iree_status_is_ok(status) && has_band) {
      status = loom_target_compile_report_record_schedule_band(
          report, summaries, schedule->scheduled_node_count, &summary_count,
          &band);
    }
  }
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < summary_count;
       ++i) {
    status = loom_target_compile_report_record_schedule_band_summary_row(
        report, &summaries[i]);
  }
  iree_allocator_free(report->allocator, summaries);
  return status;
}

static bool loom_target_compile_report_interval_is_live_at_point(
    const loom_liveness_interval_t* interval, uint32_t point) {
  return interval->start_point <= point && point < interval->end_point;
}

static bool loom_target_compile_report_pressure_origin_row_matches(
    const loom_target_compile_report_pressure_origin_row_t* row,
    const loom_target_compile_report_pressure_origin_row_t* candidate) {
  return row->origin_kind == candidate->origin_kind &&
         iree_string_view_equal(row->origin_operation_name,
                                candidate->origin_operation_name) &&
         iree_string_view_equal(row->semantic_tag, candidate->semantic_tag);
}

static iree_status_t loom_target_compile_report_record_pressure_origin_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_table_t* schedule,
    const loom_low_descriptor_set_t* descriptor_set) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS)) {
    return iree_ok_status();
  }
  report->detail_flags |=
      LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ORIGIN_ROWS;
  if (liveness->value_count == 0 || liveness->pressure_summary_count == 0 ||
      iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t origin_info_bytes = 0;
  if (!iree_host_size_checked_mul(
          liveness->value_count,
          sizeof(loom_target_compile_report_pressure_origin_info_t),
          &origin_info_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pressure origin info table is too large");
  }
  iree_host_size_t row_bytes = 0;
  if (!iree_host_size_checked_mul(
          liveness->value_count,
          sizeof(loom_target_compile_report_pressure_origin_row_t),
          &row_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "pressure origin row scratch is too large");
  }
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  loom_target_compile_report_pressure_origin_row_t* rows = NULL;
  iree_status_t status = iree_allocator_malloc(
      report->allocator, origin_info_bytes, (void**)&origin_infos);
  if (iree_status_is_ok(status)) {
    memset(origin_infos, 0, origin_info_bytes);
    status = loom_target_compile_report_build_pressure_origin_infos(
        schedule, origin_infos, liveness->value_count);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(report->allocator, row_bytes, (void**)&rows);
  }
  for (iree_host_size_t summary_index = 0;
       iree_status_is_ok(status) &&
       summary_index < liveness->pressure_summary_count;
       ++summary_index) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[summary_index];
    iree_host_size_t row_count = 0;
    for (iree_host_size_t value_ordinal = 0;
         value_ordinal < liveness->value_count; ++value_ordinal) {
      const loom_liveness_interval_t* interval =
          loom_liveness_interval_for_value_ordinal(
              liveness, (loom_value_ordinal_t)value_ordinal);
      if (interval == NULL ||
          !loom_liveness_value_class_equal(interval->value_class,
                                           summary->value_class) ||
          !loom_target_compile_report_interval_is_live_at_point(
              interval, summary->peak_point)) {
        continue;
      }
      loom_target_compile_report_pressure_origin_info_t origin_info =
          origin_infos[value_ordinal];
      if (origin_info.kind ==
          LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
        origin_info = loom_target_compile_report_pressure_origin_from_value(
            liveness->module, interval->value_id);
      }
      loom_target_compile_report_pressure_origin_row_t candidate = {
          .function_name = report->function_name,
          .register_class = loom_target_compile_report_value_class_name(
              descriptor_set, summary->value_class),
          .type_kind = summary->value_class.type_kind,
          .element_type = summary->value_class.element_type,
          .peak_point = summary->peak_point,
          .peak_block_name = loom_target_compile_report_block_name(
              liveness->module, summary->peak_block),
          .peak_operation_name = summary->peak_op
                                     ? loom_target_compile_report_op_name(
                                           liveness->module, summary->peak_op)
                                     : IREE_SV("<block-boundary>"),
          .origin_kind = origin_info.kind,
          .origin_operation_name = origin_info.operation_name,
          .semantic_tag = origin_info.semantic_tag,
          .sample_value_name = loom_target_compile_report_value_name(
              liveness->module, interval->value_id),
          .live_units = interval->unit_count,
          .live_values = 1,
      };
      loom_target_compile_report_pressure_origin_row_t* row = NULL;
      for (iree_host_size_t row_index = 0; row_index < row_count; ++row_index) {
        if (loom_target_compile_report_pressure_origin_row_matches(
                &rows[row_index], &candidate)) {
          row = &rows[row_index];
          break;
        }
      }
      if (row == NULL) {
        row = &rows[row_count++];
        *row = candidate;
      } else {
        row->live_units += candidate.live_units;
        row->live_values += candidate.live_values;
      }
    }
    for (iree_host_size_t row_index = 0;
         iree_status_is_ok(status) && row_index < row_count; ++row_index) {
      status = loom_target_compile_report_record_pressure_origin_row(
          report, &rows[row_index]);
    }
  }
  if (rows != NULL) {
    iree_allocator_free(report->allocator, rows);
  }
  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  return status;
}

static void loom_target_compile_report_record_move_cause_if_nonzero(
    loom_target_compile_report_t* report,
    loom_target_compile_report_move_cause_t cause, uint64_t packet_count,
    uint64_t unit_count) {
  if (packet_count == 0 && unit_count == 0) {
    return;
  }
  loom_target_compile_report_record_move_cause(report, cause, packet_count,
                                               unit_count);
}

static uint64_t loom_target_compile_report_value_register_unit_count(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (module == NULL || value_id >= module->values.count) {
    return 0;
  }
  const loom_type_t type = loom_module_value_type(module, value_id);
  return loom_low_type_is_register(type)
             ? loom_low_register_type_unit_count(type)
             : 0;
}

static uint64_t loom_target_compile_report_result_register_unit_count(
    const loom_module_t* module, const loom_liveness_analysis_t* liveness,
    const loom_low_schedule_node_t* node) {
  uint64_t unit_count = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[i];
    IREE_ASSERT(result_ordinal < liveness->value_count,
                "verified schedule node result ordinal must fit liveness "
                "value domain");
    unit_count += loom_target_compile_report_value_register_unit_count(
        module, liveness->value_ids[result_ordinal]);
  }
  return unit_count;
}

static const loom_low_allocation_assignment_t*
loom_target_compile_report_map_assignment(
    const loom_low_allocation_table_t* allocation, loom_value_id_t value_id) {
  return loom_low_allocation_map_active_value_assignment(allocation, value_id,
                                                         NULL);
}

static uint64_t loom_target_compile_report_slice_move_unit_count(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op) {
  uint64_t unit_count = 0;
  const int64_t offset = loom_low_slice_offset(op);
  IREE_ASSERT(offset >= 0 && offset <= UINT32_MAX,
              "verified low.slice offset must fit in uint32_t");
  const loom_low_allocation_assignment_t* source_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_slice_source(op));
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_slice_result(op));
  const uint32_t source_offset = (uint32_t)offset;
  IREE_ASSERT(source_offset <= source_assignment->location_count &&
                  result_assignment->location_count <=
                      source_assignment->location_count - source_offset,
              "verified low.slice range must fit source assignment");
  IREE_ASSERT(loom_low_allocation_storage_assignment_classes_share(
                  allocation->target.descriptor_set, source_assignment,
                  result_assignment),
              "allocated low.slice values must share one target storage class");
  for (uint32_t unit_index = 0; unit_index < result_assignment->location_count;
       ++unit_index) {
    if (!loom_low_allocation_storage_assignment_subranges_equal(
            allocation->target.descriptor_set, result_assignment, unit_index,
            source_assignment, source_offset + unit_index, /*unit_count=*/1)) {
      ++unit_count;
    }
  }
  return unit_count;
}

static uint64_t loom_target_compile_report_concat_move_unit_count(
    const loom_low_allocation_table_t* allocation, const loom_op_t* op) {
  if (!loom_low_allocation_move_topology_concat_requires_packet_materialization(
          allocation, op)) {
    return 0;
  }
  uint64_t unit_count = 0;
  const loom_low_allocation_assignment_t* result_assignment =
      loom_target_compile_report_map_assignment(allocation,
                                                loom_low_concat_result(op));
  uint32_t result_offset = 0;
  loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_low_allocation_assignment_t* source_assignment =
        loom_target_compile_report_map_assignment(allocation,
                                                  sources.values[i]);
    IREE_ASSERT(loom_low_allocation_storage_assignment_classes_share(
                    allocation->target.descriptor_set, result_assignment,
                    source_assignment),
                "allocated low.concat values must share one target storage "
                "class");
    IREE_ASSERT(result_offset <= result_assignment->location_count &&
                    source_assignment->location_count <=
                        result_assignment->location_count - result_offset,
                "verified low.concat range must fit result");
    for (uint32_t source_unit = 0;
         source_unit < source_assignment->location_count; ++source_unit) {
      if (!loom_low_allocation_storage_assignment_subranges_equal(
              allocation->target.descriptor_set, result_assignment,
              result_offset + source_unit, source_assignment, source_unit,
              /*unit_count=*/1)) {
        ++unit_count;
      }
    }
    result_offset += source_assignment->location_count;
  }
  IREE_ASSERT_EQ(result_offset, result_assignment->location_count);
  return unit_count;
}

static void loom_target_compile_report_record_low_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->copy_decision_count; ++i) {
    const loom_low_allocation_copy_decision_t* decision =
        &allocation->copy_decisions[i];
    if (decision->kind != LOOM_LOW_ALLOCATION_COPY_MATERIALIZED) {
      continue;
    }
    ++packet_count;
    IREE_ASSERT(
        decision->result_assignment_index < allocation->assignment_count,
        "verified copy decision result assignment must fit allocation "
        "table");
    unit_count += allocation->assignments[decision->result_assignment_index]
                      .location_count;
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_COPY, packet_count,
      unit_count);
}

static void loom_target_compile_report_record_edge_copy_moves(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  for (iree_host_size_t i = 0; i < allocation->edge_copy_group_count; ++i) {
    const loom_low_allocation_edge_copy_group_t* group =
        &allocation->edge_copy_groups[i];
    IREE_ASSERT(group->copy_start <= allocation->edge_copy_count &&
                    group->copy_count <=
                        allocation->edge_copy_count - group->copy_start,
                "verified edge-copy group range must fit allocation table");
    uint64_t group_unit_count = 0;
    for (uint32_t j = 0; j < group->copy_count; ++j) {
      const loom_low_allocation_edge_copy_t* edge_copy =
          &allocation->edge_copies[group->copy_start + j];
      IREE_ASSERT(
          edge_copy->source_assignment_index < allocation->assignment_count &&
              edge_copy->destination_assignment_index <
                  allocation->assignment_count,
          "verified edge-copy assignments must fit allocation table");
      const loom_low_allocation_assignment_t* source_assignment =
          &allocation->assignments[edge_copy->source_assignment_index];
      const loom_low_allocation_assignment_t* destination_assignment =
          &allocation->assignments[edge_copy->destination_assignment_index];
      for (uint32_t unit_index = 0; unit_index < edge_copy->unit_count;
           ++unit_index) {
        if (!loom_low_allocation_storage_assignment_subranges_equal(
                allocation->target.descriptor_set, destination_assignment,
                edge_copy->destination_unit_offset + unit_index,
                source_assignment, edge_copy->source_unit_offset + unit_index,
                /*unit_count=*/1)) {
          ++group_unit_count;
        }
      }
    }
    if (group_unit_count != 0) {
      ++packet_count;
      unit_count += group_unit_count;
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_BRANCH_EDGE, packet_count,
      unit_count);
}

static void
loom_target_compile_report_record_operand_bank_materialization_moves(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  uint64_t packet_count = 0;
  uint64_t unit_count = 0;
  const loom_module_t* module = frame->schedule.module;
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  const loom_low_descriptor_set_t* descriptor_set =
      frame->schedule.target.descriptor_set;
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_low_schedule_node_t* node = &frame->schedule.nodes[i];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
        !loom_target_compile_report_descriptor_semantic_tag_is(
            descriptor_set, node->descriptor, IREE_SV("register.copy.b32"))) {
      continue;
    }
    ++packet_count;
    unit_count += loom_target_compile_report_result_register_unit_count(
        module, liveness, node);
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report,
      LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_OPERAND_BANK_MATERIALIZATION,
      packet_count, unit_count);
}

static void loom_target_compile_report_record_structural_packet_moves(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  uint64_t constant_packet_count = 0;
  uint64_t constant_unit_count = 0;
  uint64_t slice_packet_count = 0;
  uint64_t slice_unit_count = 0;
  uint64_t concat_packet_count = 0;
  uint64_t concat_unit_count = 0;
  const loom_module_t* module = frame->schedule.module;
  const loom_low_allocation_table_t* allocation = &frame->allocation;
  for (iree_host_size_t i = 0; i < frame->schedule.node_count; ++i) {
    const loom_op_t* op = frame->schedule.nodes[i].op;
    if (op == NULL) {
      continue;
    }
    if (loom_low_const_isa(op)) {
      ++constant_packet_count;
      constant_unit_count +=
          loom_target_compile_report_value_register_unit_count(
              module, loom_low_const_result(op));
    } else if (loom_low_slice_isa(op)) {
      const uint64_t move_count =
          loom_target_compile_report_slice_move_unit_count(allocation, op);
      if (move_count != 0) {
        ++slice_packet_count;
        slice_unit_count += move_count;
      }
    } else if (loom_low_concat_isa(op)) {
      const uint64_t move_count =
          loom_target_compile_report_concat_move_unit_count(allocation, op);
      if (move_count != 0) {
        ++concat_packet_count;
        concat_unit_count += move_count;
      }
    }
  }
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_CONSTANT_MATERIALIZATION,
      constant_packet_count, constant_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_SLICE,
      slice_packet_count, slice_unit_count);
  loom_target_compile_report_record_move_cause_if_nonzero(
      report, LOOM_TARGET_COMPILE_REPORT_MOVE_CAUSE_LOW_CONCAT,
      concat_packet_count, concat_unit_count);
}

static void loom_target_compile_report_record_move_causes(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  loom_target_compile_report_record_low_copy_moves(report, &frame->allocation);
  loom_target_compile_report_record_edge_copy_moves(report, &frame->allocation);
  loom_target_compile_report_record_operand_bank_materialization_moves(report,
                                                                       frame);
  loom_target_compile_report_record_structural_packet_moves(report, frame);
}

static loom_target_compile_report_source_low_selection_kind_t
loom_target_compile_report_source_low_selection_kind(
    loom_low_lower_report_selection_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_LOWER_REPORT_SELECTION_RULE:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_RULE;
    case LOOM_LOW_LOWER_REPORT_SELECTION_PLAN:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_PLAN;
    case LOOM_LOW_LOWER_REPORT_SELECTION_NONE:
    default:
      return LOOM_TARGET_COMPILE_REPORT_SOURCE_LOW_SELECTION_NONE;
  }
}

iree_status_t loom_target_compile_report_record_low_lowering(
    loom_target_compile_report_t* report,
    const loom_low_lower_result_t* lower_result) {
  report->detail_flags |= LOOM_TARGET_COMPILE_REPORT_DETAIL_SOURCE_LOW_ROWS;
  report->source_low_selected_op_count +=
      lower_result->selected_source_op_count;
  report->source_low_emitted_op_count += lower_result->emitted_low_op_count;
  for (const loom_low_lower_report_row_vec_t* vec =
           lower_result->report_rows.head;
       vec != NULL; vec = vec->next) {
    const loom_low_lower_report_row_t* source_rows =
        loom_low_lower_report_row_vec_const_rows(vec);
    for (iree_host_size_t i = 0; i < vec->count; ++i) {
      const loom_low_lower_report_row_t* source_row = &source_rows[i];
      const loom_target_compile_report_source_low_row_t row = {
          .function_name = source_row->function_name,
          .source_op_name = source_row->source_op_name,
          .source_op_kind = source_row->source_op_kind,
          .selection_kind =
              loom_target_compile_report_source_low_selection_kind(
                  source_row->selection_kind),
          .rule_set_index = source_row->rule_set_index,
          .rule_index = source_row->rule_index,
          .plan_id = source_row->plan_id,
          .plan_key = source_row->plan_key,
          .descriptor_id = source_row->descriptor_id,
          .emitted_low_op_count = source_row->emitted_low_op_count,
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_source_low_row(report, &row));
    }
  }
  for (iree_host_size_t i = 0; i < lower_result->memory_report_rows.count;
       ++i) {
    const loom_low_lower_memory_report_row_t* source_row =
        &lower_result->memory_report_rows.rows[i];
    const loom_target_compile_report_source_low_memory_row_t row = {
        .function_name = source_row->function_name,
        .source_op_name = source_row->source_op_name,
        .source_op_kind = source_row->source_op_kind,
        .memory_space = source_row->memory_space,
        .operation_kind = source_row->operation_kind,
        .packet_key = source_row->packet_key,
        .address_form = source_row->address_form,
        .dynamic_term_kind = source_row->dynamic_term_kind,
        .fallback_reason = source_row->fallback_reason,
        .static_offset_bytes = source_row->static_offset_bytes,
        .element_byte_count = source_row->element_byte_count,
        .vector_lane_count = source_row->vector_lane_count,
        .dynamic_stride_bytes = source_row->dynamic_stride_bytes,
        .vector_lane_stride_bytes = source_row->vector_lane_stride_bytes,
        .bank_stride_words = source_row->bank_stride_words,
        .bank_conflict_degree = source_row->bank_conflict_degree,
        .bank_conflict_kind = source_row->bank_conflict_kind,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_source_low_memory_row(report, &row));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_pressure_rows(
    loom_target_compile_report_t* report,
    const loom_liveness_analysis_t* liveness,
    const loom_low_descriptor_set_t* descriptor_set) {
  const bool wants_pressure_rows = loom_target_compile_report_wants_details(
      report, LOOM_TARGET_COMPILE_REPORT_DETAIL_PRESSURE_ROWS);
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const loom_liveness_pressure_summary_t* summary =
        &liveness->pressure_summaries[i];
    const iree_string_view_t register_class =
        loom_target_compile_report_value_class_name(descriptor_set,
                                                    summary->value_class);
    const loom_target_compile_report_pressure_summary_t pressure_summary = {
        .register_class = register_class,
        .peak_live_units = summary->peak_live_units,
    };
    if (wants_pressure_rows) {
      const loom_target_compile_report_pressure_row_t row = {
          .function_name = report->function_name,
          .register_class = register_class,
          .type_kind = summary->value_class.type_kind,
          .element_type = summary->value_class.element_type,
          .peak_live_units = summary->peak_live_units,
          .peak_live_values = summary->peak_live_values,
          .peak_point = summary->peak_point,
          .peak_block_name = loom_target_compile_report_block_name(
              liveness->module, summary->peak_block),
          .peak_operation_name = summary->peak_op
                                     ? loom_target_compile_report_op_name(
                                           liveness->module, summary->peak_op)
                                     : IREE_SV("<block-boundary>"),
      };
      IREE_RETURN_IF_ERROR(
          loom_target_compile_report_record_pressure_row(report, &row));
    } else {
      IREE_RETURN_IF_ERROR(loom_target_compile_report_record_pressure_summary(
          report, &pressure_summary));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_compile_report_record_spill_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  if (!loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_SPILL_ROWS)) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < allocation->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan =
        &allocation->spill_plans[i];
    const loom_low_allocation_assignment_t* assignment = NULL;
    if (spill_plan->assignment_index < allocation->assignment_count) {
      assignment = &allocation->assignments[spill_plan->assignment_index];
    }
    const loom_liveness_value_class_t value_class =
        assignment != NULL ? assignment->value_class
                           : (loom_liveness_value_class_t){0};
    const loom_target_compile_report_spill_row_t row = {
        .function_name = report->function_name,
        .value_name = loom_target_compile_report_value_name(
            allocation->module, spill_plan->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            allocation->target.descriptor_set, value_class),
        .type_kind = value_class.type_kind,
        .element_type = value_class.element_type,
        .assignment_index = spill_plan->assignment_index,
        .slot_index = spill_plan->slot_index,
        .slot_space = loom_low_spill_slot_space_name(spill_plan->slot_space),
        .byte_size = spill_plan->byte_size,
        .byte_alignment = spill_plan->byte_alignment,
        .store_count = spill_plan->store_count,
        .reload_count = spill_plan->reload_count,
    };
    IREE_RETURN_IF_ERROR(
        loom_target_compile_report_record_spill_row(report, &row));
  }
  return iree_ok_status();
}

typedef struct loom_target_compile_report_allocation_high_water_scratch_t {
  // Assignment index that reaches |high_water_units|, or UINT32_MAX.
  uint32_t assignment_index;
  // One-past-last physical register unit reached by the assignment.
  uint64_t high_water_units;
} loom_target_compile_report_allocation_high_water_scratch_t;

static bool loom_target_compile_report_liveness_value_ordinal(
    const loom_liveness_analysis_t* liveness, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  if (liveness == NULL || liveness->value_ids == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < liveness->value_count; ++i) {
    if (liveness->value_ids[i] == value_id) {
      *out_value_ordinal = (loom_value_ordinal_t)i;
      return true;
    }
  }
  return false;
}

typedef struct loom_target_compile_report_allocation_high_water_blockers_t {
  // Number of active assignment blockers below the high-water assignment.
  uint32_t active_assignment_count;
  // Total units owned by active assignment blockers.
  uint64_t active_assignment_units;
  // Number of active target storage-lease blockers below the high-water
  // assignment.
  uint32_t active_storage_lease_count;
  // Total units owned by active target storage-lease blockers.
  uint64_t active_storage_lease_units;
  // Number of pressure-releasable storage-lease blockers below the high-water
  // assignment.
  uint32_t active_pressure_storage_lease_count;
  // Total units owned by pressure-releasable storage-lease blockers.
  uint64_t active_pressure_storage_lease_units;
  // Number of fallback-release storage-lease blockers below the high-water
  // assignment.
  uint32_t active_fallback_storage_lease_count;
  // Total units owned by fallback-release storage-lease blockers.
  uint64_t active_fallback_storage_lease_units;
} loom_target_compile_report_allocation_high_water_blockers_t;

typedef struct loom_target_compile_report_allocation_lower_free_runs_t {
  // Number of unoccupied physical units below the high-water assignment base.
  uint64_t free_unit_count;
  // Number of contiguous unoccupied-unit runs below the assignment base.
  uint32_t free_run_count;
  // Largest contiguous unoccupied-unit run below the assignment base.
  uint32_t largest_free_run_unit_count;
} loom_target_compile_report_allocation_lower_free_runs_t;

typedef enum loom_target_compile_report_storage_lease_occupancy_e {
  // Treat active storage leases as occupied.
  LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_ACTIVE = 0,
  // Treat pressure-releasable storage leases as free.
  LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND =
      1,
} loom_target_compile_report_storage_lease_occupancy_t;

static bool loom_target_compile_report_assignment_active_at(
    const loom_low_allocation_assignment_t* assignment, uint32_t point) {
  return assignment->start_point <= point && point < assignment->end_point;
}

static bool loom_target_compile_report_unit_inside_range(
    uint32_t unit, uint32_t location_base, uint32_t location_count) {
  return unit >= location_base &&
         (uint64_t)unit < (uint64_t)location_base + location_count;
}

static bool loom_target_compile_report_storage_lease_is_pressure_releasable(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_storage_lease_t* lease) {
  if (allocation->storage_leases.records == NULL ||
      lease->lease_record_index >= allocation->storage_leases.record_count) {
    return false;
  }
  return iree_all_bits_set(
      allocation->storage_leases.records[lease->lease_record_index].flags,
      LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE);
}

static bool loom_target_compile_report_unit_blocked_by_assignment(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit) {
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (i == high_water_assignment_index) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        !loom_target_compile_report_assignment_active_at(assignment, point)) {
      continue;
    }
    if (loom_target_compile_report_unit_inside_range(
            unit, assignment->location_base, assignment->location_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_target_compile_report_unit_blocked_by_storage_lease(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  for (iree_host_size_t i = 0; i < allocation->storage_lease_instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[i];
    if (lease->assignment_index == high_water_assignment_index ||
        lease->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        lease->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        point < lease->start_point || point >= lease->end_point) {
      continue;
    }
    if (occupancy ==
            LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND &&
        loom_target_compile_report_storage_lease_is_pressure_releasable(
            allocation, lease)) {
      continue;
    }
    if (loom_target_compile_report_unit_inside_range(unit, lease->location_base,
                                                     lease->location_count)) {
      return true;
    }
  }
  return false;
}

static bool loom_target_compile_report_unit_blocked_in_lower_window(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint32_t point, uint32_t unit,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  return loom_target_compile_report_unit_blocked_by_assignment(
             allocation, high_water_assignment, high_water_assignment_index,
             point, unit) ||
         loom_target_compile_report_unit_blocked_by_storage_lease(
             allocation, high_water_assignment, high_water_assignment_index,
             point, unit, occupancy);
}

static loom_target_compile_report_allocation_lower_free_runs_t
loom_target_compile_report_allocation_lower_free_runs(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index,
    loom_target_compile_report_storage_lease_occupancy_t occupancy) {
  loom_target_compile_report_allocation_lower_free_runs_t free_runs = {0};
  const uint32_t point = high_water_assignment->start_point;
  uint32_t current_free_run = 0;
  for (uint32_t unit = 0; unit < high_water_assignment->location_base; ++unit) {
    if (loom_target_compile_report_unit_blocked_in_lower_window(
            allocation, high_water_assignment, high_water_assignment_index,
            point, unit, occupancy)) {
      current_free_run = 0;
      continue;
    }
    ++free_runs.free_unit_count;
    if (current_free_run == 0) {
      ++free_runs.free_run_count;
    }
    ++current_free_run;
    free_runs.largest_free_run_unit_count =
        iree_max(free_runs.largest_free_run_unit_count, current_free_run);
  }
  return free_runs;
}

static uint64_t loom_target_compile_report_units_below_high_water(
    uint32_t location_base, uint32_t location_count,
    uint64_t high_water_units) {
  if (location_count == 0 || (uint64_t)location_base >= high_water_units) {
    return 0;
  }
  const uint64_t location_end = (uint64_t)location_base + location_count;
  const uint64_t clipped_end = iree_min(location_end, high_water_units);
  return clipped_end - location_base;
}

static void loom_target_compile_report_add_high_water_assignment_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units,
    loom_target_compile_report_allocation_high_water_blockers_t* blockers) {
  const uint32_t point = high_water_assignment->start_point;
  for (iree_host_size_t i = 0; i < allocation->assignment_count; ++i) {
    if (i == high_water_assignment_index) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        !loom_target_compile_report_assignment_active_at(assignment, point)) {
      continue;
    }
    const uint64_t blocker_units =
        loom_target_compile_report_units_below_high_water(
            assignment->location_base, assignment->location_count,
            high_water_units);
    if (blocker_units == 0) {
      continue;
    }
    ++blockers->active_assignment_count;
    blockers->active_assignment_units += blocker_units;
  }
}

static void loom_target_compile_report_add_high_water_storage_lease_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units,
    loom_target_compile_report_allocation_high_water_blockers_t* blockers) {
  const uint32_t point = high_water_assignment->start_point;
  for (iree_host_size_t i = 0; i < allocation->storage_lease_instance_count;
       ++i) {
    const loom_low_allocation_storage_lease_t* lease =
        &allocation->storage_lease_instances[i];
    if (lease->assignment_index == high_water_assignment_index ||
        lease->descriptor_reg_class_id !=
            high_water_assignment->descriptor_reg_class_id ||
        lease->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        point < lease->start_point || point >= lease->end_point) {
      continue;
    }
    const uint64_t blocker_units =
        loom_target_compile_report_units_below_high_water(
            lease->location_base, lease->location_count, high_water_units);
    if (blocker_units == 0) {
      continue;
    }
    ++blockers->active_storage_lease_count;
    blockers->active_storage_lease_units += blocker_units;
    const bool pressure_releasable =
        lease->lease_record_index < allocation->storage_leases.record_count &&
        allocation->storage_leases.records != NULL &&
        iree_all_bits_set(
            allocation->storage_leases.records[lease->lease_record_index].flags,
            LOOM_LOW_STORAGE_LEASE_FLAG_RELEASE_FOR_PRESSURE);
    if (pressure_releasable) {
      ++blockers->active_pressure_storage_lease_count;
      blockers->active_pressure_storage_lease_units += blocker_units;
    } else {
      ++blockers->active_fallback_storage_lease_count;
      blockers->active_fallback_storage_lease_units += blocker_units;
    }
  }
}

static loom_target_compile_report_allocation_high_water_blockers_t
loom_target_compile_report_allocation_high_water_blockers(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_assignment_t* high_water_assignment,
    uint32_t high_water_assignment_index, uint64_t high_water_units) {
  loom_target_compile_report_allocation_high_water_blockers_t blockers = {0};
  loom_target_compile_report_add_high_water_assignment_blockers(
      allocation, high_water_assignment, high_water_assignment_index,
      high_water_units, &blockers);
  loom_target_compile_report_add_high_water_storage_lease_blockers(
      allocation, high_water_assignment, high_water_assignment_index,
      high_water_units, &blockers);
  return blockers;
}

static iree_status_t
loom_target_compile_report_record_allocation_high_water_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule) {
  if (!loom_target_compile_report_wants_details(
          report,
          LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_HIGH_WATER_ROWS)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      allocation->target.descriptor_set;
  if (descriptor_set == NULL || descriptor_set->reg_class_count == 0 ||
      allocation->assignment_count == 0 ||
      iree_allocator_is_null(report->allocator)) {
    return iree_ok_status();
  }
  iree_host_size_t scratch_bytes = 0;
  if (!iree_host_size_checked_mul(
          descriptor_set->reg_class_count,
          sizeof(loom_target_compile_report_allocation_high_water_scratch_t),
          &scratch_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "allocation high-water scratch is too large");
  }
  loom_target_compile_report_allocation_high_water_scratch_t* scratch = NULL;
  iree_status_t status =
      iree_allocator_malloc(report->allocator, scratch_bytes, (void**)&scratch);
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
      scratch[i] = (loom_target_compile_report_allocation_high_water_scratch_t){
          .assignment_index = UINT32_MAX,
      };
    }
  }

  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  loom_target_compile_report_pressure_origin_info_t* origin_infos = NULL;
  if (iree_status_is_ok(status) && schedule != NULL &&
      liveness->value_count != 0) {
    iree_host_size_t origin_info_bytes = 0;
    if (!iree_host_size_checked_mul(
            liveness->value_count,
            sizeof(loom_target_compile_report_pressure_origin_info_t),
            &origin_info_bytes)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "pressure origin info table is too large");
    } else {
      status = iree_allocator_malloc(report->allocator, origin_info_bytes,
                                     (void**)&origin_infos);
    }
    if (iree_status_is_ok(status)) {
      memset(origin_infos, 0, origin_info_bytes);
      status = loom_target_compile_report_build_pressure_origin_infos(
          schedule, origin_infos, liveness->value_count);
    }
  }

  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < allocation->assignment_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[i];
    if (assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER ||
        assignment->location_count == 0) {
      continue;
    }
    if (assignment->descriptor_reg_class_id >=
        descriptor_set->reg_class_count) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "allocation assignment register class exceeds descriptor set");
      break;
    }
    const uint64_t high_water_units =
        (uint64_t)assignment->location_base + assignment->location_count;
    loom_target_compile_report_allocation_high_water_scratch_t* entry =
        &scratch[assignment->descriptor_reg_class_id];
    if (high_water_units > entry->high_water_units) {
      entry->assignment_index = (uint32_t)i;
      entry->high_water_units = high_water_units;
    }
  }

  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < descriptor_set->reg_class_count; ++i) {
    const loom_target_compile_report_allocation_high_water_scratch_t* entry =
        &scratch[i];
    if (entry->assignment_index == UINT32_MAX ||
        entry->assignment_index >= allocation->assignment_count) {
      continue;
    }
    const loom_low_allocation_assignment_t* assignment =
        &allocation->assignments[entry->assignment_index];
    const loom_target_compile_report_allocation_high_water_blockers_t blockers =
        loom_target_compile_report_allocation_high_water_blockers(
            allocation, assignment, entry->assignment_index,
            entry->high_water_units);
    const loom_target_compile_report_allocation_lower_free_runs_t free_runs =
        loom_target_compile_report_allocation_lower_free_runs(
            allocation, assignment, entry->assignment_index,
            LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_ACTIVE);
    const loom_target_compile_report_allocation_lower_free_runs_t
        pressure_release_free_runs =
            loom_target_compile_report_allocation_lower_free_runs(
                allocation, assignment, entry->assignment_index,
                LOOM_TARGET_COMPILE_REPORT_STORAGE_LEASE_OCCUPANCY_PRESSURE_RELEASE_UPPER_BOUND);
    loom_target_compile_report_pressure_origin_info_t origin_info =
        loom_target_compile_report_pressure_origin_from_value(
            allocation->module, assignment->value_id);
    loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
    if (origin_infos != NULL &&
        loom_target_compile_report_liveness_value_ordinal(
            liveness, assignment->value_id, &value_ordinal) &&
        value_ordinal < liveness->value_count &&
        origin_infos[value_ordinal].kind !=
            LOOM_TARGET_COMPILE_REPORT_PRESSURE_ORIGIN_UNKNOWN) {
      origin_info = origin_infos[value_ordinal];
    }
    const loom_target_compile_report_allocation_high_water_row_t row = {
        .function_name = report->function_name,
        .value_name = loom_target_compile_report_value_name(
            allocation->module, assignment->value_id),
        .register_class = loom_target_compile_report_value_class_name(
            descriptor_set, assignment->value_class),
        .type_kind = assignment->value_class.type_kind,
        .element_type = assignment->value_class.element_type,
        .assignment_index = entry->assignment_index,
        .origin_operation_name = origin_info.operation_name,
        .origin_kind = origin_info.kind,
        .semantic_tag = origin_info.semantic_tag,
        .start_point = assignment->start_point,
        .end_point = assignment->end_point,
        .required_unit_count = assignment->unit_count,
        .location_kind =
            loom_low_allocation_location_kind_name(assignment->location_kind),
        .location_base = assignment->location_base,
        .location_count = assignment->location_count,
        .high_water_units = entry->high_water_units,
        .lower_free_unit_count = free_runs.free_unit_count,
        .lower_free_run_count = free_runs.free_run_count,
        .lower_largest_free_run_unit_count =
            free_runs.largest_free_run_unit_count,
        .lower_pressure_releasable_free_unit_count =
            pressure_release_free_runs.free_unit_count,
        .lower_pressure_releasable_free_run_count =
            pressure_release_free_runs.free_run_count,
        .lower_pressure_releasable_largest_free_run_unit_count =
            pressure_release_free_runs.largest_free_run_unit_count,
        .active_assignment_blocker_count = blockers.active_assignment_count,
        .active_assignment_blocker_units = blockers.active_assignment_units,
        .active_storage_lease_blocker_count =
            blockers.active_storage_lease_count,
        .active_storage_lease_blocker_units =
            blockers.active_storage_lease_units,
        .active_pressure_storage_lease_blocker_count =
            blockers.active_pressure_storage_lease_count,
        .active_pressure_storage_lease_blocker_units =
            blockers.active_pressure_storage_lease_units,
        .active_fallback_storage_lease_blocker_count =
            blockers.active_fallback_storage_lease_count,
        .active_fallback_storage_lease_blocker_units =
            blockers.active_fallback_storage_lease_units,
    };
    status = loom_target_compile_report_record_allocation_high_water_row(report,
                                                                         &row);
  }

  if (origin_infos != NULL) {
    iree_allocator_free(report->allocator, origin_infos);
  }
  if (scratch != NULL) {
    iree_allocator_free(report->allocator, scratch);
  }
  return status;
}

static loom_target_compile_report_allocation_failure_blocking_kind_t
loom_target_compile_report_allocation_failure_blocking_kind(
    loom_low_allocation_failure_blocking_kind_t kind) {
  switch (kind) {
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_INTERVAL_EXCEEDS_BUDGET;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_ACTIVE_ASSIGNMENT;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_LOCATION_CONSTRAINT;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_NO_ASSIGNABLE_LOCATION;
    case LOOM_LOW_ALLOCATION_FAILURE_BLOCKING_UNKNOWN:
    default:
      return LOOM_TARGET_COMPILE_REPORT_ALLOCATION_FAILURE_BLOCKING_UNKNOWN;
  }
}

static iree_string_view_t
loom_target_compile_report_allocation_failure_conflict_value_name(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_failure_t* failure) {
  return failure->conflict_value_id == LOOM_VALUE_ID_INVALID
             ? iree_string_view_empty()
             : loom_target_compile_report_value_name(
                   allocation->module, failure->conflict_value_id);
}

static const loom_block_t*
loom_target_compile_report_allocation_failure_origin_block(
    const loom_low_allocation_table_t* allocation,
    const loom_low_allocation_failure_t* failure, const loom_op_t* origin_op) {
  if (allocation->module != NULL &&
      failure->value_id < allocation->module->values.count) {
    const loom_value_t* value =
        loom_module_value(allocation->module, failure->value_id);
    if (loom_value_is_block_arg(value)) {
      return loom_value_def_block(value);
    }
  }
  return origin_op != NULL ? origin_op->parent_block : NULL;
}

static iree_status_t loom_target_compile_report_record_allocation_failure_rows(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  if (!loom_low_allocation_failure_is_present(&allocation->failure) ||
      !loom_target_compile_report_wants_details(
          report, LOOM_TARGET_COMPILE_REPORT_DETAIL_ALLOCATION_FAILURE_ROWS)) {
    return iree_ok_status();
  }
  const loom_low_allocation_failure_t* failure = &allocation->failure;
  const loom_op_t* origin_op = loom_low_diagnostic_value_origin_op(
      allocation->module, failure->value_id, allocation->function_op);
  const loom_block_t* origin_block =
      loom_target_compile_report_allocation_failure_origin_block(
          allocation, failure, origin_op);
  const loom_target_compile_report_allocation_failure_row_t row = {
      .function_name = report->function_name,
      .value_name = loom_target_compile_report_value_name(allocation->module,
                                                          failure->value_id),
      .register_class = loom_target_compile_report_value_class_name(
          allocation->target.descriptor_set, failure->value_class),
      .type_kind = failure->value_class.type_kind,
      .element_type = failure->value_class.element_type,
      .failure_code = failure->failure_code,
      .blocking_kind =
          loom_target_compile_report_allocation_failure_blocking_kind(
              failure->blocking_kind),
      .origin_operation_name =
          loom_target_compile_report_op_name(allocation->module, origin_op),
      .origin_block_name = loom_target_compile_report_block_name(
          allocation->module, origin_block),
      .start_point = failure->start_point,
      .end_point = failure->end_point,
      .required_unit_count = failure->required_unit_count,
      .budget_units = failure->budget_units,
      .peak_live_units = failure->peak_live_units,
      .location_kind =
          loom_low_allocation_location_kind_name(failure->location_kind),
      .location_base = failure->location_base,
      .location_count = failure->location_count,
      .conflict_assignment_index = failure->conflict_assignment_index,
      .conflict_value_name =
          loom_target_compile_report_allocation_failure_conflict_value_name(
              allocation, failure),
      .conflict_start_point = failure->conflict_start_point,
      .conflict_end_point = failure->conflict_end_point,
      .conflict_location_kind =
          failure->conflict_value_id == LOOM_VALUE_ID_INVALID
              ? iree_string_view_empty()
              : loom_low_allocation_location_kind_name(
                    failure->conflict_location_kind),
      .conflict_location_base = failure->conflict_location_base,
      .conflict_location_count = failure->conflict_location_count,
  };
  return loom_target_compile_report_record_allocation_failure_row(report, &row);
}

static void loom_target_compile_report_record_low_allocation_identity(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  const iree_string_view_t export_symbol =
      allocation->target.bundle_storage.export_plan.export_symbol;
  report->function_name =
      !iree_string_view_is_empty(export_symbol)
          ? export_symbol
          : loom_low_diagnostic_function_name(allocation->module,
                                              allocation->function_op);
  report->lowered_symbol = loom_low_diagnostic_function_name(
      allocation->module, allocation->function_op);
  report->target_bundle_name = allocation->target.bundle_storage.bundle.name;
  report->target_snapshot_name =
      allocation->target.bundle_storage.snapshot.name;
  report->target_export_name =
      allocation->target.bundle_storage.export_plan.name;
  report->target_export_symbol =
      allocation->target.bundle_storage.export_plan.export_symbol;
  report->target_config_name = allocation->target.bundle_storage.config.name;
}

static iree_status_t loom_target_compile_report_record_low_allocation_contents(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_table_t* schedule) {
  const loom_liveness_analysis_t* liveness = &allocation->liveness;
  loom_target_compile_report_record_allocation(
      report, allocation->assignment_count, allocation->spill_count,
      allocation->spill_plan_count, allocation->coalesced_copy_count,
      allocation->materialized_copy_count,
      allocation->storage_leases.record_count,
      allocation->storage_lease_instance_count,
      allocation->storage_release_action_count);
  iree_status_t status = loom_target_compile_report_record_pressure_rows(
      report, liveness, allocation->target.descriptor_set);
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_pressure_origin_rows(
        report, liveness, schedule, allocation->target.descriptor_set);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_schedule_band_rows(
        report, liveness, schedule);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_spill_rows(report, allocation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_allocation_failure_rows(
        report, allocation);
  }
  if (iree_status_is_ok(status)) {
    status = loom_target_compile_report_record_allocation_high_water_rows(
        report, allocation, schedule);
  }
  return status;
}

iree_status_t loom_target_compile_report_record_low_allocation(
    loom_target_compile_report_t* report,
    const loom_low_allocation_table_t* allocation) {
  loom_target_compile_report_record_low_allocation_identity(report, allocation);
  return loom_target_compile_report_record_low_allocation_contents(
      report, allocation, /*schedule=*/NULL);
}

iree_status_t loom_target_compile_report_record_low_emission_frame(
    loom_target_compile_report_t* report,
    const loom_low_emission_frame_t* frame) {
  loom_target_compile_report_record_low_frame_identity(report, frame);
  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_acquire_value_scratch(
      &frame->allocation, &value_scratch));
  const loom_liveness_analysis_t* liveness = &frame->allocation.liveness;
  uint64_t peak_live_units = 0;
  for (iree_host_size_t i = 0; i < liveness->pressure_summary_count; ++i) {
    const uint64_t live_units = liveness->pressure_summaries[i].peak_live_units;
    peak_live_units = iree_max(peak_live_units, live_units);
  }
  loom_target_compile_report_record_schedule(
      report, frame->schedule.node_count, frame->schedule.scheduled_node_count,
      frame->schedule.dependency_count, frame->schedule.resource_use_count,
      frame->schedule.hazard_gap_count, frame->schedule.model_summary_count,
      liveness->pressure_summary_count, peak_live_units);
  loom_target_compile_report_record_low_static_instruction_mix(report, frame);
  loom_target_compile_report_record_move_causes(report, frame);
  iree_status_t status =
      loom_target_compile_report_record_low_allocation_contents(
          report, &frame->allocation, &frame->schedule);
  loom_low_allocation_release_value_scratch(&value_scratch);
  return status;
}
