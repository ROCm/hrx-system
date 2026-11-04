// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/reporting/report.h"

void loom_target_compile_report_accumulate_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source) {
#define LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(field) \
  target->field += source->field
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(descriptor_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(unknown_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(scalar_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(vector_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(matrix_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(mfma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(smfmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(wmma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(swmmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(dot_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(global_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(global_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(global_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(buffer_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(buffer_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(flat_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(local_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(scalar_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(private_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(generic_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(memory_read_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(memory_write_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(memory_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(memory_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(global_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(global_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(buffer_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(buffer_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(flat_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(flat_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(local_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(local_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(scalar_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(scalar_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(private_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(private_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(unclassified_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(unclassified_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(atomic_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(branch_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(execution_barrier_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(control_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(conversion_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(cache_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD(register_move_count);
#undef LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_FIELD
}

bool loom_target_compile_report_accumulate_scaled_static_mix(
    loom_target_compile_report_static_instruction_mix_t* target,
    const loom_target_compile_report_static_instruction_mix_t* source,
    uint64_t scale) {
#define LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(field)             \
  do {                                                                        \
    uint64_t scaled_value = 0;                                                \
    if (!iree_checked_mul_u64(source->field, scale, &scaled_value) ||         \
        !iree_checked_add_u64(target->field, scaled_value, &target->field)) { \
      return false;                                                           \
    }                                                                         \
  } while (0)
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(descriptor_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(unknown_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(vector_alu_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(matrix_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(mfma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(smfmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(wmma_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(swmmac_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(dot_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_load_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_store_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(generic_memory_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      memory_read_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      memory_write_unknown_width_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(memory_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(memory_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(global_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_load_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(buffer_store_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(flat_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(local_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(scalar_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(private_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      unclassified_read_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(
      unclassified_write_byte_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(atomic_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(branch_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(execution_barrier_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(control_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(conversion_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(cache_count);
  LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD(register_move_count);
#undef LOOM_TARGET_COMPILE_REPORT_ACCUMULATE_SCALED_FIELD
  return true;
}
