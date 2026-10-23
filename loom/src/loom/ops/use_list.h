// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact operand use lists and ownership-sensitive use summaries.

#ifndef LOOM_OPS_USE_LIST_H_
#define LOOM_OPS_USE_LIST_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initial overflow capacity when transitioning from inline to overflow.
// 8 covers the common case of values used 4-8 times without further
// reallocation. Values used more than 8 times get geometric growth.
#define LOOM_USE_INITIAL_OVERFLOW_CAPACITY 8

// Updates exact ownership summaries retained beside an overflow use list.
static inline void loom_value_update_overflow_use_counts(
    loom_value_t* value, loom_use_flags_t old_flags,
    loom_use_flags_t new_flags) {
  if (old_flags == new_flags || !loom_value_has_overflow_uses(value)) {
    return;
  }
  value->overflow_ownership_use_count -= old_flags != 0;
  value->overflow_ownership_use_count += new_flags != 0;
  value->overflow_consuming_use_count -=
      iree_any_bit_set(old_flags, LOOM_USE_FLAG_CONSUMES);
  value->overflow_consuming_use_count +=
      iree_any_bit_set(new_flags, LOOM_USE_FLAG_CONSUMES);
}

// Registers a newly constructed operation's operand occurrences and their
// ownership classifications. Called once by finalization or use reconstruction
// after operands, result descriptors and explicit ties are available.
iree_status_t loom_op_record_operand_uses(loom_module_t* module, loom_op_t* op);

// Refreshes existing use classifications after changing an operation's result
// shape or populating a previously absent operand. Operand use records must
// already be registered. Ordinary operand replacement carries its old tag.
void loom_op_refresh_operand_ownership(loom_module_t* module,
                                       const loom_op_t* op);

// Removes a use record: |user_op| no longer uses |value_id| at
// |operand_index|. Reads the operand's retained use index, swaps with the last
// entry, and updates the moved operand's index in O(1). The operand must retain
// the exact use entry established by construction or a previous operand edit.
// No overflow-to-inline transition (arena cannot free the overflow
// array; loom_module_compute_uses handles repack).
void loom_value_remove_use(loom_module_t* module, loom_value_id_t value_id,
                           loom_op_t* user_op, uint16_t operand_index);

// Changes an operand on an existing op, maintaining use lists. Removes
// the use from the old value, writes the new value ID, and adds a use
// to the new value. Skips LOOM_VALUE_ID_INVALID for both old and new.
iree_status_t loom_op_set_operand(loom_module_t* module, loom_op_t* op,
                                  uint16_t operand_index,
                                  loom_value_id_t new_value_id);

// Rebuilds all use lists from scratch by walking every live op in the
// module. Clears all values' use data, then re-adds uses from operands.
// Used after parsing (the parser fills operands but not use lists) and
// as a recovery path after bulk IR mutations.
iree_status_t loom_module_compute_uses(loom_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_OPS_USE_LIST_H_
