// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_REF_VERIFY_H_
#define LOOM_OPS_REF_VERIFY_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies the owned-to-borrowed projection of one exact nominal payload.
// Source and Low borrowing share this type contract; Low unwraps its complete
// register carriers before calling this external-input verifier.
iree_status_t loom_ref_verify_borrow_types(const loom_op_t* op,
                                           loom_type_t owner_type,
                                           loom_type_t pointer_type,
                                           iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_REF_VERIFY_H_
