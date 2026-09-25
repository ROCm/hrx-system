// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_VM_FUNCTION_H_
#define LOOM_TARGET_ARCH_VM_FUNCTION_H_

#include "iree/io/stream.h"
#include "iree/vm/bytecode/wire/module.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_vm_module_plan_t loom_vm_module_plan_t;

// Exact logical signature retained by module collection for function emission.
typedef struct loom_vm_function_signature_t {
  // Physical argument/result counts. Serialization assigns descriptor_base.
  iree_vm_bytecode_v0_signature_row_t row;
  // Module-owned argument descriptors followed by result descriptors, in
  // source order. Metadata planning finalizes them before function emission.
  iree_vm_bytecode_v0_signature_descriptor_row_t* fields;
} loom_vm_function_signature_t;

// Schedules and allocates one prepared VM function with the common frame
// builder, then appends its instruction stream exactly once. |out_row| receives
// its byte length and frame high waters; the module writer owns callable and
// section-relative offsets. Branches target block markers using signed word
// offsets patched after emission. Constants retain canonical Low form until
// emission chooses an equivalent compact encoding of the complete value cell.
// Block offsets and byte lengths come from the emitted stream, not nominal
// sizes. The shared allocator owns edge and packet moves, including cycle
// temporaries. Common allocation repair materializes scalar spills; the
// scheduler's stack layout owns their relative byte offsets. Outgoing overflow
// packets occupy the canonical offset-zero prefix; ordinary locals follow,
// then aligned call snapshots. Overflow entry loads execute once before the
// branchable body; overflow returns store exact value cells before direct
// register permutations. Reference overflow uses an independent local-ref
// prefix ahead of caller snapshots. Returning a ref through overflow preserves
// its source in one local-ref slot until all aliased results are published.
// All compiler scratch belongs to |request|'s arena and can be released when
// this call returns. The emitted stream, scalar |out_row| fields, and
// module-owned rodata references retain no planning storage. Structured frame
// errors are forwarded to its diagnostic emitter and return OK with
// |out_emitted| false. Sink and infrastructure failures return a status and
// also leave |out_emitted| false.
// |functions| supplies callable signatures and symbol ordinals; data operands
// append their referenced payload once to its read-only section plan.
iree_status_t loom_vm_function_emit(
    const loom_target_emit_request_t* request, loom_func_like_t function,
    const loom_target_function_version_t* function_version,
    const loom_vm_function_signature_t* signature,
    loom_vm_module_plan_t* functions, iree_io_stream_t* stream,
    bool* out_emitted, iree_vm_bytecode_v0_function_row_t* out_row);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_FUNCTION_H_
