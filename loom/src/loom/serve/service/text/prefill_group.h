// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
#define LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
#include "loom/serve/service/text/model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Publishes source-owned state mapping after complete A+B growth.
iree_status_t loom_serve_text_model_publish_state_bindings(
    loom_serve_text_model_t* model);

// Source ABI text_prefill_group_v1:
//   text_prefill_group_plan_v1(state) -> descriptor
//     LE u32 header {version=1, shape_count, layer_count, resource_abi};
//     followed by stage IDs [shape][layer][prefix,ffn,residual]. UINT32_MAX
//     omits FFN/residual. Prefix publishes all shared state required by B.
//     Resource ABI 1 supplies the six ordinary packed roots; ABI 2 additionally
//     supplies the six draft roots in ordinary epoch order. Draft models must
//     declare ABI 2 and include bank-local warm/capture work in their leaves.
//   text_encode_prefill_group_v1(state, plan_a, input_a, plan_b, input_b,
//                                metadata_a, tokens_a, metadata_b, tokens_b)
//     Receives owned nine-u32 semantic plans and independent packet storage.
//     Reserves B's position/source_position without committing row state. The
//     complete packet extents are uploaded before either bank executes.
//   text_publish_prefill_group_v1(plan_a, plan_b, outputs_b, progress)
//     Runs once after successful joined readiness. Writes one ordinary progress
//     record for the total known input and optional B selection. No partial
//     group result is published. Source must not require mid-group host reads.
// Storage extents come from the source's existing packed storage declaration.
// The model owner freezes the complete A+B page map before group submission.
//
// Called during linking after source and native stages are ready. Resolves the
// explicit text_prefill_group_v1 exports and reserves both banks before
// serving.
iree_status_t loom_serve_text_prefill_group_initialize(
    loom_serve_text_model_t* model);
iree_status_t loom_serve_text_prefill_group_complete(
    loom_serve_text_model_t* model, loom_serve_text_result_t* result);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_PREFILL_GROUP_H_
