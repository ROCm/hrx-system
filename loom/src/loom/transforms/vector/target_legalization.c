// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/to_scalar.h"

static loom_vector_mma_to_scalar_options_t loom_vector_mma_options(
    const loom_target_legalization_context_t* context) {
  const loom_target_contract_query_result_t* query_result =
      context->contract_query_result;
  loom_vector_to_scalar_flags_t flags = LOOM_VECTOR_TO_SCALAR_FLAG_NONE;
  if (loom_kernel_def_isa(context->function.op)) {
    flags |= LOOM_VECTOR_TO_SCALAR_FLAG_ALLOW_SUBGROUP_COMMUNICATION;
  }
  return (loom_vector_mma_to_scalar_options_t){
      .matrix_fragment_layout =
          query_result ? query_result->selected_matrix_fragment_layout : NULL,
      .flags = flags,
  };
}

static bool loom_vector_mma_has_fragment_store_user(const loom_module_t* module,
                                                    const loom_op_t* op) {
  const loom_value_t* result =
      loom_module_value(module, loom_vector_mma_result(op));
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    const loom_op_t* user = loom_use_user_op(*use);
    if (loom_vector_fragment_store_isa(user) &&
        loom_vector_fragment_store_value(user) == loom_vector_mma_result(op)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_vector_legalize_reduce_axes(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_axes_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_descriptor(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_descriptor_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_dotf(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_dotf_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_mma(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }
  if (loom_vector_mma_has_fragment_store_user(context->module, op)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  const loom_vector_mma_to_scalar_options_t options =
      loom_vector_mma_options(context);
  IREE_RETURN_IF_ERROR(loom_vector_mma_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, options, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_bits(
                          context->pass, context->rewriter, op, options),
      .source_rejection_detail =
          rewritten ? 0
                    : loom_vector_mma_to_scalar_reference_rejection_detail(
                          context->pass, context->rewriter, op, options),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_fragment_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_fragment_store_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  *out_result = (loom_target_legalizer_result_t){
      .action = rewritten
                    ? LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN
                    : LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
      .source_rejection_bits =
          rewritten
              ? 0
              : loom_vector_fragment_store_to_scalar_reference_rejection_bits(
                    context->pass, context->rewriter, op),
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_extract(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (context->mode != LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_DEFER,
    };
    return iree_ok_status();
  }

  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_extract_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static const loom_target_legalizer_entry_t kVectorLegalizerEntries[] = {
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE,
        .legalize = loom_vector_legalize_reduce,
    },
    {
        .root_kind = LOOM_OP_VECTOR_REDUCE_AXES,
        .legalize = loom_vector_legalize_reduce_axes,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTU,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_EXTRACTS,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITFIELD_INSERT,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITPACK,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKU,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_BITUNPACKS,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_vector_legalize_dotf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT2F,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4I,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT8I4,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOT4F8,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_MMA,
        .legalize = loom_vector_legalize_mma,
    },
    {
        .root_kind = LOOM_OP_VECTOR_STORE,
        .legalize = loom_vector_legalize_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_FRAGMENT_STORE,
        .legalize = loom_vector_legalize_fragment_store,
    },
    {
        .root_kind = LOOM_OP_VECTOR_EXTRACT,
        .legalize = loom_vector_legalize_extract,
    },
};

static const loom_target_legalizer_provider_t kVectorLegalizerProvider = {
    .name = IREE_SVL("vector"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .entries = kVectorLegalizerEntries,
    .entry_count = IREE_ARRAYSIZE(kVectorLegalizerEntries),
};

const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void) {
  return &kVectorLegalizerProvider;
}
