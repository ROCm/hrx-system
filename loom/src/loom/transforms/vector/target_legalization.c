// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/vector/memory.h"
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
  const loom_value_id_t result_id = loom_vector_mma_result(op);
  const loom_value_t* result = loom_module_value(module, result_id);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(result, use) {
    const loom_op_t* user = loom_use_user_op(*use);
    if (loom_vector_memory_op_footprint_kind(module, user) !=
        LOOM_VECTOR_MEMORY_FOOTPRINT_FRAGMENT) {
      continue;
    }
    loom_memory_access_t access = loom_memory_access_cast(module, user);
    if (loom_memory_access_operation_kind(access) ==
            LOOM_MEMORY_ACCESS_OPERATION_STORE &&
        loom_memory_access_value(access) == result_id) {
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

// Expands a variadic vector constructor into the fixed-arity structural ops
// that targets commonly select for linear register vectors. Splatting the first
// lane provides a defined seed without requiring a target-level poison or zero
// materialization. Higher-rank constructors remain intact so the target can
// diagnose or lower their multidimensional representation directly.
static iree_status_t loom_vector_legalize_from_elements(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  if (elements.count == 0) return iree_ok_status();
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_from_elements_result(op));
  if (!loom_type_is_vector(result_type) ||
      !loom_type_is_all_static(result_type) ||
      loom_type_rank(result_type) != 1) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(&rewriter->builder,
                                               elements.values[0], result_type,
                                               op->location, &splat_op));
  loom_value_id_t replacement = loom_vector_splat_result(splat_op);

  for (iree_host_size_t element_index = 1; element_index < elements.count;
       ++element_index) {
    const int64_t static_index = (int64_t)element_index;
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        &rewriter->builder, elements.values[element_index], replacement, NULL,
        0, &static_index, 1, result_type, op->location, &insert_op));
    replacement = loom_vector_insert_result(insert_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
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

static iree_status_t loom_vector_legalize_transform(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_transform_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_legalize_decode(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  bool rewritten = false;
  IREE_RETURN_IF_ERROR(loom_vector_decode_to_scalar_rewrite_op(
      context->pass, context->rewriter, op, &rewritten));
  if (rewritten) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }
  if (context->mode == LOOM_TARGET_LEGALIZATION_MODE_FINAL) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REJECT_UNSUPPORTED_FINAL,
        .source_rejection_bits =
            loom_vector_decode_to_scalar_reference_rejection_bits(
                context->pass, context->rewriter, op),
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

static const loom_target_legalizer_rule_t kVectorLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_VECTOR_FROM_ELEMENTS,
        .legalize = loom_vector_legalize_from_elements,
    },
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
        .root_kind = LOOM_OP_VECTOR_SHLI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRSI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_SHRUI,
        .legalize = loom_vector_legalize_descriptor,
    },
    {
        .root_kind = LOOM_OP_VECTOR_DOTF,
        .legalize = loom_vector_legalize_dotf,
    },
    {
        .root_kind = LOOM_OP_VECTOR_TRANSFORM,
        .legalize = loom_vector_legalize_transform,
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
        .root_kind = LOOM_OP_VECTOR_DECODE,
        .legalize = loom_vector_legalize_decode,
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
    .rules = kVectorLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kVectorLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_vector_target_legalizer_provider(
    void) {
  return &kVectorLegalizerProvider;
}
