// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_OPS_SCF_SELECTIFY_FACTS_H_
#define LOOM_OPS_SCF_SELECTIFY_FACTS_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |op| is a removable branch-local fact identity.
bool loom_scf_if_op_is_strippable_fact_identity(const loom_module_t* module,
                                                const loom_op_t* op,
                                                loom_trait_flags_t traits);

typedef struct loom_scf_if_selectification_facts_t {
  // Facts inferred for the result after both arms execute eagerly.
  loom_value_facts_t eager_facts;
  // Facts published by the original structured conditional.
  loom_value_facts_t structured_facts;
} loom_scf_if_selectification_facts_t;

// Previews the bounded arm programs as they will execute after selectification.
// Branch-local fact identities are removed when |strips_fact_identities| is
// true. The preview rejects operations whose fact-dependent execution safety
// does not hold without the branch and rejects result facts that cannot be
// preserved by directly representable integer predicates. On success, each
// entry in |out_result_facts| records both the facts inferred for the eager
// replacement and the facts retained from the structured result. Both records
// omit context-local extensions and remain valid after |scratch_arena| is
// reset.
iree_status_t loom_scf_if_preview_selectification(
    loom_op_t* op, loom_rewriter_t* rewriter, const loom_block_t* then_block,
    const loom_op_t* then_yield, const loom_block_t* else_block,
    const loom_op_t* else_yield, bool strips_fact_identities,
    iree_arena_allocator_t* scratch_arena,
    loom_scf_if_selectification_facts_t* out_result_facts,
    bool* out_can_selectify);

// Materializes stronger preserved integer facts as grouped index.assume and
// scalar.assume identities. Entries without stronger directly representable
// facts retain their existing replacements.
iree_status_t loom_scf_if_materialize_preserved_result_facts(
    loom_op_t* op, loom_rewriter_t* rewriter,
    const loom_scf_if_selectification_facts_t* result_facts,
    loom_value_id_t* inout_replacements);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_SCF_SELECTIFY_FACTS_H_
