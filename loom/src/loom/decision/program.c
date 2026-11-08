// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/decision/program.h"

typedef struct loom_decision_program_conjunction_result_t {
  // Combined ternary conjunction outcome.
  loom_decision_truth_t feasibility;

  // First constraint deciding rejection or uncertainty, or invalid on match.
  loom_decision_program_constraint_ref_t decisive_constraint;
} loom_decision_program_conjunction_result_t;

IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_decision_program_resolve_table_operand(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_operand_ref_t ref,
    loom_decision_predicate_operand_t* out_operand) {
  if (loom_decision_program_operand_is_constant(ref)) {
    const int64_t value =
        program->constants[loom_decision_program_operand_ordinal(ref)];
    *out_operand = (loom_decision_predicate_operand_t){
        .facts = loom_value_facts_exact_i64(value),
        .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
    };
    return;
  }
  const uint32_t ordinal = loom_decision_program_operand_ordinal(ref);
  const loom_value_id_t value_id = loom_decision_program_operand_is_result(ref)
                                       ? binding->result_values[ordinal]
                                       : binding->argument_values[ordinal];
  *out_operand = (loom_decision_predicate_operand_t){
      .facts = loom_value_fact_table_lookup(binding->facts, value_id),
      .identity =
          loom_value_fact_table_query_identity(binding->facts, value_id),
  };
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_decision_program_resolve_indexed_operand(
    const loom_decision_program_t* program,
    const loom_decision_program_indexed_binding_t* binding,
    loom_decision_program_operand_ref_t ref,
    loom_decision_predicate_operand_t* out_operand) {
  if (loom_decision_program_operand_is_constant(ref)) {
    const int64_t value =
        program->constants[loom_decision_program_operand_ordinal(ref)];
    *out_operand = (loom_decision_predicate_operand_t){
        .facts = loom_value_facts_exact_i64(value),
        .identity = LOOM_DECISION_OPERAND_IDENTITY_NONE,
    };
    return;
  }
  const uint32_t ordinal = loom_decision_program_operand_ordinal(ref);
  const loom_value_id_t value_id = loom_decision_program_operand_is_result(ref)
                                       ? binding->result_values[ordinal]
                                       : binding->argument_values[ordinal];
  *out_operand = (loom_decision_predicate_operand_t){
      .facts = binding->facts[value_id],
      .identity = value_id,
  };
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline loom_decision_truth_t
loom_decision_program_finish_predicate(
    const loom_decision_predicate_operand_t operands[3],
    loom_decision_program_predicate_refiner_t predicate_refiner,
    uint8_t predicate_kind) {
  loom_decision_truth_t truth =
      loom_decision_predicate_evaluate(predicate_kind, operands);
  if (truth == LOOM_DECISION_TRUTH_UNKNOWN && predicate_refiner.fn != NULL) {
    truth = predicate_refiner.fn(predicate_refiner.user_data, predicate_kind,
                                 operands);
  }
  return truth;
}

IREE_ATTRIBUTE_NOINLINE static loom_decision_truth_t
loom_decision_program_evaluate_table_predicate(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    const loom_decision_program_predicate_t* predicate) {
  loom_decision_predicate_operand_t operands[3];
  switch (predicate->operand_count) {
    case 3:
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[0], &operands[0]);
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[1], &operands[1]);
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[2], &operands[2]);
      break;
    case 2:
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[0], &operands[0]);
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[1], &operands[1]);
      break;
    case 1:
      loom_decision_program_resolve_table_operand(
          program, binding, predicate->operands[0], &operands[0]);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("invalid decision predicate operand count");
      IREE_BUILTIN_UNREACHABLE();
  }

  return loom_decision_program_finish_predicate(operands, predicate_refiner,
                                                predicate->kind);
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline loom_decision_truth_t
loom_decision_program_evaluate_indexed_predicate(
    const loom_decision_program_t* program,
    const loom_decision_program_indexed_binding_t* binding,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    uint32_t predicate_ordinal) {
  const loom_decision_program_predicate_t* predicate =
      &program->predicates[predicate_ordinal];
  loom_decision_predicate_operand_t operands[3];
  switch (predicate->operand_count) {
    case 3:
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[0], &operands[0]);
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[1], &operands[1]);
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[2], &operands[2]);
      break;
    case 2:
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[0], &operands[0]);
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[1], &operands[1]);
      break;
    case 1:
      loom_decision_program_resolve_indexed_operand(
          program, binding, predicate->operands[0], &operands[0]);
      break;
    default:
      IREE_ASSERT_UNREACHABLE("invalid decision predicate operand count");
      IREE_BUILTIN_UNREACHABLE();
  }
  return loom_decision_program_finish_predicate(operands, predicate_refiner,
                                                predicate->kind);
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline loom_decision_program_conjunction_result_t
loom_decision_program_evaluate_conjunction(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* table_binding,
    const loom_decision_program_indexed_binding_t* indexed_binding,
    bool use_indexed_binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_conjunction_t conjunction) {
  loom_decision_program_conjunction_result_t result = {
      .feasibility = LOOM_DECISION_TRUTH_TRUE,
      .decisive_constraint = LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
  };
  for (uint16_t i = 0; i < conjunction.feature_count; ++i) {
    const uint32_t feature_ordinal = conjunction.first_feature + i;
    const loom_decision_truth_t truth =
        feature_evaluator.fn(feature_evaluator.user_data, feature_ordinal);
    if (truth == LOOM_DECISION_TRUTH_FALSE) {
      return (loom_decision_program_conjunction_result_t){
          .feasibility = LOOM_DECISION_TRUTH_FALSE,
          .decisive_constraint =
              loom_decision_program_feature_constraint_ref(feature_ordinal),
      };
    }
    if (truth == LOOM_DECISION_TRUTH_UNKNOWN) {
      result.feasibility = LOOM_DECISION_TRUTH_UNKNOWN;
      if (result.decisive_constraint ==
          LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID) {
        result.decisive_constraint =
            loom_decision_program_feature_constraint_ref(feature_ordinal);
      }
    }
  }
  for (uint16_t i = 0; i < conjunction.predicate_count; ++i) {
    const uint32_t predicate_ordinal = conjunction.first_predicate + i;
    const loom_decision_truth_t truth =
        use_indexed_binding ? loom_decision_program_evaluate_indexed_predicate(
                                  program, indexed_binding, predicate_refiner,
                                  predicate_ordinal)
                            : loom_decision_program_evaluate_table_predicate(
                                  program, table_binding, predicate_refiner,
                                  &program->predicates[predicate_ordinal]);
    if (truth == LOOM_DECISION_TRUTH_FALSE) {
      return (loom_decision_program_conjunction_result_t){
          .feasibility = LOOM_DECISION_TRUTH_FALSE,
          .decisive_constraint =
              loom_decision_program_predicate_constraint_ref(predicate_ordinal),
      };
    }
    if (truth == LOOM_DECISION_TRUTH_UNKNOWN) {
      result.feasibility = LOOM_DECISION_TRUTH_UNKNOWN;
      if (result.decisive_constraint ==
          LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID) {
        result.decisive_constraint =
            loom_decision_program_predicate_constraint_ref(predicate_ordinal);
      }
    }
  }
  return result;
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_decision_program_append_live(uint32_t action_ordinal,
                                  uint32_t* live_action_ordinals,
                                  uint32_t* live_action_count) {
  if (live_action_ordinals != NULL) {
    live_action_ordinals[*live_action_count] = action_ordinal;
  }
  ++*live_action_count;
}

static loom_decision_program_result_t loom_decision_program_empty_result(void) {
  return (loom_decision_program_result_t){
      .kind = LOOM_DECISION_PROGRAM_RESULT_NO_MATCH,
      .action_ordinal = LOOM_DECISION_PROGRAM_ACTION_INVALID,
      .unresolved_action_ordinal = LOOM_DECISION_PROGRAM_ACTION_INVALID,
      .unresolved_constraint = LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
  };
}

static void loom_decision_program_preserve_all_choices(
    const loom_decision_program_t* program, uint32_t* live_action_ordinals,
    uint32_t* out_live_action_count) {
  *out_live_action_count = 0;
  for (uint32_t i = 0; i < program->choice_count; ++i) {
    loom_decision_program_append_live(program->choices[i].action_ordinal,
                                      live_action_ordinals,
                                      out_live_action_count);
  }
}

IREE_ATTRIBUTE_ALWAYS_INLINE static inline void
loom_decision_program_evaluate_impl(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* table_binding,
    const loom_decision_program_indexed_binding_t* indexed_binding,
    bool use_indexed_binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result) {
  *out_live_action_count = 0;
  *out_result = loom_decision_program_empty_result();

  const loom_decision_program_conjunction_result_t hard_requirements =
      loom_decision_program_evaluate_conjunction(
          program, table_binding, indexed_binding, use_indexed_binding,
          feature_evaluator, predicate_refiner, program->hard_requirements);
  if (hard_requirements.feasibility == LOOM_DECISION_TRUTH_FALSE) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT;
    return;
  }
  if (hard_requirements.feasibility == LOOM_DECISION_TRUTH_UNKNOWN) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    out_result->unresolved_constraint = hard_requirements.decisive_constraint;
    if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
      for (uint32_t i = 0; i < program->choice_count; ++i) {
        const loom_decision_program_choice_t* choice = &program->choices[i];
        const loom_decision_program_conjunction_result_t choice_result =
            loom_decision_program_evaluate_conjunction(
                program, table_binding, indexed_binding, use_indexed_binding,
                feature_evaluator, predicate_refiner, choice->conjunction);
        if (choice_result.feasibility != LOOM_DECISION_TRUTH_FALSE) {
          loom_decision_program_append_live(choice->action_ordinal,
                                            live_action_ordinals,
                                            out_live_action_count);
        }
      }
    }
    return;
  }

  uint32_t highest_maybe_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  loom_decision_program_constraint_ref_t highest_maybe_constraint =
      LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID;
  bool has_higher_maybe = false;
  uint32_t first_choice = 0;
  for (uint32_t group_ordinal = 0;
       group_ordinal < program->priority_group_count; ++group_ordinal) {
    const loom_decision_program_priority_group_t* group =
        &program->priority_groups[group_ordinal];
    bool group_has_maybe = false;
    uint8_t match_count = 0;
    uint32_t selected_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
    for (uint32_t i = 0; i < group->choice_count; ++i) {
      const loom_decision_program_choice_t* choice =
          &program->choices[first_choice + i];
      const loom_decision_program_conjunction_result_t choice_result =
          loom_decision_program_evaluate_conjunction(
              program, table_binding, indexed_binding, use_indexed_binding,
              feature_evaluator, predicate_refiner, choice->conjunction);
      if (choice_result.feasibility == LOOM_DECISION_TRUTH_UNKNOWN) {
        group_has_maybe = true;
        if (highest_maybe_action == LOOM_DECISION_PROGRAM_ACTION_INVALID) {
          highest_maybe_action = choice->action_ordinal;
          highest_maybe_constraint = choice_result.decisive_constraint;
        }
        if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
          loom_decision_program_append_live(choice->action_ordinal,
                                            live_action_ordinals,
                                            out_live_action_count);
        }
      } else if (choice_result.feasibility == LOOM_DECISION_TRUTH_TRUE) {
        if (match_count == 0) {
          selected_action = choice->action_ordinal;
        }
        if (match_count < 2) {
          ++match_count;
        }
        loom_decision_program_append_live(choice->action_ordinal,
                                          live_action_ordinals,
                                          out_live_action_count);
      }
    }

    if (match_count == 0) {
      has_higher_maybe |= group_has_maybe;
      first_choice += group->choice_count;
      continue;
    }
    if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED &&
        (has_higher_maybe || group_has_maybe)) {
      out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
      out_result->unresolved_action_ordinal = highest_maybe_action;
      out_result->unresolved_constraint = highest_maybe_constraint;
      return;
    }
    if (match_count > 1) {
      out_result->kind = LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS;
      return;
    }

    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_SELECTED;
    out_result->action_ordinal = selected_action;
    if (live_action_ordinals != NULL) {
      live_action_ordinals[0] = selected_action;
    }
    *out_live_action_count = 1;
    return;
  }

  if (highest_maybe_action != LOOM_DECISION_PROGRAM_ACTION_INVALID) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    out_result->unresolved_action_ordinal = highest_maybe_action;
    out_result->unresolved_constraint = highest_maybe_constraint;
    if (resolution_policy == LOOM_DECISION_PROGRAM_SELECT_PROVEN) {
      *out_live_action_count = 0;
    }
    return;
  }

  if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
    loom_decision_program_preserve_all_choices(program, live_action_ordinals,
                                               out_live_action_count);
  } else {
    *out_live_action_count = 0;
  }
}

void loom_decision_program_evaluate(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result) {
  loom_decision_program_evaluate_impl(
      program, binding, /*indexed_binding=*/NULL,
      /*use_indexed_binding=*/false, feature_evaluator, predicate_refiner,
      resolution_policy, live_action_ordinals, out_live_action_count,
      out_result);
}

void loom_decision_program_evaluate_indexed(
    const loom_decision_program_t* program,
    const loom_decision_program_indexed_binding_t* binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result) {
  loom_decision_program_evaluate_impl(program, /*table_binding=*/NULL, binding,
                                      /*use_indexed_binding=*/true,
                                      feature_evaluator, predicate_refiner,
                                      resolution_policy, live_action_ordinals,
                                      out_live_action_count, out_result);
}

static void loom_decision_program_reduce_evidence(
    const loom_decision_program_t* program,
    const loom_decision_program_choice_evidence_t* choice_evidence,
    loom_decision_program_resolution_policy_t resolution_policy,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result) {
  uint32_t highest_maybe_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
  loom_decision_program_constraint_ref_t highest_maybe_constraint =
      LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID;
  bool has_higher_maybe = false;
  uint32_t first_choice = 0;
  for (uint32_t group_ordinal = 0;
       group_ordinal < program->priority_group_count; ++group_ordinal) {
    const loom_decision_program_priority_group_t* group =
        &program->priority_groups[group_ordinal];
    bool group_has_maybe = false;
    uint8_t match_count = 0;
    uint32_t selected_action = LOOM_DECISION_PROGRAM_ACTION_INVALID;
    for (uint32_t i = 0; i < group->choice_count; ++i) {
      const uint32_t choice_ordinal = first_choice + i;
      const loom_decision_program_choice_t* choice =
          &program->choices[choice_ordinal];
      const loom_decision_program_choice_evidence_t* evidence =
          &choice_evidence[choice_ordinal];
      if (evidence->feasibility == LOOM_DECISION_TRUTH_UNKNOWN) {
        group_has_maybe = true;
        if (highest_maybe_action == LOOM_DECISION_PROGRAM_ACTION_INVALID) {
          highest_maybe_action = choice->action_ordinal;
          highest_maybe_constraint = evidence->decisive_constraint;
        }
        if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
          loom_decision_program_append_live(choice->action_ordinal,
                                            live_action_ordinals,
                                            out_live_action_count);
        }
      } else if (evidence->feasibility == LOOM_DECISION_TRUTH_TRUE) {
        if (match_count == 0) {
          selected_action = choice->action_ordinal;
        }
        if (match_count < 2) {
          ++match_count;
        }
        loom_decision_program_append_live(choice->action_ordinal,
                                          live_action_ordinals,
                                          out_live_action_count);
      }
    }

    if (match_count == 0) {
      has_higher_maybe |= group_has_maybe;
      first_choice += group->choice_count;
      continue;
    }
    if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED &&
        (has_higher_maybe || group_has_maybe)) {
      out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
      out_result->unresolved_action_ordinal = highest_maybe_action;
      out_result->unresolved_constraint = highest_maybe_constraint;
      return;
    }
    if (match_count > 1) {
      out_result->kind = LOOM_DECISION_PROGRAM_RESULT_AMBIGUOUS;
      return;
    }

    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_SELECTED;
    out_result->action_ordinal = selected_action;
    live_action_ordinals[0] = selected_action;
    *out_live_action_count = 1;
    return;
  }

  if (highest_maybe_action != LOOM_DECISION_PROGRAM_ACTION_INVALID) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    out_result->unresolved_action_ordinal = highest_maybe_action;
    out_result->unresolved_constraint = highest_maybe_constraint;
    if (resolution_policy == LOOM_DECISION_PROGRAM_SELECT_PROVEN) {
      *out_live_action_count = 0;
    }
    return;
  }
  if (resolution_policy == LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
    loom_decision_program_preserve_all_choices(program, live_action_ordinals,
                                               out_live_action_count);
  } else {
    *out_live_action_count = 0;
  }
}

void loom_decision_program_evaluate_all(
    const loom_decision_program_t* program,
    const loom_decision_program_binding_t* binding,
    loom_decision_program_feature_evaluator_t feature_evaluator,
    loom_decision_program_predicate_refiner_t predicate_refiner,
    loom_decision_program_resolution_policy_t resolution_policy,
    loom_decision_program_choice_evidence_t* choice_evidence,
    uint32_t* live_action_ordinals, uint32_t* out_live_action_count,
    loom_decision_program_result_t* out_result) {
  *out_live_action_count = 0;
  *out_result = loom_decision_program_empty_result();

  const loom_decision_program_conjunction_result_t hard_requirements =
      loom_decision_program_evaluate_conjunction(
          program, binding, /*indexed_binding=*/NULL,
          /*use_indexed_binding=*/false, feature_evaluator, predicate_refiner,
          program->hard_requirements);
  if (hard_requirements.feasibility == LOOM_DECISION_TRUTH_FALSE) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_HARD_REJECT;
    return;
  }
  if (hard_requirements.feasibility == LOOM_DECISION_TRUTH_UNKNOWN) {
    out_result->kind = LOOM_DECISION_PROGRAM_RESULT_UNRESOLVED;
    out_result->unresolved_constraint = hard_requirements.decisive_constraint;
    if (resolution_policy != LOOM_DECISION_PROGRAM_DEFER_UNRESOLVED) {
      return;
    }
    for (uint32_t i = 0; i < program->choice_count; ++i) {
      const loom_decision_program_choice_t* choice = &program->choices[i];
      const loom_decision_program_conjunction_result_t choice_result =
          loom_decision_program_evaluate_conjunction(
              program, binding, /*indexed_binding=*/NULL,
              /*use_indexed_binding=*/false, feature_evaluator,
              predicate_refiner, choice->conjunction);
      choice_evidence[i] = (loom_decision_program_choice_evidence_t){
          .feasibility = choice_result.feasibility,
          .decisive_constraint = choice_result.decisive_constraint,
      };
      if (choice_result.feasibility != LOOM_DECISION_TRUTH_FALSE) {
        loom_decision_program_append_live(choice->action_ordinal,
                                          live_action_ordinals,
                                          out_live_action_count);
      }
    }
    return;
  }

  for (uint32_t i = 0; i < program->choice_count; ++i) {
    const loom_decision_program_conjunction_result_t choice_result =
        loom_decision_program_evaluate_conjunction(
            program, binding, /*indexed_binding=*/NULL,
            /*use_indexed_binding=*/false, feature_evaluator, predicate_refiner,
            program->choices[i].conjunction);
    choice_evidence[i] = (loom_decision_program_choice_evidence_t){
        .feasibility = choice_result.feasibility,
        .decisive_constraint = choice_result.decisive_constraint,
    };
  }
  loom_decision_program_reduce_evidence(program, choice_evidence,
                                        resolution_policy, live_action_ordinals,
                                        out_live_action_count, out_result);
}
