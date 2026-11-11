// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks complete launch-site class collection over an immutable kernel
// classifier. Classifier construction and command fact propagation happen once
// in the fixture; each iteration allocates, evaluates, groups, and releases one
// collection as a command-product planner would.

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/transforms/kernel/kernel_class_classifier.h"

namespace loom {
namespace {

// Distinguishes where structural requirements enter the final class key.
enum class ShapeMode { kNone, kSelected, kGeneric, kDifferent, kSource };

class KernelClassClassifierBenchmarkFixture {
 public:
  KernelClassClassifierBenchmarkFixture(iree_host_size_t site_count,
                                        uint32_t decision_count,
                                        ShapeMode shape_mode)
      : shape_mode_(shape_mode),
        argument_values_(site_count * kKernelArgumentCount),
        sites_(site_count),
        decisions_(decision_count) {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &facts_arena_);
    iree_arena_initialize(&block_pool_, &collection_arena_);

    IREE_CHECK_OK(loom_value_fact_table_initialize(
        &fact_table_, &facts_arena_, site_count * kKernelArgumentCount));
    for (iree_host_size_t i = 0; i < site_count; ++i) {
      const iree_host_size_t first_argument = i * kKernelArgumentCount;
      argument_values_[first_argument + 0] =
          static_cast<loom_value_id_t>(first_argument + 0);
      argument_values_[first_argument + 1] =
          static_cast<loom_value_id_t>(first_argument + 1);
      argument_values_[first_argument + 2] =
          static_cast<loom_value_id_t>(first_argument + 2);
      IREE_CHECK_OK(loom_value_fact_table_define(
          &fact_table_, first_argument + 0,
          loom_value_facts_exact_i64(static_cast<int64_t>(i))));
      IREE_CHECK_OK(loom_value_fact_table_define(
          &fact_table_, first_argument + 1,
          loom_value_facts_exact_i64(static_cast<int64_t>((i * 7) % 5))));
      IREE_CHECK_OK(loom_value_fact_table_define(
          &fact_table_, first_argument + 2,
          loom_value_facts_exact_i64(static_cast<int64_t>(i % 7))));
      sites_[i] = {
          .facts = &fact_table_,
          .argument_values = &argument_values_[first_argument],
      };
    }

    predicate_ = {
        .kind = LOOM_PREDICATE_GE,
        .operand_count = 2,
        .reserved = {},
        .operands =
            {
                loom_decision_program_argument_ref(0),
                loom_decision_program_constant_ref(0),
            },
    };
    choices_[0] = {
        .conjunction =
            {
                .first_predicate = 0,
                .first_feature = 0,
                .predicate_count = 1,
                .feature_count = 0,
            },
        .action_ordinal = 0,
    };
    choices_[1] = {
        .conjunction = {},
        .action_ordinal = 1,
    };
    actions_[0].choice_ordinal = 0;
    actions_[1].choice_ordinal = 1;
    priority_groups_[0].choice_count = 1;
    priority_groups_[1].choice_count = 1;
    model_.program = {
        .predicates = &predicate_,
        .choices = choices_,
        .priority_groups = priority_groups_,
        .constants = constants_,
        .hard_requirements = {},
        .predicate_count = 1,
        .feature_count = 0,
        .constant_count = 1,
        .choice_count = 2,
        .priority_group_count = 2,
    };

    projection_terms_[0] = {
        .coefficient = 1,
        .argument_ordinal = 0,
        .reserved = {},
    };
    projection_terms_[1] = {
        .coefficient = 1,
        .argument_ordinal = 1,
        .reserved = {},
    };
    projections_[0] = {
        .source_value_id = 0,
        .kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE,
        .reserved = {},
        .constant = 0,
        .terms = projection_terms_,
        .static_facts = {},
        .term_count = 2,
        .trailing_reserved = {},
    };
    projection_terms_[2] = {.coefficient = 1, .argument_ordinal = 2};
    for (uint32_t i = 1; i < 3; ++i) {
      projections_[i] = {
          .source_value_id = i,
          .kind = LOOM_KERNEL_CLASS_PROJECTION_AFFINE,
          .terms = &projection_terms_[i],
          .term_count = 1,
      };
    }
    if (shape_mode == ShapeMode::kSelected ||
        shape_mode == ShapeMode::kDifferent ||
        shape_mode == ShapeMode::kSource) {
      shapes_[0] = {.projections = &shape_projections_[0], .count = 1};
    }
    if (shape_mode == ShapeMode::kGeneric || shape_mode == ShapeMode::kSource) {
      shapes_[1] = {.projections = &shape_projections_[0], .count = 1};
    } else if (shape_mode == ShapeMode::kDifferent) {
      shapes_[1] = {.projections = &shape_projections_[1], .count = 1};
    }
    for (loom_kernel_class_decision_t& decision : decisions_) {
      decision = {
          .application = nullptr,
          .model = &model_,
          .argument_values = projection_value_ids_,
          .result_values = nullptr,
          .projection_ordinals = &projection_ordinal_,
          .feature_outcomes = nullptr,
          .actions = actions_,
          .action_shapes = shapes_,
          .residual_shape =
              {
                  .projections = shape_projections_,
                  .count = shape_mode == ShapeMode::kNone        ? 0u
                           : shape_mode == ShapeMode::kDifferent ? 2u
                                                                 : 1u,
              },
          .generic_result =
              {
                  .kind = LOOM_DECISION_PROGRAM_RESULT_SELECTED,
                  .reserved = {},
                  .action_ordinal = 1,
                  .unresolved_action_ordinal =
                      LOOM_DECISION_PROGRAM_ACTION_INVALID,
                  .unresolved_constraint =
                      LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
              },
          .argument_count = 3,
          .result_count = 0,
          .projection_count = 1,
          .unavailable_reason = LOOM_KERNEL_CLASS_DECISION_AVAILABLE,
          .hard_requirement_flags = 0,
          .can_reduce_structure = shape_mode != ShapeMode::kNone &&
                                  shape_mode != ShapeMode::kSource,
          .reserved = {},
      };
    }
    classifier_ = {
        .module = nullptr,
        .kernel_symbol_id = 0,
        .projections = projections_,
        .decisions = decisions_.data(),
        .shape = shape_mode == ShapeMode::kSource ? shapes_[0]
                                                  : loom_kernel_class_shape_t{},
        .has_shape_requirements = shape_mode != ShapeMode::kNone,
        .projection_count = 3,
        .decision_count = decision_count,
        .maximum_provider_count = 2,
        .kernel_argument_count = kKernelArgumentCount,
        .specialization_count = 0,
    };
  }

  ~KernelClassClassifierBenchmarkFixture() {
    iree_arena_deinitialize(&collection_arena_);
    iree_arena_deinitialize(&facts_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint64_t Collect() {
    const loom_kernel_class_collection_options_t options =
        loom_kernel_class_collection_options_default();
    loom_kernel_class_collection_t collection = {};
    IREE_CHECK_OK(loom_kernel_class_classifier_collect(
        &classifier_, nullptr, sites_.data(), sites_.size(), &options,
        &collection_arena_, &collection));
    constexpr uint32_t kExpectedClassCounts[] = {2, 6, 6, 12, 10};
    IREE_ASSERT_EQ(collection.class_count,
                   kExpectedClassCounts[static_cast<uint32_t>(shape_mode_)]);
    IREE_ASSERT_EQ(collection.accepted_decision_count,
                   classifier_.decision_count);
    const uint64_t sink =
        collection.class_count + collection.classes[0].member_count +
        collection.classes[1].member_count + collection.trace_count;
    arena_used_bytes_ = collection_arena_.used_allocation_size;
    arena_owned_bytes_ = collection_arena_.total_allocation_size;
    iree_arena_reset(&collection_arena_);
    return sink;
  }

  void Warm() {
    Collect();
    iree_arena_block_pool_query_statistics(&block_pool_, &warm_statistics_);
  }

  void ReportMemory(benchmark::State& state) {
    iree_arena_block_pool_statistics_t statistics;
    iree_arena_block_pool_query_statistics(&block_pool_, &statistics);
    state.counters["arena_used_bytes"] = arena_used_bytes_;
    state.counters["arena_owned_bytes"] = arena_owned_bytes_;
    state.counters["pool_system_bytes"] =
        statistics.block_system_allocation_bytes;
    state.counters["pool_steady_allocations"] =
        statistics.block_system_allocation_count -
        warm_statistics_.block_system_allocation_count;
    state.counters["oversized_allocations_per_iteration"] =
        double(statistics.oversized_allocation_count -
               warm_statistics_.oversized_allocation_count) /
        state.iterations();
    state.counters["oversized_bytes_per_iteration"] =
        double(statistics.oversized_allocation_bytes -
               warm_statistics_.oversized_allocation_bytes) /
        state.iterations();
  }

 private:
  // Structural requirements participating in the measured class partition.
  ShapeMode shape_mode_;
  // Collection arena payload after the most recent collection.
  iree_host_size_t arena_used_bytes_ = 0;
  // Collection arena backing after the most recent collection.
  iree_host_size_t arena_owned_bytes_ = 0;
  // Pool counters after one warm collection, excluding cold allocation costs.
  iree_arena_block_pool_statistics_t warm_statistics_ = {};
  static constexpr uint16_t kKernelArgumentCount = 3;

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t facts_arena_ = {};
  iree_arena_allocator_t collection_arena_ = {};
  std::vector<loom_value_id_t> argument_values_;
  std::vector<loom_kernel_class_site_t> sites_;
  loom_value_fact_table_t fact_table_ = {};

  loom_decision_program_predicate_t predicate_ = {};
  loom_decision_program_choice_t choices_[2] = {};
  loom_decision_program_priority_group_t priority_groups_[2] = {};
  int64_t constants_[1] = {128};
  loom_template_decision_model_t model_ = {};

  loom_kernel_class_projection_term_t projection_terms_[3] = {};
  // Header expression followed by the two independent structural dimensions.
  loom_kernel_class_projection_t projections_[3] = {};
  // Projection ordinals referenced only by provider structure.
  uint32_t shape_projections_[2] = {1, 2};
  // Structural requirements for the selected and generic providers.
  loom_kernel_class_shape_t shapes_[2] = {};
  // Provider formals mapped to the corresponding boundary projections.
  loom_value_id_t projection_value_ids_[3] = {0, 1, 2};
  uint32_t projection_ordinal_ = 0;
  loom_kernel_class_action_t actions_[2] = {};
  std::vector<loom_kernel_class_decision_t> decisions_;
  loom_kernel_class_classifier_t classifier_ = {};
};

static void BM_CollectBoundaryClasses(benchmark::State& state) {
  const iree_host_size_t site_count =
      static_cast<iree_host_size_t>(state.range(0));
  const uint32_t decision_count = static_cast<uint32_t>(state.range(1));
  const auto shape_mode = static_cast<ShapeMode>(state.range(2));
  KernelClassClassifierBenchmarkFixture fixture(site_count, decision_count,
                                                shape_mode);
  fixture.Warm();
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Collect());
  }
  fixture.ReportMemory(state);
  state.SetItemsProcessed(state.iterations() * site_count);
}
BENCHMARK(BM_CollectBoundaryClasses)
    ->ArgsProduct({{1000, 10000}, {1, 16, 128}, {0, 1, 2, 3, 4}})
    ->ArgNames({"sites", "decisions", "shape"});

}  // namespace
}  // namespace loom
