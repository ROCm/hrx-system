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

class KernelClassClassifierBenchmarkFixture {
 public:
  KernelClassClassifierBenchmarkFixture(iree_host_size_t site_count,
                                        uint32_t decision_count)
      : fact_entries_(site_count * kKernelArgumentCount),
        argument_values_(site_count * kKernelArgumentCount),
        sites_(site_count),
        decisions_(decision_count) {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &collection_arena_);

    fact_table_.entries = fact_entries_.data();
    fact_table_.count = fact_entries_.size();
    fact_table_.capacity = fact_entries_.size();
    for (iree_host_size_t i = 0; i < site_count; ++i) {
      const iree_host_size_t first_argument = i * kKernelArgumentCount;
      argument_values_[first_argument + 0] =
          static_cast<loom_value_id_t>(first_argument + 0);
      argument_values_[first_argument + 1] =
          static_cast<loom_value_id_t>(first_argument + 1);
      argument_values_[first_argument + 2] =
          static_cast<loom_value_id_t>(first_argument + 2);
      fact_entries_[first_argument + 0] =
          loom_value_facts_exact_i64(static_cast<int64_t>(i));
      fact_entries_[first_argument + 1] =
          loom_value_facts_exact_i64(static_cast<int64_t>((i * 7) % 5));
      fact_entries_[first_argument + 2] =
          loom_value_facts_exact_i64(static_cast<int64_t>(1000003 * i + 17));
      sites_[i] = {
          /*.facts=*/&fact_table_,
          /*.argument_values=*/&argument_values_[first_argument],
      };
    }

    predicate_ = {
        /*.kind=*/LOOM_PREDICATE_GE,
        /*.operand_count=*/2,
        /*.reserved=*/{},
        /*.operands=*/
        {
            loom_decision_program_argument_ref(0),
            loom_decision_program_constant_ref(0),
        },
    };
    choices_[0] = {
        /*.conjunction=*/
        {
            /*.first_predicate=*/0,
            /*.first_feature=*/0,
            /*.predicate_count=*/1,
            /*.feature_count=*/0,
        },
        /*.action_ordinal=*/0,
    };
    choices_[1] = {
        /*.conjunction=*/{},
        /*.action_ordinal=*/1,
    };
    priority_groups_[0].choice_count = 1;
    priority_groups_[1].choice_count = 1;
    model_.program = {
        /*.predicates=*/&predicate_,
        /*.choices=*/choices_,
        /*.priority_groups=*/priority_groups_,
        /*.constants=*/constants_,
        /*.hard_requirements=*/{},
        /*.predicate_count=*/1,
        /*.feature_count=*/0,
        /*.constant_count=*/1,
        /*.choice_count=*/2,
        /*.priority_group_count=*/2,
    };

    projection_terms_[0] = {
        /*.coefficient=*/1,
        /*.argument_ordinal=*/0,
        /*.reserved=*/{},
    };
    projection_terms_[1] = {
        /*.coefficient=*/1,
        /*.argument_ordinal=*/1,
        /*.reserved=*/{},
    };
    projection_ = {
        /*.source_value_id=*/0,
        /*.kind=*/LOOM_KERNEL_CLASS_PROJECTION_AFFINE,
        /*.reserved=*/{},
        /*.constant=*/0,
        /*.terms=*/projection_terms_,
        /*.static_facts=*/{},
        /*.term_count=*/2,
        /*.trailing_reserved=*/{},
    };
    for (loom_kernel_class_decision_t& decision : decisions_) {
      decision = {
          /*.demand=*/nullptr,
          /*.model=*/&model_,
          /*.argument_values=*/&projection_value_id_,
          /*.result_values=*/nullptr,
          /*.projection_ordinals=*/&projection_ordinal_,
          /*.feature_outcomes=*/nullptr,
          /*.generic_result=*/
          {
              /*.kind=*/LOOM_DECISION_PROGRAM_RESULT_SELECTED,
              /*.reserved=*/{},
              /*.action_ordinal=*/1,
              /*.unresolved_action_ordinal=*/
              LOOM_DECISION_PROGRAM_ACTION_INVALID,
              /*.unresolved_constraint=*/
              LOOM_DECISION_PROGRAM_CONSTRAINT_INVALID,
          },
          /*.argument_count=*/1,
          /*.result_count=*/0,
          /*.projection_count=*/1,
          /*.unavailable_reason=*/LOOM_KERNEL_CLASS_DECISION_AVAILABLE,
          /*.reserved=*/{},
      };
    }
    classifier_ = {
        /*.module=*/nullptr,
        /*.kernel_symbol_id=*/0,
        /*.projections=*/&projection_,
        /*.decisions=*/decisions_.data(),
        /*.projection_count=*/1,
        /*.decision_count=*/decision_count,
        /*.maximum_provider_count=*/2,
        /*.kernel_argument_count=*/kKernelArgumentCount,
        /*.reserved=*/{},
    };
  }

  ~KernelClassClassifierBenchmarkFixture() {
    iree_arena_deinitialize(&collection_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint64_t Collect() {
    const loom_kernel_class_collection_options_t options =
        loom_kernel_class_collection_options_default();
    loom_kernel_class_collection_t collection = {};
    IREE_CHECK_OK(loom_kernel_class_classifier_collect(
        &classifier_, sites_.data(), sites_.size(), &options,
        &collection_arena_, &collection));
    IREE_ASSERT_EQ(collection.class_count, 2u);
    IREE_ASSERT_EQ(collection.accepted_decision_count,
                   classifier_.decision_count);
    const uint64_t sink =
        collection.class_count + collection.classes[0].member_count +
        collection.classes[1].member_count + collection.trace_count;
    iree_arena_reset(&collection_arena_);
    return sink;
  }

 private:
  static constexpr uint16_t kKernelArgumentCount = 3;

  iree_arena_block_pool_t block_pool_ = {};
  iree_arena_allocator_t collection_arena_ = {};
  std::vector<loom_value_facts_t> fact_entries_;
  std::vector<loom_value_id_t> argument_values_;
  std::vector<loom_kernel_class_site_t> sites_;
  loom_value_fact_table_t fact_table_ = {};

  loom_decision_program_predicate_t predicate_ = {};
  loom_decision_program_choice_t choices_[2] = {};
  loom_decision_program_priority_group_t priority_groups_[2] = {};
  int64_t constants_[1] = {128};
  loom_template_decision_model_t model_ = {};

  loom_kernel_class_projection_term_t projection_terms_[2] = {};
  loom_kernel_class_projection_t projection_ = {};
  loom_value_id_t projection_value_id_ = 0;
  uint32_t projection_ordinal_ = 0;
  std::vector<loom_kernel_class_decision_t> decisions_;
  loom_kernel_class_classifier_t classifier_ = {};
};

static void BM_CollectBoundaryClasses(benchmark::State& state) {
  const iree_host_size_t site_count =
      static_cast<iree_host_size_t>(state.range(0));
  const uint32_t decision_count = static_cast<uint32_t>(state.range(1));
  KernelClassClassifierBenchmarkFixture fixture(site_count, decision_count);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fixture.Collect());
  }
  state.SetItemsProcessed(state.iterations() * site_count);
}
BENCHMARK(BM_CollectBoundaryClasses)
    ->Args({1000, 1})
    ->Args({1000, 16})
    ->Args({10000, 1})
    ->Args({10000, 16});

}  // namespace
}  // namespace loom
