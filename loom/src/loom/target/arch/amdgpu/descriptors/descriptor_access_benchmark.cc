// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks production descriptor-table access over hot and family-sized
// dynamic ordinal traces. The rows isolate structural, scheduling, and
// assembly projections while the mixed row models consumers that need all
// three. An ordinal-only row measures trace and loop overhead.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"

namespace {

constexpr iree_host_size_t kTraceLength = 16 * 1024;
constexpr uint32_t kFullDescriptorSet = 0;
constexpr uint32_t kHotDescriptorCount = 64;

class DescriptorAccessFixture {
 public:
  DescriptorAccessFixture(const char* descriptor_set_key,
                          uint32_t working_descriptor_count) {
    loom_target_low_descriptor_registry_t registry;
    loom_amdgpu_low_descriptor_registry_initialize(&registry);
    descriptor_set_ = loom_low_descriptor_registry_lookup(
        &registry.registry, iree_make_cstring_view(descriptor_set_key));
    if (descriptor_set_ == nullptr || descriptor_set_->descriptor_count == 0) {
      std::abort();
    }

    working_descriptor_count_ =
        working_descriptor_count == kFullDescriptorSet
            ? descriptor_set_->descriptor_count
            : std::min(working_descriptor_count,
                       descriptor_set_->descriptor_count);
    ordinals_.reserve(kTraceLength);
    uint32_t random_state = 0x9e3779b9u;
    for (iree_host_size_t i = 0; i < kTraceLength; ++i) {
      random_state = random_state * 1664525u + 1013904223u;
      ordinals_.push_back((random_state >> 8) % working_descriptor_count_);
    }
  }

  const loom_low_descriptor_set_t* descriptor_set() const {
    return descriptor_set_;
  }

  const std::vector<uint32_t>& ordinals() const { return ordinals_; }

  void SetCounters(benchmark::State& state) const {
    state.counters["descriptor_count"] =
        static_cast<double>(descriptor_set_->descriptor_count);
    state.counters["descriptor_row_bytes"] =
        static_cast<double>(sizeof(loom_low_descriptor_t));
    state.counters["descriptor_view_row_bytes"] =
        static_cast<double>(sizeof(loom_low_descriptor_view_t));
    state.counters["working_descriptors"] =
        static_cast<double>(working_descriptor_count_);
    state.SetItemsProcessed(state.iterations() * ordinals_.size());
  }

 private:
  const loom_low_descriptor_set_t* descriptor_set_ = nullptr;
  uint32_t working_descriptor_count_ = 0;
  std::vector<uint32_t> ordinals_;
};

template <typename Accessor>
static void RunAccessBenchmark(benchmark::State& state,
                               const char* descriptor_set_key,
                               uint32_t working_descriptor_count,
                               Accessor accessor) {
  DescriptorAccessFixture fixture(descriptor_set_key, working_descriptor_count);
  const loom_low_descriptor_set_t* descriptor_set = fixture.descriptor_set();
  const loom_low_schedule_model_t* schedule_model =
      loom_low_descriptor_set_lookup_schedule_model_by_id(
          descriptor_set, UINT64_C(0x2a42ccc9fb04b34e));
  IREE_ASSERT(schedule_model != NULL);
  const std::vector<uint32_t>& ordinals = fixture.ordinals();
  benchmark::DoNotOptimize(descriptor_set);
  benchmark::DoNotOptimize(schedule_model);
  for (auto _ : state) {
    uint64_t sums[4] = {};
    for (iree_host_size_t i = 0; i < ordinals.size(); i += 4) {
      sums[0] += accessor(descriptor_set, schedule_model, ordinals[i + 0]);
      sums[1] += accessor(descriptor_set, schedule_model, ordinals[i + 1]);
      sums[2] += accessor(descriptor_set, schedule_model, ordinals[i + 2]);
      sums[3] += accessor(descriptor_set, schedule_model, ordinals[i + 3]);
    }
    benchmark::DoNotOptimize(sums);
  }
  fixture.SetCounters(state);
}

static void BM_OrdinalTrace(benchmark::State& state,
                            const char* descriptor_set_key,
                            uint32_t working_descriptor_count) {
  RunAccessBenchmark(
      state, descriptor_set_key, working_descriptor_count,
      [](const loom_low_descriptor_set_t*, const loom_low_schedule_model_t*,
         uint32_t descriptor_ordinal) {
        return static_cast<uint64_t>(descriptor_ordinal);
      });
}

static void BM_StructuralFacts(benchmark::State& state,
                               const char* descriptor_set_key,
                               uint32_t working_descriptor_count) {
  RunAccessBenchmark(
      state, descriptor_set_key, working_descriptor_count,
      [](const loom_low_descriptor_set_t* descriptor_set,
         const loom_low_schedule_model_t*, uint32_t descriptor_ordinal) {
        const loom_low_descriptor_t* descriptor =
            &descriptor_set->descriptors[descriptor_ordinal];
        const auto* layout =
            loom_low_descriptor_layout(descriptor_set, descriptor);
        return loom_low_descriptor_stable_id(descriptor_set, descriptor) +
               descriptor->operand_start + layout->immediate_start +
               layout->effect_start + descriptor->flags;
      });
}

static void BM_ScheduleFacts(benchmark::State& state,
                             const char* descriptor_set_key,
                             uint32_t working_descriptor_count) {
  RunAccessBenchmark(
      state, descriptor_set_key, working_descriptor_count,
      [](const loom_low_descriptor_set_t* descriptor_set,
         const loom_low_schedule_model_t* schedule_model,
         uint32_t descriptor_ordinal) {
        const loom_low_descriptor_view_t* descriptor_view =
            &descriptor_set->descriptor_views[descriptor_ordinal];
        const loom_low_schedule_class_t* schedule_class =
            loom_low_schedule_model_schedule_class(schedule_model,
                                                   descriptor_view);
        return static_cast<uint64_t>(descriptor_view->instruction_class_flags) +
               schedule_class->latency_cycles +
               schedule_class->schedule_distance_cycles +
               static_cast<uint64_t>(
                   schedule_class->minimum_issue_separation_cycles) +
               schedule_class->flags;
      });
}

static void BM_AssemblyFacts(benchmark::State& state,
                             const char* descriptor_set_key,
                             uint32_t working_descriptor_count) {
  RunAccessBenchmark(
      state, descriptor_set_key, working_descriptor_count,
      [](const loom_low_descriptor_set_t* descriptor_set,
         const loom_low_schedule_model_t*, uint32_t descriptor_ordinal) {
        const loom_low_descriptor_view_t* descriptor_view =
            &descriptor_set->descriptor_views[descriptor_ordinal];
        const uint32_t asm_form_ordinal =
            descriptor_view->canonical_asm_form_ordinal;
        if (asm_form_ordinal == LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
          return UINT64_C(0);
        }
        const loom_low_asm_form_t* asm_form =
            &descriptor_set->asm_forms[asm_form_ordinal];
        const loom_low_asm_layout_t* layout =
            &descriptor_set->asm_layouts[asm_form->layout_index];
        return static_cast<uint64_t>(asm_form_ordinal) +
               asm_form->descriptor_ordinal + layout->tied_result_count +
               layout->explicit_immediate_count;
      });
}

static void BM_MixedFacts(benchmark::State& state,
                          const char* descriptor_set_key,
                          uint32_t working_descriptor_count) {
  RunAccessBenchmark(
      state, descriptor_set_key, working_descriptor_count,
      [](const loom_low_descriptor_set_t* descriptor_set,
         const loom_low_schedule_model_t* schedule_model,
         uint32_t descriptor_ordinal) {
        const loom_low_descriptor_t* descriptor =
            &descriptor_set->descriptors[descriptor_ordinal];
        const loom_low_descriptor_view_t* descriptor_view =
            &descriptor_set->descriptor_views[descriptor_ordinal];
        const loom_low_schedule_class_t* schedule_class =
            loom_low_schedule_model_schedule_class(schedule_model,
                                                   descriptor_view);
        uint64_t result =
            loom_low_descriptor_stable_id(descriptor_set, descriptor) +
            descriptor->operand_start +
            loom_low_descriptor_layout(descriptor_set, descriptor)
                ->effect_start +
            descriptor_view->instruction_class_flags +
            schedule_class->latency_cycles + schedule_class->flags;
        const uint32_t asm_form_ordinal =
            descriptor_view->canonical_asm_form_ordinal;
        if (asm_form_ordinal != LOOM_LOW_ASM_FORM_ORDINAL_NONE) {
          const loom_low_asm_form_t* asm_form =
              &descriptor_set->asm_forms[asm_form_ordinal];
          const loom_low_asm_layout_t* layout =
              &descriptor_set->asm_layouts[asm_form->layout_index];
          result += asm_form_ordinal + asm_form->descriptor_ordinal +
                    layout->tied_result_count;
        }
        return result;
      });
}

constexpr char kGfx1250A0DescriptorSet[] = "amdgpu.rdna4.gfx1250_a0.core";
constexpr char kGfx12_5GenericDescriptorSet[] = "amdgpu.gfx12_5.generic.core";

#define LOOM_BENCHMARK_ACCESS_SHAPES(function)                            \
  BENCHMARK_CAPTURE(function, gfx1250_a0_hot, kGfx1250A0DescriptorSet,    \
                    kHotDescriptorCount)                                  \
      ->Unit(benchmark::kNanosecond);                                     \
  BENCHMARK_CAPTURE(function, gfx1250_a0_family, kGfx1250A0DescriptorSet, \
                    kFullDescriptorSet)                                   \
      ->Unit(benchmark::kNanosecond);                                     \
  BENCHMARK_CAPTURE(function, gfx12_5_generic_family,                     \
                    kGfx12_5GenericDescriptorSet, kFullDescriptorSet)     \
      ->Unit(benchmark::kNanosecond)

LOOM_BENCHMARK_ACCESS_SHAPES(BM_OrdinalTrace);
LOOM_BENCHMARK_ACCESS_SHAPES(BM_StructuralFacts);
LOOM_BENCHMARK_ACCESS_SHAPES(BM_ScheduleFacts);
LOOM_BENCHMARK_ACCESS_SHAPES(BM_AssemblyFacts);
LOOM_BENCHMARK_ACCESS_SHAPES(BM_MixedFacts);

#undef LOOM_BENCHMARK_ACCESS_SHAPES

}  // namespace
