// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/work_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/tools/loom-benchmark/module_query.h"
#include "loom/util/adaptive_sort.h"

static iree_status_t loom_benchmark_duration_ms_to_ns(
    int64_t duration_ms, iree_duration_t* out_duration_ns) {
  if (duration_ms < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duration must be non-negative; got %" PRIi64,
                            duration_ms);
  }
  if (duration_ms > INT64_MAX / 1000000) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "duration %" PRIi64 "ms overflows nanoseconds",
                            duration_ms);
  }
  *out_duration_ns = (iree_duration_t)(duration_ms * 1000000);
  return iree_ok_status();
}

static iree_status_t loom_benchmark_validate_dispatch_sequence(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan) {
  const loom_testbench_case_plan_t* case_plan =
      &module_plan->cases[benchmark_plan->case_index];
  for (iree_host_size_t i = 1; i < case_plan->kernel_launch_count; ++i) {
    if (case_plan->kernel_launches[i] !=
        case_plan->kernel_launches[i - 1] + 1) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch_complete benchmark `%.*s` selects check.case `%.*s`, "
          "whose kernel launches are separated by a non-kernel invocation; "
          "kernel launches must form one contiguous device sequence",
          (int)benchmark_plan->name.size, benchmark_plan->name.data,
          (int)case_plan->name.size, case_plan->name.data);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_benchmark_policy_from_benchmark(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_options_t* options,
    loom_benchmark_benchmark_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);

  const bool is_scenario =
      benchmark_plan->scenario_index < module_plan->scenario_count;
  iree_string_view_t measure = options->measure;
  loom_benchmark_measure_t measure_kind =
      LOOM_BENCHMARK_MEASURE_CASE_END_TO_END;
  if (iree_string_view_equal(measure, IREE_SV("auto"))) {
    measure =
        is_scenario ? IREE_SV("dispatch_complete") : IREE_SV("case_end_to_end");
    measure_kind = is_scenario ? LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE
                               : LOOM_BENCHMARK_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("case_end_to_end")) ||
             iree_string_view_equal(measure, IREE_SV("end_to_end"))) {
    measure = IREE_SV("case_end_to_end");
    measure_kind = LOOM_BENCHMARK_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("dispatch_complete"))) {
    measure_kind = LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "benchmark `%.*s` requested unsupported V0 "
                            "measure '%.*s'",
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, (int)measure.size,
                            measure.data);
  }
  if (is_scenario && measure_kind != LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "scenario benchmark `%.*s` requires measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }
  if (!is_scenario &&
      measure_kind == LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE) {
    IREE_RETURN_IF_ERROR(
        loom_benchmark_validate_dispatch_sequence(module_plan, benchmark_plan));
  }
  if (measure_kind != LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE &&
      (!iree_string_view_is_empty(options->profile_artifacts_dir) ||
       options->profile_data_requested ||
       options->profile_counters.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "final-batch profile flags require benchmark `%.*s` "
        "to use measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }

  *out_policy = (loom_benchmark_benchmark_policy_t){
      .measure_kind = measure_kind,
      .measure = measure,
      .warmup_iterations = options->warmup_iterations,
      .iterations = options->iterations,
  };
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);
  if (measure_kind == LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE) {
    bool profile_final_batch = options->profile_final_batch;
    const bool profile_artifacts_requested =
        !iree_string_view_is_empty(options->profile_artifacts_dir);
    if ((profile_artifacts_requested || options->profile_data_requested ||
         options->profile_counters.count != 0) &&
        !profile_final_batch) {
      if (options->profile_final_batch_specified &&
          !options->profile_final_batch) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--profile-final-batch=false conflicts with "
                                "requested final-batch profile data");
      }
      profile_final_batch = true;
    }
    iree_duration_t min_duration_ns = 0;
    iree_duration_t warmup_min_duration_ns = 0;
    IREE_RETURN_IF_ERROR(loom_benchmark_duration_ms_to_ns(options->min_time_ms,
                                                          &min_duration_ns));
    IREE_RETURN_IF_ERROR(loom_benchmark_duration_ms_to_ns(
        options->warmup_time_ms, &warmup_min_duration_ns));
    out_policy->hal_options.timing.batch_size = options->batch_size;
    out_policy->hal_options.timing.warmup_batch_count =
        options->warmup_iterations;
    out_policy->hal_options.timing.warmup_min_duration_ns =
        warmup_min_duration_ns;
    out_policy->hal_options.timing.min_batch_count = options->iterations;
    out_policy->hal_options.timing.min_duration_ns = min_duration_ns;
    out_policy->hal_options.timing.max_batch_count = options->max_batches;
    out_policy->hal_options.timing.stable_p90_to_p50_delta_ppm =
        options->stable_p90_to_p50_ppm;
    out_policy->hal_options.dispatch_batch.dispatch_count = options->batch_size;
    IREE_RETURN_IF_ERROR(loom_benchmark_parse_profile_data_families(
        options->profile_data, &out_policy->hal_options.profile_data_families));
    const iree_hal_device_profiling_data_families_t counter_data =
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_SAMPLES |
        IREE_HAL_DEVICE_PROFILING_DATA_COUNTER_RANGES;
    if (options->profile_counters.count != 0) {
      if (!iree_any_bit_set(out_policy->hal_options.profile_data_families,
                            counter_data)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--profile-counter requires --profile-data to include counters or "
            "counter-ranges");
      }
      out_policy->profile_counter_set =
          (iree_hal_profile_counter_set_selection_t){
              .name = IREE_SV("loom-benchmark"),
              .counter_name_count = options->profile_counters.count,
              .counter_names = options->profile_counters.values,
          };
      out_policy->hal_options.profile_counter_set_count = 1;
      out_policy->hal_options.profile_counter_sets =
          &out_policy->profile_counter_set;
    }
    if (profile_final_batch) {
      out_policy->hal_options.flags |=
          LOOM_RUN_HAL_BENCHMARK_FLAG_PROFILE_FINAL_BATCH;
    }
  }
  return iree_ok_status();
}

static iree_host_size_t loom_benchmark_compare_token_count(
    iree_string_view_t compare_list) {
  iree_host_size_t count = 0;
  iree_string_view_t remaining = iree_string_view_trim(compare_list);
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      ++count;
    }
    remaining = iree_string_view_trim(remaining);
  }
  return count;
}

static bool loom_benchmark_benchmark_name_less(
    const loom_testbench_module_plan_t* module_plan,
    const iree_host_size_t* lhs, const iree_host_size_t* rhs) {
  const int order = iree_string_view_compare(
      module_plan->benchmarks[*lhs].name, module_plan->benchmarks[*rhs].name);
  return order != 0 ? order < 0 : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_benchmark_sort_benchmark_names,
                                       iree_host_size_t,
                                       const loom_testbench_module_plan_t*,
                                       loom_benchmark_benchmark_name_less)

static iree_status_t loom_benchmark_find_benchmark_by_name(
    const loom_testbench_module_plan_t* module_plan, iree_string_view_t name,
    const iree_host_size_t* sorted_indices,
    iree_host_size_t* out_benchmark_index,
    const loom_testbench_benchmark_plan_t** out_benchmark) {
  *out_benchmark_index = IREE_HOST_SIZE_MAX;
  *out_benchmark = NULL;
  if (sorted_indices != NULL) {
    iree_host_size_t begin = 0;
    iree_host_size_t end = module_plan->benchmark_count;
    while (begin < end) {
      const iree_host_size_t middle = begin + (end - begin) / 2;
      if (iree_string_view_compare(
              module_plan->benchmarks[sorted_indices[middle]].name, name) < 0) {
        begin = middle + 1;
      } else {
        end = middle;
      }
    }
    if (begin < module_plan->benchmark_count &&
        iree_string_view_equal(
            module_plan->benchmarks[sorted_indices[begin]].name, name)) {
      *out_benchmark_index = sorted_indices[begin];
    }
  } else {
    for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
      if (iree_string_view_equal(module_plan->benchmarks[i].name, name)) {
        *out_benchmark_index = i;
        break;
      }
    }
  }
  if (*out_benchmark_index != IREE_HOST_SIZE_MAX) {
    *out_benchmark = &module_plan->benchmarks[*out_benchmark_index];
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "--compare benchmark '@%.*s' was not found",
                          (int)name.size, name.data);
}

static iree_status_t loom_benchmark_sample_window(
    const loom_benchmark_options_t* options, iree_host_size_t sample_count,
    iree_host_size_t* out_begin_sample, iree_host_size_t* out_end_sample);

static iree_status_t loom_benchmark_selected_benchmark_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const loom_benchmark_options_t* options, iree_host_size_t benchmark_index,
    iree_host_size_t candidate_index,
    loom_benchmark_selected_benchmark_t* out_selection) {
  const bool is_case = benchmark_plan->case_index < module_plan->case_count;
  const bool is_scenario =
      benchmark_plan->scenario_index < module_plan->scenario_count;
  if (benchmark_plan->sample_count == 0) {
    const iree_string_view_t record_name =
        is_case ? module_plan->cases[benchmark_plan->case_index].name
                : module_plan->scenarios[benchmark_plan->scenario_index].name;
    const char* record_kind = is_case ? "case" : "scenario";
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected benchmark `%.*s` for check.%s `%.*s` has zero executable "
        "samples; selected_benchmark='%.*s', selected_case='%.*s', "
        "sample=%d",
        (int)benchmark_plan->name.size, benchmark_plan->name.data, record_kind,
        (int)record_name.size, record_name.data,
        (int)options->selected_benchmark.size, options->selected_benchmark.data,
        (int)options->selected_case.size, options->selected_case.data,
        options->sample_ordinal);
  }
  memset(out_selection, 0, sizeof(*out_selection));
  snprintf(out_selection->candidate_id_storage,
           sizeof(out_selection->candidate_id_storage), "c%" PRIhsz,
           candidate_index);
  out_selection->identity = (loom_benchmark_candidate_identity_t){
      .candidate_id =
          iree_make_cstring_view(out_selection->candidate_id_storage),
      .candidate_index = candidate_index,
  };
  out_selection->benchmark_plan = benchmark_plan;
  out_selection->benchmark_index = benchmark_index;
  out_selection->case_plan =
      is_case ? &module_plan->cases[benchmark_plan->case_index] : NULL;
  out_selection->scenario_plan =
      is_scenario ? &module_plan->scenarios[benchmark_plan->scenario_index]
                  : NULL;
  IREE_RETURN_IF_ERROR(loom_benchmark_policy_from_benchmark(
      module_plan, benchmark_plan, options, &out_selection->policy));
  return loom_benchmark_sample_window(options, benchmark_plan->sample_count,
                                      &out_selection->begin_benchmark_sample,
                                      &out_selection->end_benchmark_sample);
}

static iree_status_t loom_benchmark_sample_window(
    const loom_benchmark_options_t* options, iree_host_size_t sample_count,
    iree_host_size_t* out_begin_sample, iree_host_size_t* out_end_sample) {
  if (options->sample_ordinal < 0) {
    *out_begin_sample = 0;
    *out_end_sample = sample_count;
    return iree_ok_status();
  }
  const iree_host_size_t sample_ordinal =
      (iree_host_size_t)options->sample_ordinal;
  if (sample_ordinal >= sample_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "--sample=%" PRIhsz
                            " exceeds selected benchmark sample count %" PRIhsz,
                            sample_ordinal, sample_count);
  }
  *out_begin_sample = sample_ordinal;
  *out_end_sample = sample_ordinal + 1;
  return iree_ok_status();
}

static iree_host_size_t loom_benchmark_find_or_append_compile_item(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t* case_compile_indices) {
  const loom_benchmark_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  // Zero means unseen; populated entries are compile indices plus one.
  iree_host_size_t* entry =
      &case_compile_indices[selection->benchmark_plan->case_index];
  if (*entry != 0) {
    return *entry - 1;
  }
  const iree_host_size_t compile_item_index = plan->hal_compile_item_count++;
  *entry = compile_item_index + 1;
  plan->hal_compile_items[compile_item_index] =
      (loom_benchmark_hal_compile_item_t){
          .compile_item_index = compile_item_index,
          .representative_selection_index = selection_index,
      };
  return compile_item_index;
}

static int loom_benchmark_compare_ordinal(iree_host_size_t lhs,
                                          iree_host_size_t rhs) {
  return (lhs > rhs) - (lhs < rhs);
}

// Multi-sample windows are prefixes: --sample selects exactly one sample.
// A parameter's prefix is identified by its initial value and its increment
// period. Pinned parameters and dimensions that never advance within the
// prefix are constant. Comparing these keys avoids expanding the Cartesian
// product, including when different assignments describe the same capped
// prefix.
static int loom_benchmark_compare_sample_prefixes(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_benchmark_plan_t* lhs,
    const loom_testbench_benchmark_plan_t* rhs, iree_host_size_t sample_count) {
  iree_host_size_t strides[2] = {1, 1};
  const loom_testbench_benchmark_plan_t* benchmarks[2] = {lhs, rhs};
  for (iree_host_size_t i = 0; i < case_plan->parameter_count; ++i) {
    const iree_host_size_t radix = case_plan->parameters[i].sample_count;
    iree_host_size_t initial_values[2] = {0, 0};
    iree_host_size_t periods[2] = {0, 0};
    for (int side = 0; side < 2; ++side) {
      const loom_testbench_benchmark_plan_t* benchmark = benchmarks[side];
      const iree_host_size_t ordinal =
          i < benchmark->parameter_sample_ordinal_count
              ? benchmark->parameter_sample_ordinals[i]
              : LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL;
      if (ordinal != LOOM_TESTBENCH_PARAMETER_SAMPLE_ORDINAL_ALL) {
        initial_values[side] = ordinal;
      } else {
        if (radix > 1 && strides[side] < sample_count) {
          periods[side] = strides[side];
        }
        // Strides at or beyond the prefix length all describe a constant
        // dimension. Saturation also avoids multiplying an unbounded domain.
        strides[side] = strides[side] > sample_count / radix
                            ? sample_count
                            : strides[side] * radix;
      }
    }
    int order =
        loom_benchmark_compare_ordinal(initial_values[0], initial_values[1]);
    if (order != 0) {
      return order;
    }
    order = loom_benchmark_compare_ordinal(periods[0], periods[1]);
    if (order != 0) {
      return order;
    }
  }
  return 0;
}

// Policy is fixed by the invocation's options and record kind, so it is not
// part of the key. Physical identity is the record and its measured sample.
typedef struct loom_benchmark_work_item_key_t {
  // Work kind discriminating the record and measured sample domain.
  loom_benchmark_work_item_kind_t kind;
  // Case or scenario index within the module, selected by |kind|.
  iree_host_size_t record_index;
  // Scenario ordinal or saturated Cartesian case ordinal. Equal saturated case
  // keys require coordinate comparison. Retained only during deduplication.
  iree_host_size_t sample_key;
} loom_benchmark_work_item_key_t;

typedef struct loom_benchmark_work_item_index_t {
  // Plan retaining selections for exact comparison of windows and large keys.
  const loom_benchmark_work_plan_t* plan;
  // Cached keys in original work-item order, alive only during deduplication.
  const loom_benchmark_work_item_key_t* keys;
} loom_benchmark_work_item_index_t;

static IREE_ATTRIBUTE_NOINLINE int loom_benchmark_compare_sample_details(
    const loom_benchmark_work_item_index_t* index, iree_host_size_t lhs_index,
    iree_host_size_t rhs_index) {
  const loom_benchmark_work_plan_t* plan = index->plan;
  const loom_benchmark_work_item_key_t* lhs_key = &index->keys[lhs_index];
  const loom_benchmark_work_item_key_t* rhs_key = &index->keys[rhs_index];
  int order = 0;
  const loom_benchmark_work_item_t* lhs = &plan->work_items[lhs_index];
  const loom_benchmark_work_item_t* rhs = &plan->work_items[rhs_index];
  const loom_benchmark_selected_benchmark_t* lhs_selection =
      &plan->selected_benchmarks[lhs->representative_selection_index];
  const loom_benchmark_selected_benchmark_t* rhs_selection =
      &plan->selected_benchmarks[rhs->representative_selection_index];
  const loom_testbench_benchmark_plan_t* lhs_benchmark =
      lhs_selection->benchmark_plan;
  const loom_testbench_benchmark_plan_t* rhs_benchmark =
      rhs_selection->benchmark_plan;
  if (lhs->kind == LOOM_BENCHMARK_WORK_ITEM_CASE_END_TO_END) {
    order = loom_benchmark_compare_ordinal(lhs->begin_benchmark_sample,
                                           rhs->begin_benchmark_sample);
    if (order != 0) {
      return order;
    }
    order = loom_benchmark_compare_ordinal(lhs->end_benchmark_sample,
                                           rhs->end_benchmark_sample);
    if (order != 0) {
      return order;
    }
    if (lhs->end_benchmark_sample - lhs->begin_benchmark_sample != 1) {
      return loom_benchmark_compare_sample_prefixes(
          lhs_selection->case_plan, lhs_benchmark, rhs_benchmark,
          lhs->end_benchmark_sample);
    }
  }
  order =
      loom_benchmark_compare_ordinal(lhs_key->sample_key, rhs_key->sample_key);
  if (order != 0 || lhs_key->sample_key != IREE_HOST_SIZE_MAX) {
    return order;
  }
  return loom_testbench_case_sample_compare(
      lhs_selection->case_plan,
      loom_testbench_benchmark_case_sample(lhs_benchmark,
                                           lhs->begin_benchmark_sample),
      loom_testbench_benchmark_case_sample(rhs_benchmark,
                                           rhs->begin_benchmark_sample));
}

static int loom_benchmark_compare_work_items(
    const loom_benchmark_work_item_index_t* index, iree_host_size_t lhs_index,
    iree_host_size_t rhs_index) {
  const loom_benchmark_work_item_key_t* lhs = &index->keys[lhs_index];
  const loom_benchmark_work_item_key_t* rhs = &index->keys[rhs_index];
  int order = loom_benchmark_compare_ordinal(lhs->kind, rhs->kind);
  if (order != 0) {
    return order;
  }
  order = loom_benchmark_compare_ordinal(lhs->record_index, rhs->record_index);
  if (order != 0) {
    return order;
  }
  if (lhs->kind != LOOM_BENCHMARK_WORK_ITEM_CASE_END_TO_END &&
      (lhs->sample_key != IREE_HOST_SIZE_MAX ||
       rhs->sample_key != IREE_HOST_SIZE_MAX)) {
    return loom_benchmark_compare_ordinal(lhs->sample_key, rhs->sample_key);
  }
  return loom_benchmark_compare_sample_details(index, lhs_index, rhs_index);
}

static bool loom_benchmark_work_item_less(
    const loom_benchmark_work_item_index_t* index, const iree_host_size_t* lhs,
    const iree_host_size_t* rhs) {
  const int order = loom_benchmark_compare_work_items(index, *lhs, *rhs);
  return order != 0 ? order < 0 : *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_benchmark_sort_work_items,
                                       iree_host_size_t,
                                       const loom_benchmark_work_item_index_t*,
                                       loom_benchmark_work_item_less)

static void loom_benchmark_deduplicate_work_items(
    loom_benchmark_work_plan_t* plan, iree_host_size_t* order,
    loom_benchmark_work_item_key_t* keys) {
  for (iree_host_size_t i = 0; i < plan->work_item_count; ++i) {
    const loom_benchmark_work_item_t* work_item = &plan->work_items[i];
    const loom_benchmark_selected_benchmark_t* selection =
        &plan->selected_benchmarks[work_item->representative_selection_index];
    order[i] = i;
    keys[i] = (loom_benchmark_work_item_key_t){
        .kind = work_item->kind,
        .record_index = selection->case_plan != NULL
                            ? selection->benchmark_plan->case_index
                            : selection->benchmark_plan->scenario_index,
        .sample_key = work_item->begin_benchmark_sample,
    };
    if (selection->case_plan != NULL &&
        work_item->end_benchmark_sample - work_item->begin_benchmark_sample ==
            1 &&
        !loom_testbench_case_sample_flatten(
            selection->case_plan,
            loom_testbench_benchmark_case_sample(
                selection->benchmark_plan, work_item->begin_benchmark_sample),
            &keys[i].sample_key)) {
      keys[i].sample_key = IREE_HOST_SIZE_MAX;
    }
  }
  const loom_benchmark_work_item_index_t index = {
      .plan = plan,
      .keys = keys,
  };
  loom_benchmark_sort_work_items(&index, order, plan->work_item_count);
  // Equal candidates sort by logical ordinal, so linking each group preserves
  // report order without changing the logical sample array.
  iree_host_size_t representative = 0;
  for (iree_host_size_t i = 0; i < plan->work_item_count; ++i) {
    if (i == 0 || loom_benchmark_compare_work_items(&index, representative,
                                                    order[i]) != 0) {
      representative = order[i];
    } else {
      plan->logical_samples[order[i - 1]].next_alias_index = order[i];
    }
    plan->logical_samples[order[i]].work_item_index = representative;
  }

  // Candidates and logical samples were appended one-to-one in report order.
  // Equal candidates sort by original ordinal, leaving every alias pointing
  // backward to its first owner. Compact in that original order, translating
  // aliases through the already-finalized owner's logical sample.
  plan->work_item_count = 0;
  for (iree_host_size_t i = 0; i < plan->logical_sample_count; ++i) {
    loom_benchmark_logical_sample_t* logical = &plan->logical_samples[i];
    representative = logical->work_item_index;
    if (representative == i) {
      const iree_host_size_t work_item_index = plan->work_item_count++;
      plan->work_items[work_item_index] = plan->work_items[i];
      plan->work_items[work_item_index].work_item_index = work_item_index;
      plan->work_items[work_item_index].first_logical_sample_index = i;
      logical->work_item_index = work_item_index;
    } else {
      logical->work_item_index =
          plan->logical_samples[representative].work_item_index;
    }
  }
}

static iree_host_size_t loom_benchmark_append_work_item(
    loom_benchmark_work_plan_t* plan, loom_benchmark_work_item_kind_t kind,
    iree_host_size_t selection_index, iree_host_size_t begin_sample,
    iree_host_size_t end_sample,
    loom_testbench_scenario_sample_coordinate_t scenario_coordinate,
    iree_host_size_t hal_compile_item_index) {
  const iree_host_size_t work_item_index = plan->work_item_count++;
  plan->work_items[work_item_index] = (loom_benchmark_work_item_t){
      .kind = kind,
      .work_item_index = work_item_index,
      .representative_selection_index = selection_index,
      .hal_compile_item_index = hal_compile_item_index,
      .begin_benchmark_sample = begin_sample,
      .end_benchmark_sample = end_sample,
      .scenario_coordinate = scenario_coordinate,
  };
  return work_item_index;
}

static void loom_benchmark_append_logical_sample(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    loom_testbench_scenario_sample_coordinate_t scenario_coordinate,
    iree_host_size_t work_item_index) {
  plan->logical_samples[plan->logical_sample_count++] =
      (loom_benchmark_logical_sample_t){
          .selection_index = selection_index,
          .begin_benchmark_sample = begin_sample,
          .end_benchmark_sample = end_sample,
          .scenario_coordinate = scenario_coordinate,
          .work_item_index = work_item_index,
          .next_alias_index = LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID,
      };
}

static iree_status_t loom_benchmark_count_logical_samples(
    const loom_benchmark_selected_benchmark_t* selection,
    iree_host_size_t* inout_logical_sample_count) {
  const iree_host_size_t sample_count =
      selection->policy.measure_kind == LOOM_BENCHMARK_MEASURE_CASE_END_TO_END
          ? 1
          : selection->end_benchmark_sample - selection->begin_benchmark_sample;
  if (sample_count > IREE_HOST_SIZE_MAX - *inout_logical_sample_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected benchmark sample count exceeds host "
                            "limits");
  }
  *inout_logical_sample_count += sample_count;
  return iree_ok_status();
}

static void loom_benchmark_append_case_end_to_end_work(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t compile_item_index) {
  const loom_benchmark_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  const iree_host_size_t begin_sample = selection->begin_benchmark_sample;
  const iree_host_size_t end_sample = selection->end_benchmark_sample;
  const iree_host_size_t work_item_index = loom_benchmark_append_work_item(
      plan, LOOM_BENCHMARK_WORK_ITEM_CASE_END_TO_END, selection_index,
      begin_sample, end_sample,
      (loom_testbench_scenario_sample_coordinate_t){0}, compile_item_index);
  loom_benchmark_append_logical_sample(
      plan, selection_index, begin_sample, end_sample,
      (loom_testbench_scenario_sample_coordinate_t){0}, work_item_index);
}

static void loom_benchmark_append_dispatch_sample_work(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t benchmark_sample_ordinal,
    iree_host_size_t compile_item_index) {
  const iree_host_size_t work_item_index = loom_benchmark_append_work_item(
      plan, LOOM_BENCHMARK_WORK_ITEM_DISPATCH_SAMPLE, selection_index,
      benchmark_sample_ordinal, benchmark_sample_ordinal + 1,
      (loom_testbench_scenario_sample_coordinate_t){0}, compile_item_index);
  loom_benchmark_append_logical_sample(
      plan, selection_index, benchmark_sample_ordinal,
      benchmark_sample_ordinal + 1,
      (loom_testbench_scenario_sample_coordinate_t){0}, work_item_index);
}

static void loom_benchmark_append_scenario_trial_work(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t benchmark_sample_ordinal,
    loom_testbench_scenario_sample_coordinate_t coordinate) {
  const iree_host_size_t work_item_index = loom_benchmark_append_work_item(
      plan, LOOM_BENCHMARK_WORK_ITEM_SCENARIO_TRIAL, selection_index,
      benchmark_sample_ordinal, benchmark_sample_ordinal + 1, coordinate,
      LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID);
  loom_benchmark_append_logical_sample(
      plan, selection_index, benchmark_sample_ordinal,
      benchmark_sample_ordinal + 1, coordinate, work_item_index);
}

static void loom_benchmark_append_scenario_work(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index) {
  const loom_benchmark_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  const loom_testbench_scenario_plan_t* scenario = selection->scenario_plan;
  loom_testbench_scenario_sample_coordinate_t coordinate =
      loom_testbench_benchmark_sample_scenario_coordinate(
          scenario, selection->benchmark_plan,
          selection->begin_benchmark_sample);
  for (iree_host_size_t sample_ordinal = selection->begin_benchmark_sample;
       sample_ordinal < selection->end_benchmark_sample; ++sample_ordinal) {
    loom_benchmark_append_scenario_trial_work(plan, selection_index,
                                              sample_ordinal, coordinate);
    ++coordinate.trial_ordinal;
    if (coordinate.trial_ordinal ==
        scenario->trials[coordinate.trial_index].trial_count) {
      coordinate.trial_ordinal = 0;
      if (++coordinate.trial_index == scenario->trial_count) {
        coordinate.trial_index = 0;
        ++coordinate.configuration_ordinal;
      }
    }
  }
}

static void loom_benchmark_append_selection_work(
    loom_benchmark_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t compile_item_index) {
  const loom_benchmark_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  if (selection->policy.measure_kind ==
      LOOM_BENCHMARK_MEASURE_CASE_END_TO_END) {
    loom_benchmark_append_case_end_to_end_work(plan, selection_index,
                                               compile_item_index);
    return;
  }
  if (selection->scenario_plan != NULL) {
    loom_benchmark_append_scenario_work(plan, selection_index);
    return;
  }
  for (iree_host_size_t sample_ordinal = selection->begin_benchmark_sample;
       sample_ordinal < selection->end_benchmark_sample; ++sample_ordinal) {
    loom_benchmark_append_dispatch_sample_work(
        plan, selection_index, sample_ordinal, compile_item_index);
  }
}

static iree_status_t loom_benchmark_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  // Every plan array publishes only its completely initialized prefix.
  return iree_allocator_malloc_array_uninitialized(allocator, count,
                                                   element_size, out_ptr);
}

static iree_status_t loom_benchmark_select_compare_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_options_t* options, loom_benchmark_work_plan_t* plan) {
  iree_string_view_t compare = iree_string_view_trim(options->compare);
  const iree_host_size_t selection_count =
      loom_benchmark_compare_token_count(compare);
  if (selection_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires at least two comma-separated check.benchmark "
        "symbols");
  }
  IREE_RETURN_IF_ERROR(loom_benchmark_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  // Small comparison lists need at most eight linear scans. Larger lists
  // amortize a temporary name index; selections themselves stay in request
  // order and retain stable addresses for their candidate identifiers.
  iree_host_size_t* sorted_indices = NULL;
  if (selection_count > 8) {
    IREE_RETURN_IF_ERROR(loom_benchmark_allocate_array(
        plan->host_allocator, module_plan->benchmark_count,
        sizeof(*sorted_indices), (void**)&sorted_indices));
    for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
      sorted_indices[i] = i;
    }
    loom_benchmark_sort_benchmark_names(module_plan, sorted_indices,
                                        module_plan->benchmark_count);
  }

  iree_status_t status = iree_ok_status();
  iree_host_size_t selection_index = 0;
  iree_string_view_t remaining = compare;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = loom_benchmark_normalize_selection_name(token);
    if (iree_string_view_is_empty(token)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--compare contains an empty benchmark name");
      break;
    }
    iree_host_size_t benchmark_index = IREE_HOST_SIZE_MAX;
    const loom_testbench_benchmark_plan_t* benchmark = NULL;
    status = loom_benchmark_find_benchmark_by_name(
        module_plan, token, sorted_indices, &benchmark_index, &benchmark);
    if (iree_status_is_ok(status)) {
      // Resolve identities before policies or samples: a zero sample count can
      // be the consequence of an authored planning issue, not an empty domain.
      plan->selected_benchmarks[selection_index].benchmark_plan = benchmark;
      plan->selected_benchmarks[selection_index].benchmark_index =
          benchmark_index;
    }
    ++selection_index;
    remaining = iree_string_view_trim(remaining);
  }
  iree_allocator_free(plan->host_allocator, sorted_indices);
  if (iree_status_is_ok(status)) {
    plan->selected_benchmark_count = selection_count;
  }
  return status;
}

static bool loom_benchmark_matches_selection(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark,
    const loom_benchmark_options_t* options) {
  if (!loom_benchmark_benchmark_matches_selection(
          benchmark, options->selected_benchmark)) {
    return false;
  }
  if (iree_string_view_is_empty(options->selected_case)) {
    return true;
  }
  if (benchmark->case_index != LOOM_TESTBENCH_CASE_INDEX_INVALID) {
    return loom_benchmark_case_matches_selection(
        &module_plan->cases[benchmark->case_index], options->selected_case);
  }
  if (benchmark->scenario_index != LOOM_TESTBENCH_SCENARIO_INDEX_INVALID) {
    return loom_benchmark_scenario_matches_selection(
        &module_plan->scenarios[benchmark->scenario_index],
        options->selected_case);
  }
  return false;
}

static iree_status_t loom_benchmark_select_matching_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_options_t* options, loom_benchmark_work_plan_t* plan) {
  if (module_plan->benchmark_count == 0 &&
      iree_string_view_is_empty(options->selected_benchmark) &&
      iree_string_view_is_empty(options->selected_case)) {
    return iree_ok_status();
  }

  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    if (loom_benchmark_matches_selection(module_plan, benchmark_plan,
                                         options)) {
      ++selection_count;
    }
  }
  if (selection_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no check.benchmark matched '%.*s'",
                            (int)options->selected_benchmark.size,
                            options->selected_benchmark.data);
  }
  IREE_RETURN_IF_ERROR(loom_benchmark_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  iree_host_size_t selection_index = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    if (!loom_benchmark_matches_selection(module_plan, benchmark_plan,
                                          options)) {
      continue;
    }
    plan->selected_benchmarks[selection_index].benchmark_plan = benchmark_plan;
    plan->selected_benchmarks[selection_index].benchmark_index = i;
    ++selection_index;
  }
  plan->selected_benchmark_count = selection_count;
  return iree_ok_status();
}

static bool loom_benchmark_planning_issue_is_selected(
    const loom_testbench_issue_t* issue, const bool* benchmarks,
    const bool* cases, const bool* scenarios) {
  // Assignment errors belong to a benchmark, not every alias of its record.
  if (issue->benchmark_index != LOOM_TESTBENCH_BENCHMARK_INDEX_INVALID) {
    return benchmarks[issue->benchmark_index];
  }
  if (issue->case_index != LOOM_TESTBENCH_CASE_INDEX_INVALID) {
    return cases[issue->case_index];
  }
  return scenarios[issue->scenario_index];
}

static iree_status_t loom_benchmark_select_planning_issues(
    const loom_testbench_module_plan_t* module_plan,
    loom_benchmark_work_plan_t* plan) {
  if (module_plan->issue_count == 0 || plan->selected_benchmark_count == 0) {
    return iree_ok_status();
  }
  bool* selected = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      plan->host_allocator,
      module_plan->benchmark_count + module_plan->case_count +
          module_plan->scenario_count,
      sizeof(*selected), (void**)&selected));
  bool* benchmarks = selected;
  bool* cases = benchmarks + module_plan->benchmark_count;
  bool* scenarios = cases + module_plan->case_count;
  for (iree_host_size_t i = 0; i < plan->selected_benchmark_count; ++i) {
    const loom_benchmark_selected_benchmark_t* selection =
        &plan->selected_benchmarks[i];
    const loom_testbench_benchmark_plan_t* benchmark =
        selection->benchmark_plan;
    benchmarks[selection->benchmark_index] = true;
    if (benchmark->case_index != LOOM_TESTBENCH_CASE_INDEX_INVALID) {
      cases[benchmark->case_index] = true;
    }
    if (benchmark->scenario_index != LOOM_TESTBENCH_SCENARIO_INDEX_INVALID) {
      scenarios[benchmark->scenario_index] = true;
    }
  }
  iree_host_size_t issue_count = 0;
  for (iree_host_size_t i = 0; i < module_plan->issue_count; ++i) {
    issue_count += loom_benchmark_planning_issue_is_selected(
        &module_plan->issues[i], benchmarks, cases, scenarios);
  }
  iree_status_t status = loom_benchmark_allocate_array(
      plan->host_allocator, issue_count, sizeof(*plan->planning_issues),
      (void**)&plan->planning_issues);
  if (iree_status_is_ok(status) && issue_count != 0) {
    for (iree_host_size_t i = 0; i < module_plan->issue_count; ++i) {
      if (loom_benchmark_planning_issue_is_selected(
              &module_plan->issues[i], benchmarks, cases, scenarios)) {
        plan->planning_issues[plan->planning_issue_count++] =
            &module_plan->issues[i];
      }
    }
  }
  iree_allocator_free(plan->host_allocator, selected);
  return status;
}

iree_status_t loom_benchmark_work_plan_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_benchmark_options_t* options, iree_allocator_t allocator,
    loom_benchmark_work_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module_plan);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_benchmark_work_plan_t){0};

  loom_benchmark_work_plan_t plan = {
      .host_allocator = allocator,
  };
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(options->compare)) {
    status =
        loom_benchmark_select_compare_benchmarks(module_plan, options, &plan);
  } else {
    status =
        loom_benchmark_select_matching_benchmarks(module_plan, options, &plan);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_select_planning_issues(module_plan, &plan);
  }
  if (iree_status_is_ok(status) && plan.planning_issue_count != 0) {
    // Invalid authored plans expose their diagnostics, never partially
    // initialized executable selections or expanded sample arrays.
    iree_allocator_free(allocator, plan.selected_benchmarks);
    plan.selected_benchmarks = NULL;
    plan.selected_benchmark_count = 0;
    *out_plan = plan;
    return iree_ok_status();
  }

  iree_host_size_t logical_sample_capacity = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.selected_benchmark_count; ++i) {
    loom_benchmark_selected_benchmark_t* selection =
        &plan.selected_benchmarks[i];
    status = loom_benchmark_selected_benchmark_initialize(
        module_plan, selection->benchmark_plan, options,
        selection->benchmark_index, i, selection);
    if (iree_status_is_ok(status) &&
        !iree_string_view_is_empty(options->compare) && i != 0 &&
        (selection->scenario_plan != NULL) !=
            (plan.selected_benchmarks[0].scenario_plan != NULL)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "--compare candidates must all reference check.case or all "
          "reference check.scenario");
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_count_logical_samples(selection,
                                                    &logical_sample_capacity);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_allocate_array(allocator, logical_sample_capacity,
                                           sizeof(*plan.logical_samples),
                                           (void**)&plan.logical_samples);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_allocate_array(
        allocator,
        iree_min(module_plan->case_count, plan.selected_benchmark_count),
        sizeof(*plan.hal_compile_items), (void**)&plan.hal_compile_items);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_allocate_array(allocator, logical_sample_capacity,
                                           sizeof(*plan.work_items),
                                           (void**)&plan.work_items);
  }
  iree_host_size_t* case_compile_indices = NULL;
  iree_host_size_t* work_item_order = NULL;
  if (iree_status_is_ok(status) && plan.selected_benchmark_count != 0 &&
      module_plan->case_count != 0) {
    status = iree_allocator_malloc_array(allocator, module_plan->case_count,
                                         sizeof(*case_compile_indices),
                                         (void**)&case_compile_indices);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_allocate_array(
        allocator, logical_sample_capacity,
        sizeof(*work_item_order) + sizeof(loom_benchmark_work_item_key_t),
        (void**)&work_item_order);
  }
  if (iree_status_is_ok(status) && logical_sample_capacity != 0) {
    for (iree_host_size_t i = 0; i < plan.selected_benchmark_count; ++i) {
      const loom_benchmark_selected_benchmark_t* selection =
          &plan.selected_benchmarks[i];
      iree_host_size_t compile_item_index =
          LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID;
      if (selection->case_plan != NULL &&
          (selection->policy.measure_kind ==
               LOOM_BENCHMARK_MEASURE_DISPATCH_COMPLETE ||
           selection->case_plan->kernel_launch_count != 0)) {
        compile_item_index = loom_benchmark_find_or_append_compile_item(
            &plan, i, case_compile_indices);
      }
      loom_benchmark_append_selection_work(&plan, i, compile_item_index);
    }
    // The single scratch allocation holds sort indices followed by aligned
    // keys. Sorting moves indices, not the larger key records.
    loom_benchmark_deduplicate_work_items(
        &plan, work_item_order,
        (loom_benchmark_work_item_key_t*)(work_item_order +
                                          logical_sample_capacity));
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.work_item_count; ++i) {
    loom_benchmark_work_item_t* work_item = &plan.work_items[i];
    if (work_item->hal_compile_item_index ==
        LOOM_BENCHMARK_WORK_PLAN_INDEX_INVALID) {
      continue;
    }
    loom_benchmark_hal_compile_item_t* compile_item =
        &plan.hal_compile_items[work_item->hal_compile_item_index];
    work_item->execution_sample_base = compile_item->execution_sample_count;
    if (!iree_host_size_checked_add(
            compile_item->execution_sample_count,
            work_item->end_benchmark_sample - work_item->begin_benchmark_sample,
            &compile_item->execution_sample_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "selected execution sample count exceeds "
                                "host limits");
    }
  }
  iree_allocator_free(allocator, work_item_order);
  iree_allocator_free(allocator, case_compile_indices);

  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    loom_benchmark_work_plan_deinitialize(&plan);
  }
  return status;
}

void loom_benchmark_work_plan_deinitialize(loom_benchmark_work_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  iree_allocator_t allocator = plan->host_allocator;
  iree_allocator_free(allocator, plan->work_items);
  iree_allocator_free(allocator, plan->hal_compile_items);
  iree_allocator_free(allocator, plan->logical_samples);
  iree_allocator_free(allocator, plan->selected_benchmarks);
  iree_allocator_free(allocator, plan->planning_issues);
  *plan = (loom_benchmark_work_plan_t){0};
}
