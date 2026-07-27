// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/work_plan.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "loom/tools/iree-benchmark-loom/module_query.h"

static iree_status_t iree_benchmark_loom_duration_ms_to_ns(
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

static iree_status_t iree_benchmark_loom_policy_from_benchmark(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_benchmark_policy_t* out_policy) {
  memset(out_policy, 0, sizeof(*out_policy));
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);

  (void)module_plan;
  iree_string_view_t measure = options->measure;
  iree_benchmark_loom_measure_t measure_kind =
      IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  if (iree_string_view_equal(measure, IREE_SV("case_end_to_end")) ||
      iree_string_view_equal(measure, IREE_SV("end_to_end"))) {
    measure = IREE_SV("case_end_to_end");
    measure_kind = IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END;
  } else if (iree_string_view_equal(measure, IREE_SV("dispatch_complete"))) {
    measure_kind = IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE;
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "benchmark `%.*s` requested unsupported V0 "
                            "measure '%.*s'",
                            (int)benchmark_plan->name.size,
                            benchmark_plan->name.data, (int)measure.size,
                            measure.data);
  }
  if (measure_kind != IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE &&
      (!iree_string_view_is_empty(options->profile_artifacts_dir) ||
       options->profile_data_requested ||
       options->profile_counters.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "final-batch profile flags require benchmark `%.*s` "
        "to use measure = \"dispatch_complete\"",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }

  *out_policy = (iree_benchmark_loom_benchmark_policy_t){
      .measure_kind = measure_kind,
      .measure = measure,
      .warmup_iterations = options->warmup_iterations,
      .iterations = options->iterations,
  };
  loom_run_hal_benchmark_options_initialize(&out_policy->hal_options);
  if (measure_kind == IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
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
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_duration_ms_to_ns(
        options->min_time_ms, &min_duration_ns));
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_duration_ms_to_ns(
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
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_parse_profile_data_families(
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
              .name = IREE_SV("iree-benchmark-loom"),
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

static iree_host_size_t iree_benchmark_loom_compare_token_count(
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

static iree_status_t iree_benchmark_loom_find_benchmark_by_name(
    const loom_testbench_module_plan_t* module_plan, iree_string_view_t name,
    const loom_testbench_benchmark_plan_t** out_benchmark) {
  *out_benchmark = NULL;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark =
        &module_plan->benchmarks[i];
    if (iree_string_view_equal(benchmark->name, name)) {
      *out_benchmark = benchmark;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "--compare benchmark '@%.*s' was not found",
                          (int)name.size, name.data);
}

static iree_status_t iree_benchmark_loom_selected_benchmark_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const loom_testbench_benchmark_plan_t* benchmark_plan,
    const iree_benchmark_loom_options_t* options,
    iree_host_size_t candidate_index,
    iree_benchmark_loom_selected_benchmark_t* out_selection) {
  if (benchmark_plan->case_index >= module_plan->case_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "benchmark `%.*s` does not reference a planned check.case",
        (int)benchmark_plan->name.size, benchmark_plan->name.data);
  }
  if (benchmark_plan->sample_count == 0) {
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[benchmark_plan->case_index];
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected benchmark `%.*s` for check.case `%.*s` has zero executable "
        "samples; selected_benchmark='%.*s', selected_case='%.*s', "
        "sample=%d",
        (int)benchmark_plan->name.size, benchmark_plan->name.data,
        (int)case_plan->name.size, case_plan->name.data,
        (int)options->selected_benchmark.size, options->selected_benchmark.data,
        (int)options->selected_case.size, options->selected_case.data,
        options->sample_ordinal);
  }
  memset(out_selection, 0, sizeof(*out_selection));
  snprintf(out_selection->candidate_id_storage,
           sizeof(out_selection->candidate_id_storage), "c%" PRIhsz,
           candidate_index);
  out_selection->identity = (iree_benchmark_loom_candidate_identity_t){
      .candidate_id =
          iree_make_cstring_view(out_selection->candidate_id_storage),
      .candidate_index = candidate_index,
  };
  out_selection->benchmark_plan = benchmark_plan;
  out_selection->case_plan = &module_plan->cases[benchmark_plan->case_index];
  return iree_benchmark_loom_policy_from_benchmark(
      module_plan, benchmark_plan, options, &out_selection->policy);
}

static iree_status_t iree_benchmark_loom_sample_window(
    const iree_benchmark_loom_options_t* options, iree_host_size_t sample_count,
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

static bool iree_benchmark_loom_profile_counter_sets_equal(
    const iree_hal_profile_counter_set_selection_t* lhs,
    const iree_hal_profile_counter_set_selection_t* rhs,
    iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (!iree_string_view_equal(lhs[i].name, rhs[i].name) ||
        lhs[i].counter_name_count != rhs[i].counter_name_count) {
      return false;
    }
    for (iree_host_size_t j = 0; j < lhs[i].counter_name_count; ++j) {
      if (!iree_string_view_equal(lhs[i].counter_names[j],
                                  rhs[i].counter_names[j])) {
        return false;
      }
    }
  }
  return true;
}

static bool iree_benchmark_loom_policy_equal(
    const iree_benchmark_loom_benchmark_policy_t* lhs,
    const iree_benchmark_loom_benchmark_policy_t* rhs) {
  if (lhs->measure_kind != rhs->measure_kind ||
      !iree_string_view_equal(lhs->measure, rhs->measure) ||
      lhs->warmup_iterations != rhs->warmup_iterations ||
      lhs->iterations != rhs->iterations) {
    return false;
  }
  const loom_run_hal_benchmark_options_t* lhs_hal = &lhs->hal_options;
  const loom_run_hal_benchmark_options_t* rhs_hal = &rhs->hal_options;
  if (lhs_hal->flags != rhs_hal->flags ||
      lhs_hal->timing.batch_size != rhs_hal->timing.batch_size ||
      lhs_hal->timing.warmup_batch_count !=
          rhs_hal->timing.warmup_batch_count ||
      lhs_hal->timing.warmup_min_duration_ns !=
          rhs_hal->timing.warmup_min_duration_ns ||
      lhs_hal->timing.min_batch_count != rhs_hal->timing.min_batch_count ||
      lhs_hal->timing.min_duration_ns != rhs_hal->timing.min_duration_ns ||
      lhs_hal->timing.max_batch_count != rhs_hal->timing.max_batch_count ||
      lhs_hal->timing.stable_p90_to_p50_delta_ppm !=
          rhs_hal->timing.stable_p90_to_p50_delta_ppm ||
      lhs_hal->dispatch_batch.dispatch_count !=
          rhs_hal->dispatch_batch.dispatch_count ||
      lhs_hal->profile_flags != rhs_hal->profile_flags ||
      lhs_hal->profile_data_families != rhs_hal->profile_data_families ||
      lhs_hal->profile_counter_set_count !=
          rhs_hal->profile_counter_set_count ||
      !iree_string_view_equal(lhs_hal->profile_artifact_path,
                              rhs_hal->profile_artifact_path)) {
    return false;
  }
  return iree_benchmark_loom_profile_counter_sets_equal(
      lhs_hal->profile_counter_sets, rhs_hal->profile_counter_sets,
      lhs_hal->profile_counter_set_count);
}

static bool iree_benchmark_loom_dispatch_compile_item_matches(
    const iree_benchmark_loom_dispatch_compile_item_t* item,
    const iree_benchmark_loom_work_plan_t* plan,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_string_view_t sample_compilation, bool has_case_sample_ordinal,
    iree_host_size_t case_sample_ordinal) {
  const iree_benchmark_loom_selected_benchmark_t* representative =
      &plan->selected_benchmarks[item->representative_selection_index];
  return representative->case_plan == selection->case_plan &&
         iree_string_view_equal(item->sample_compilation, sample_compilation) &&
         item->has_case_sample_ordinal == has_case_sample_ordinal &&
         (!has_case_sample_ordinal ||
          item->case_sample_ordinal == case_sample_ordinal);
}

static iree_host_size_t iree_benchmark_loom_find_or_append_compile_item(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_string_view_t sample_compilation, bool has_case_sample_ordinal,
    iree_host_size_t case_sample_ordinal) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  for (iree_host_size_t i = 0; i < plan->dispatch_compile_item_count; ++i) {
    if (iree_benchmark_loom_dispatch_compile_item_matches(
            &plan->dispatch_compile_items[i], plan, selection,
            sample_compilation, has_case_sample_ordinal, case_sample_ordinal)) {
      return i;
    }
  }
  const iree_host_size_t compile_item_index =
      plan->dispatch_compile_item_count++;
  plan->dispatch_compile_items[compile_item_index] =
      (iree_benchmark_loom_dispatch_compile_item_t){
          .compile_item_index = compile_item_index,
          .representative_selection_index = selection_index,
          .sample_compilation = sample_compilation,
          .has_case_sample_ordinal = has_case_sample_ordinal,
          .case_sample_ordinal = case_sample_ordinal,
      };
  return compile_item_index;
}

static bool iree_benchmark_loom_benchmark_sample_ranges_equal(
    const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_benchmark_plan_t* lhs,
    iree_host_size_t lhs_begin_sample,
    const loom_testbench_benchmark_plan_t* rhs,
    iree_host_size_t rhs_begin_sample, iree_host_size_t sample_count) {
  for (iree_host_size_t i = 0; i < sample_count; ++i) {
    const iree_host_size_t lhs_case_sample_ordinal =
        loom_testbench_benchmark_sample_case_ordinal(case_plan, lhs,
                                                     lhs_begin_sample + i);
    const iree_host_size_t rhs_case_sample_ordinal =
        loom_testbench_benchmark_sample_case_ordinal(case_plan, rhs,
                                                     rhs_begin_sample + i);
    if (lhs_case_sample_ordinal != rhs_case_sample_ordinal) {
      return false;
    }
  }
  return true;
}

static bool iree_benchmark_loom_work_item_matches(
    const iree_benchmark_loom_work_item_t* item,
    const iree_benchmark_loom_work_plan_t* plan,
    iree_benchmark_loom_work_item_kind_t kind,
    const iree_benchmark_loom_selected_benchmark_t* selection,
    iree_string_view_t sample_compilation, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, bool has_case_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    iree_host_size_t dispatch_compile_item_index) {
  if (item->kind != kind ||
      !iree_string_view_equal(item->sample_compilation, sample_compilation) ||
      item->has_case_sample_ordinal != has_case_sample_ordinal ||
      item->dispatch_compile_item_index != dispatch_compile_item_index) {
    return false;
  }
  if (kind == IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE) {
    if (!has_case_sample_ordinal ||
        item->case_sample_ordinal != case_sample_ordinal) {
      return false;
    }
  } else {
    if (item->begin_benchmark_sample != begin_sample ||
        item->end_benchmark_sample != end_sample) {
      return false;
    }
    if (has_case_sample_ordinal &&
        item->case_sample_ordinal != case_sample_ordinal) {
      return false;
    }
  }
  const iree_benchmark_loom_selected_benchmark_t* representative =
      &plan->selected_benchmarks[item->representative_selection_index];
  if (representative->case_plan != selection->case_plan ||
      !iree_benchmark_loom_policy_equal(&representative->policy,
                                        &selection->policy)) {
    return false;
  }
  if (kind == IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END &&
      representative->benchmark_plan != selection->benchmark_plan &&
      !iree_benchmark_loom_benchmark_sample_ranges_equal(
          selection->case_plan, representative->benchmark_plan,
          item->begin_benchmark_sample, selection->benchmark_plan, begin_sample,
          end_sample - begin_sample)) {
    return false;
  }
  return true;
}

static iree_host_size_t iree_benchmark_loom_find_or_append_work_item(
    iree_benchmark_loom_work_plan_t* plan,
    iree_benchmark_loom_work_item_kind_t kind, iree_host_size_t selection_index,
    iree_string_view_t sample_compilation, iree_host_size_t begin_sample,
    iree_host_size_t end_sample, bool has_case_sample_ordinal,
    iree_host_size_t case_sample_ordinal,
    iree_host_size_t dispatch_compile_item_index) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  for (iree_host_size_t i = 0; i < plan->work_item_count; ++i) {
    if (iree_benchmark_loom_work_item_matches(
            &plan->work_items[i], plan, kind, selection, sample_compilation,
            begin_sample, end_sample, has_case_sample_ordinal,
            case_sample_ordinal, dispatch_compile_item_index)) {
      return i;
    }
  }
  const iree_host_size_t work_item_index = plan->work_item_count++;
  plan->work_items[work_item_index] = (iree_benchmark_loom_work_item_t){
      .kind = kind,
      .work_item_index = work_item_index,
      .representative_selection_index = selection_index,
      .dispatch_compile_item_index = dispatch_compile_item_index,
      .sample_compilation = sample_compilation,
      .begin_benchmark_sample = begin_sample,
      .end_benchmark_sample = end_sample,
      .has_case_sample_ordinal = has_case_sample_ordinal,
      .case_sample_ordinal = case_sample_ordinal,
  };
  return work_item_index;
}

static void iree_benchmark_loom_append_logical_sample(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_host_size_t begin_sample, iree_host_size_t end_sample,
    bool has_case_sample_ordinal, iree_host_size_t case_sample_ordinal,
    iree_string_view_t sample_compilation, iree_host_size_t work_item_index) {
  plan->logical_samples[plan->logical_sample_count++] =
      (iree_benchmark_loom_logical_sample_t){
          .selection_index = selection_index,
          .begin_benchmark_sample = begin_sample,
          .end_benchmark_sample = end_sample,
          .has_case_sample_ordinal = has_case_sample_ordinal,
          .case_sample_ordinal = case_sample_ordinal,
          .sample_compilation = sample_compilation,
          .work_item_index = work_item_index,
      };
}

static iree_status_t iree_benchmark_loom_count_logical_samples(
    const iree_benchmark_loom_selected_benchmark_t* selection,
    const iree_benchmark_loom_options_t* options,
    iree_host_size_t* inout_logical_sample_count) {
  if (selection->policy.measure_kind ==
      IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END) {
    *inout_logical_sample_count += 1;
    return iree_ok_status();
  }

  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_sample_window(
      options, selection->benchmark_plan->sample_count, &begin_sample,
      &end_sample));
  const iree_host_size_t selected_sample_count = end_sample - begin_sample;
  if (iree_benchmark_loom_sample_compilation_runs_once(
          options->sample_compilation_mode)) {
    *inout_logical_sample_count += selected_sample_count;
  }
  if (iree_benchmark_loom_sample_compilation_runs_per_sample(
          options->sample_compilation_mode)) {
    *inout_logical_sample_count += selected_sample_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_append_case_end_to_end_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    const iree_benchmark_loom_options_t* options) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_sample_window(
      options, selection->benchmark_plan->sample_count, &begin_sample,
      &end_sample));
  bool has_case_sample_ordinal = false;
  iree_host_size_t case_sample_ordinal = 0;
  if (end_sample == begin_sample + 1) {
    has_case_sample_ordinal = true;
    case_sample_ordinal = loom_testbench_benchmark_sample_case_ordinal(
        selection->case_plan, selection->benchmark_plan, begin_sample);
  }
  const iree_host_size_t work_item_index =
      iree_benchmark_loom_find_or_append_work_item(
          plan, IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END, selection_index,
          iree_string_view_empty(), begin_sample, end_sample,
          has_case_sample_ordinal, case_sample_ordinal,
          IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID);
  iree_benchmark_loom_append_logical_sample(
      plan, selection_index, begin_sample, end_sample, has_case_sample_ordinal,
      case_sample_ordinal, iree_string_view_empty(), work_item_index);
  return iree_ok_status();
}

static void iree_benchmark_loom_append_dispatch_sample_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    iree_string_view_t sample_compilation, bool has_compile_case_sample_ordinal,
    iree_host_size_t compile_case_sample_ordinal,
    iree_host_size_t benchmark_sample_ordinal) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  const iree_host_size_t case_sample_ordinal =
      loom_testbench_benchmark_sample_case_ordinal(selection->case_plan,
                                                   selection->benchmark_plan,
                                                   benchmark_sample_ordinal);
  const iree_host_size_t compile_item_index =
      iree_benchmark_loom_find_or_append_compile_item(
          plan, selection_index, sample_compilation,
          has_compile_case_sample_ordinal, compile_case_sample_ordinal);
  const iree_host_size_t work_item_index =
      iree_benchmark_loom_find_or_append_work_item(
          plan, IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE, selection_index,
          sample_compilation, benchmark_sample_ordinal,
          benchmark_sample_ordinal + 1, /*has_case_sample_ordinal=*/true,
          case_sample_ordinal, compile_item_index);
  iree_benchmark_loom_append_logical_sample(
      plan, selection_index, benchmark_sample_ordinal,
      benchmark_sample_ordinal + 1, /*has_case_sample_ordinal=*/true,
      case_sample_ordinal, sample_compilation, work_item_index);
}

static iree_status_t iree_benchmark_loom_append_dispatch_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    const iree_benchmark_loom_options_t* options) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  iree_host_size_t begin_sample = 0;
  iree_host_size_t end_sample = 0;
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_sample_window(
      options, selection->benchmark_plan->sample_count, &begin_sample,
      &end_sample));
  if (iree_benchmark_loom_sample_compilation_runs_once(
          options->sample_compilation_mode)) {
    for (iree_host_size_t sample_ordinal = begin_sample;
         sample_ordinal < end_sample; ++sample_ordinal) {
      iree_benchmark_loom_append_dispatch_sample_work(
          plan, selection_index, IREE_SV("once"),
          /*has_compile_case_sample_ordinal=*/false,
          /*compile_case_sample_ordinal=*/0, sample_ordinal);
    }
  }
  if (iree_benchmark_loom_sample_compilation_runs_per_sample(
          options->sample_compilation_mode)) {
    for (iree_host_size_t sample_ordinal = begin_sample;
         sample_ordinal < end_sample; ++sample_ordinal) {
      const iree_host_size_t case_sample_ordinal =
          loom_testbench_benchmark_sample_case_ordinal(
              selection->case_plan, selection->benchmark_plan, sample_ordinal);
      iree_benchmark_loom_append_dispatch_sample_work(
          plan, selection_index, IREE_SV("per_sample"),
          /*has_compile_case_sample_ordinal=*/true, case_sample_ordinal,
          sample_ordinal);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_append_selection_work(
    iree_benchmark_loom_work_plan_t* plan, iree_host_size_t selection_index,
    const iree_benchmark_loom_options_t* options) {
  const iree_benchmark_loom_selected_benchmark_t* selection =
      &plan->selected_benchmarks[selection_index];
  if (selection->policy.measure_kind ==
      IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE) {
    return iree_benchmark_loom_append_dispatch_work(plan, selection_index,
                                                    options);
  }
  return iree_benchmark_loom_append_case_end_to_end_work(plan, selection_index,
                                                         options);
}

static iree_status_t iree_benchmark_loom_allocate_array(
    iree_allocator_t allocator, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(allocator, count, element_size, out_ptr));
  memset(*out_ptr, 0, count * element_size);
  return iree_ok_status();
}

static iree_status_t iree_benchmark_loom_select_compare_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_work_plan_t* plan) {
  iree_string_view_t compare = iree_string_view_trim(options->compare);
  const iree_host_size_t selection_count =
      iree_benchmark_loom_compare_token_count(compare);
  if (selection_count < 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--compare requires at least two comma-separated check.benchmark "
        "symbols");
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  iree_status_t status = iree_ok_status();
  iree_host_size_t selection_index = 0;
  iree_string_view_t remaining = compare;
  while (iree_status_is_ok(status) && !iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_benchmark_loom_normalize_selection_name(token);
    if (iree_string_view_is_empty(token)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "--compare contains an empty benchmark name");
      break;
    }
    const loom_testbench_benchmark_plan_t* benchmark = NULL;
    status = iree_benchmark_loom_find_benchmark_by_name(module_plan, token,
                                                        &benchmark);
    if (iree_status_is_ok(status)) {
      status = iree_benchmark_loom_selected_benchmark_initialize(
          module_plan, benchmark, options, selection_index,
          &plan->selected_benchmarks[selection_index]);
    }
    ++selection_index;
    remaining = iree_string_view_trim(remaining);
  }
  if (iree_status_is_ok(status)) {
    plan->selected_benchmark_count = selection_count;
  }
  return status;
}

static iree_status_t iree_benchmark_loom_select_matching_benchmarks(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options,
    iree_benchmark_loom_work_plan_t* plan) {
  iree_host_size_t selection_count = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    if (benchmark_plan->case_index >= module_plan->case_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "benchmark `%.*s` does not reference a planned check.case",
          (int)benchmark_plan->name.size, benchmark_plan->name.data);
    }
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[benchmark_plan->case_index];
    if (iree_benchmark_loom_benchmark_matches_selection(
            benchmark_plan, options->selected_benchmark) &&
        iree_benchmark_loom_case_matches_selection(case_plan,
                                                   options->selected_case)) {
      ++selection_count;
    }
  }
  if (selection_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no check.benchmark matched '%.*s'",
                            (int)options->selected_benchmark.size,
                            options->selected_benchmark.data);
  }
  IREE_RETURN_IF_ERROR(iree_benchmark_loom_allocate_array(
      plan->host_allocator, selection_count, sizeof(*plan->selected_benchmarks),
      (void**)&plan->selected_benchmarks));

  iree_host_size_t selection_index = 0;
  for (iree_host_size_t i = 0; i < module_plan->benchmark_count; ++i) {
    const loom_testbench_benchmark_plan_t* benchmark_plan =
        &module_plan->benchmarks[i];
    const loom_testbench_case_plan_t* case_plan =
        &module_plan->cases[benchmark_plan->case_index];
    if (!iree_benchmark_loom_benchmark_matches_selection(
            benchmark_plan, options->selected_benchmark) ||
        !iree_benchmark_loom_case_matches_selection(case_plan,
                                                    options->selected_case)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_benchmark_loom_selected_benchmark_initialize(
        module_plan, benchmark_plan, options, selection_index,
        &plan->selected_benchmarks[selection_index]));
    ++selection_index;
  }
  plan->selected_benchmark_count = selection_count;
  return iree_ok_status();
}

iree_status_t iree_benchmark_loom_work_plan_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options, iree_allocator_t allocator,
    iree_benchmark_loom_work_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module_plan);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (iree_benchmark_loom_work_plan_t){0};

  iree_benchmark_loom_work_plan_t plan = {
      .host_allocator = allocator,
  };
  iree_status_t status = iree_ok_status();
  if (!iree_string_view_is_empty(options->compare)) {
    status = iree_benchmark_loom_select_compare_benchmarks(module_plan, options,
                                                           &plan);
  } else {
    status = iree_benchmark_loom_select_matching_benchmarks(module_plan,
                                                            options, &plan);
  }

  iree_host_size_t logical_sample_capacity = 0;
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.selected_benchmark_count; ++i) {
    status = iree_benchmark_loom_count_logical_samples(
        &plan.selected_benchmarks[i], options, &logical_sample_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity, sizeof(*plan.logical_samples),
        (void**)&plan.logical_samples);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity,
        sizeof(*plan.dispatch_compile_items),
        (void**)&plan.dispatch_compile_items);
  }
  if (iree_status_is_ok(status)) {
    status = iree_benchmark_loom_allocate_array(
        allocator, logical_sample_capacity, sizeof(*plan.work_items),
        (void**)&plan.work_items);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < plan.selected_benchmark_count; ++i) {
    status = iree_benchmark_loom_append_selection_work(&plan, i, options);
  }

  if (iree_status_is_ok(status)) {
    *out_plan = plan;
  } else {
    iree_benchmark_loom_work_plan_deinitialize(&plan);
  }
  return status;
}

void iree_benchmark_loom_work_plan_deinitialize(
    iree_benchmark_loom_work_plan_t* plan) {
  IREE_ASSERT_ARGUMENT(plan);
  iree_allocator_t allocator = plan->host_allocator;
  iree_allocator_free(allocator, plan->work_items);
  iree_allocator_free(allocator, plan->dispatch_compile_items);
  iree_allocator_free(allocator, plan->logical_samples);
  iree_allocator_free(allocator, plan->selected_benchmarks);
  *plan = (iree_benchmark_loom_work_plan_t){0};
}
