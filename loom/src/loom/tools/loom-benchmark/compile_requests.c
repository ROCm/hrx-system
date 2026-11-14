// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-benchmark/compile_requests.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "loom/tooling/io/file.h"
#include "loom/tooling/testbench/source_report.h"
#include "loom/tools/loom-benchmark/case_request.h"
#include "loom/tools/loom-benchmark/diagnostics.h"
#include "loom/tools/loom-benchmark/module_query.h"
#include "loom/tools/loom-benchmark/report.h"
#include "loom/util/adaptive_sort.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"
#include "loomc/interop.h"
#include "loomc/iree.h"

typedef struct loom_benchmark_compile_sample_ref_t {
  // Selected benchmark retaining the case and parameter assignments.
  iree_host_size_t selection_index;
  // Sample within that benchmark's reduced parameter sweep.
  iree_host_size_t benchmark_sample_ordinal;
  // Saturated Cartesian ordinal used only for grouping. Equal HOST_SIZE_MAX
  // keys are compared by their full parameter coordinates.
  iree_host_size_t sample_key;
  // Deduplicated capture satisfying this logical sample.
  iree_host_size_t capture_index;
  // Original report order, restored after grouping equal samples.
  iree_host_size_t order_index;
} loom_benchmark_compile_sample_ref_t;

typedef struct loom_benchmark_compile_capture_t {
  // Representative selected benchmark retaining case and assignments.
  iree_host_size_t selection_index;
  // Representative sample within the selected benchmark.
  iree_host_size_t benchmark_sample_ordinal;
  // True when capture or any published request fails compilation.
  bool failed;
} loom_benchmark_compile_capture_t;

static int loom_benchmark_compare_sample_ref_key(
    const loom_benchmark_work_plan_t* plan,
    const loom_benchmark_compile_sample_ref_t* lhs,
    const loom_benchmark_compile_sample_ref_t* rhs) {
  const loom_benchmark_selected_benchmark_t* lhs_selection =
      &plan->selected_benchmarks[lhs->selection_index];
  const loom_benchmark_selected_benchmark_t* rhs_selection =
      &plan->selected_benchmarks[rhs->selection_index];
  const iree_host_size_t lhs_case = lhs_selection->benchmark_plan->case_index;
  const iree_host_size_t rhs_case = rhs_selection->benchmark_plan->case_index;
  if (lhs_case != rhs_case) {
    return lhs_case < rhs_case ? -1 : 1;
  }
  if (lhs->sample_key != rhs->sample_key) {
    return lhs->sample_key < rhs->sample_key ? -1 : 1;
  }
  if (lhs->sample_key != IREE_HOST_SIZE_MAX) {
    return 0;
  }
  return loom_testbench_case_sample_compare(
      lhs_selection->case_plan,
      loom_testbench_benchmark_case_sample(lhs_selection->benchmark_plan,
                                           lhs->benchmark_sample_ordinal),
      loom_testbench_benchmark_case_sample(rhs_selection->benchmark_plan,
                                           rhs->benchmark_sample_ordinal));
}

static bool loom_benchmark_sample_ref_less(
    const loom_benchmark_work_plan_t* plan,
    const loom_benchmark_compile_sample_ref_t* lhs,
    const loom_benchmark_compile_sample_ref_t* rhs) {
  const int order = loom_benchmark_compare_sample_ref_key(plan, lhs, rhs);
  return order != 0 ? order < 0 : lhs->order_index < rhs->order_index;
}

LOOM_DEFINE_ADAPTIVE_SORT_WITH_CONTEXT(loom_benchmark_sort_sample_refs,
                                       loom_benchmark_compile_sample_ref_t,
                                       const loom_benchmark_work_plan_t*,
                                       loom_benchmark_sample_ref_less)

static int loom_benchmark_compare_sample_ref_order(const void* lhs_ptr,
                                                   const void* rhs_ptr) {
  const loom_benchmark_compile_sample_ref_t* lhs = lhs_ptr;
  const loom_benchmark_compile_sample_ref_t* rhs = rhs_ptr;
  if (lhs->order_index != rhs->order_index) {
    return lhs->order_index < rhs->order_index ? -1 : 1;
  }
  return 0;
}

static iree_status_t loom_benchmark_index_compile_samples(
    const loom_benchmark_compile_request_export_options_t* options,
    loom_benchmark_compile_sample_ref_t** out_sample_refs,
    iree_host_size_t* out_sample_ref_count,
    loom_benchmark_compile_capture_t** out_captures,
    iree_host_size_t* out_capture_count) {
  *out_sample_refs = NULL;
  *out_sample_ref_count = 0;
  *out_captures = NULL;
  *out_capture_count = 0;

  iree_host_size_t sample_ref_count = 0;
  for (iree_host_size_t i = 0; i < options->work_plan->selected_benchmark_count;
       ++i) {
    const loom_benchmark_selected_benchmark_t* selection =
        &options->work_plan->selected_benchmarks[i];
    if (selection->case_plan == NULL) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "compile-only request capture does not support check.scenario "
          "benchmark `%.*s`",
          (int)selection->benchmark_plan->name.size,
          selection->benchmark_plan->name.data);
    }
    const iree_host_size_t selection_sample_count =
        selection->end_benchmark_sample - selection->begin_benchmark_sample;
    if (!iree_host_size_checked_add(sample_ref_count, selection_sample_count,
                                    &sample_ref_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "selected compile sample count exceeds host "
                              "limits");
    }
  }

  loom_benchmark_compile_sample_ref_t* sample_refs = NULL;
  loom_benchmark_compile_capture_t* captures = NULL;
  if (sample_ref_count != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        options->host_allocator, sample_ref_count, sizeof(*sample_refs),
        (void**)&sample_refs));
    iree_status_t status =
        iree_allocator_malloc_array(options->host_allocator, sample_ref_count,
                                    sizeof(*captures), (void**)&captures);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(options->host_allocator, sample_refs);
      return status;
    }
  }

  iree_host_size_t sample_ref_index = 0;
  for (iree_host_size_t selection_index = 0;
       selection_index < options->work_plan->selected_benchmark_count;
       ++selection_index) {
    const loom_benchmark_selected_benchmark_t* selection =
        &options->work_plan->selected_benchmarks[selection_index];
    for (iree_host_size_t benchmark_sample_ordinal =
             selection->begin_benchmark_sample;
         benchmark_sample_ordinal < selection->end_benchmark_sample;
         ++benchmark_sample_ordinal) {
      sample_refs[sample_ref_index] = (loom_benchmark_compile_sample_ref_t){
          .selection_index = selection_index,
          .benchmark_sample_ordinal = benchmark_sample_ordinal,
          .order_index = sample_ref_index,
      };
      if (!loom_testbench_case_sample_flatten(
              selection->case_plan,
              loom_testbench_benchmark_case_sample(selection->benchmark_plan,
                                                   benchmark_sample_ordinal),
              &sample_refs[sample_ref_index].sample_key)) {
        sample_refs[sample_ref_index].sample_key = IREE_HOST_SIZE_MAX;
      }
      ++sample_ref_index;
    }
  }

  if (sample_ref_count > 1) {
    loom_benchmark_sort_sample_refs(options->work_plan, sample_refs,
                                    sample_ref_count);
  }
  iree_host_size_t capture_count = 0;
  for (iree_host_size_t i = 0; i < sample_ref_count; ++i) {
    const bool begins_capture =
        i == 0 ||
        loom_benchmark_compare_sample_ref_key(
            options->work_plan, &sample_refs[i - 1], &sample_refs[i]) != 0;
    if (begins_capture) {
      captures[capture_count] = (loom_benchmark_compile_capture_t){
          .selection_index = sample_refs[i].selection_index,
          .benchmark_sample_ordinal = sample_refs[i].benchmark_sample_ordinal,
      };
      ++capture_count;
    }
    sample_refs[i].capture_index = capture_count - 1;
  }
  if (sample_ref_count > 1) {
    qsort(sample_refs, sample_ref_count, sizeof(*sample_refs),
          loom_benchmark_compare_sample_ref_order);
  }

  *out_sample_refs = sample_refs;
  *out_sample_ref_count = sample_ref_count;
  *out_captures = captures;
  *out_capture_count = capture_count;
  return iree_ok_status();
}

static uint64_t loom_benchmark_compile_request_hash(
    loomc_byte_span_t contents) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (loomc_host_size_t i = 0; i < contents.data_length; ++i) {
    hash ^= contents.data[i];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

static iree_string_view_t loom_benchmark_source_format_name(
    loomc_source_format_t format) {
  switch (format) {
    case LOOMC_SOURCE_FORMAT_TEXT:
      return IREE_SV("text");
    case LOOMC_SOURCE_FORMAT_BYTECODE:
      return IREE_SV("bytecode");
    default:
      return IREE_SV("unknown");
  }
}

static iree_status_t loom_benchmark_write_config_json(
    const loom_config_text_binding_set_t* config_text,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < config_text->binding_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("key"), config_text->bindings[i].key));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("value"), config_text->bindings[i].value));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_benchmark_write_binding_type_json(
    loom_type_formatter_t type_formatter, loom_type_t type,
    loom_json_object_writer_t* object) {
  IREE_RETURN_IF_ERROR(loom_json_object_begin_field(object, IREE_SV("type")));
  loom_json_string_writer_t string;
  IREE_RETURN_IF_ERROR(loom_json_string_begin(object->stream, &string));
  IREE_RETURN_IF_ERROR(
      type_formatter.fn(type, type_formatter.user_data, &string.stream));
  return loom_json_string_end(&string);
}

static iree_status_t loom_benchmark_write_bindings_json(
    const loomc_module_interop_view_t* module_view,
    const loom_benchmark_case_request_capture_t* capture,
    loom_output_stream_t* stream) {
  const loom_module_t* module = module_view->module;
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  for (iree_host_size_t i = 0; i < capture->binding_count; ++i) {
    const loom_value_id_t value_id = capture->binding_value_ids[i];
    const loom_value_t* value = loom_module_value(module, value_id);
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("binding_index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("value_id"), value_id));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field_if_nonempty(
        &object, IREE_SV("name"), loom_benchmark_value_name(module, value_id)));
    IREE_RETURN_IF_ERROR(loom_benchmark_write_binding_type_json(
        module_view->type_formatter, value->type, &object));
    if (!loom_value_is_block_arg(value)) {
      const loom_op_t* defining_op = loom_value_owner_op(value);
      if (defining_op != NULL) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
            &object, IREE_SV("defining_op"),
            loom_op_name(module, defining_op)));
        IREE_RETURN_IF_ERROR(loom_testbench_write_source_location_json(
            module, defining_op->location, &object));
      }
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_benchmark_emit_compile_request_failure(
    const loom_benchmark_compile_request_export_options_t* options,
    iree_string_view_t kind, iree_string_view_t message,
    const loomc_result_t* result) {
  loom_benchmark_diagnostic_capture_t diagnostics;
  loom_benchmark_diagnostic_capture_initialize(options->host_allocator,
                                               &diagnostics);
  iree_status_t status =
      loom_benchmark_diagnostic_capture_loomc_result(&diagnostics, result);
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_event_sink_emit_failure(
        options->event_sink, options->run, IREE_SV("compile"), kind, message,
        &diagnostics);
  }
  loom_benchmark_diagnostic_capture_deinitialize(&diagnostics);
  return status;
}

static iree_status_t loom_benchmark_write_result_diagnostics_json(
    const loom_benchmark_compile_request_export_options_t* options,
    const loomc_result_t* result, loom_json_object_writer_t* object) {
  loom_benchmark_diagnostic_capture_t diagnostics;
  loom_benchmark_diagnostic_capture_initialize(options->host_allocator,
                                               &diagnostics);
  iree_status_t status =
      loom_benchmark_diagnostic_capture_loomc_result(&diagnostics, result);
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(object, IREE_SV("diagnostics"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_write_diagnostic_array_json(&diagnostics,
                                                        object->stream);
  }
  loom_benchmark_diagnostic_capture_deinitialize(&diagnostics);
  return status;
}

static iree_status_t loom_benchmark_resolve_request_roots(
    const loom_benchmark_compile_request_export_options_t* options,
    const loomc_request_t* request, loomc_string_view_t** out_root_names,
    loomc_module_t** out_module) {
  *out_root_names = NULL;
  *out_module = NULL;
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  loomc_string_view_t* root_names = NULL;
  if (root_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_allocator_malloc_array(options->host_allocator, root_count,
                                    sizeof(*root_names), (void**)&root_names));
  }
  loomc_result_t* result = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_module_deserialize_request(
          options->context, options->workspace, request,
          loomc_allocator_from_iree(options->host_allocator), out_module,
          root_names, &result));
  // Capture produced these immutable bytes with this context's serializer.
  // Successful deserialization must initialize every projected root name.
  IREE_ASSERT(!iree_status_is_ok(status) || loomc_result_succeeded(result));
  loomc_result_release(result);
  if (!iree_status_is_ok(status)) {
    loomc_module_release(*out_module);
    *out_module = NULL;
    iree_allocator_free(options->host_allocator, root_names);
    return status;
  }
  *out_root_names = root_names;
  return iree_ok_status();
}

static iree_status_t loom_benchmark_write_request_roots_json(
    const loomc_request_t* request, const loomc_string_view_t* root_names,
    loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  const loomc_host_size_t root_count = loomc_request_root_count(request);
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    loomc_request_root_t root;
    loomc_request_root_at(request, (loomc_request_root_ordinal_t)i, &root);
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("root_ordinal"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("module_ordinal"), root.module_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("symbol_ordinal"), root.symbol_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("symbol"),
        iree_string_view_from_loomc(root_names[i])));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_benchmark_write_request_bindings_json(
    const loomc_request_t* request, loom_output_stream_t* stream) {
  loom_json_array_writer_t array;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &array));
  const loomc_host_size_t binding_count = loomc_request_binding_count(request);
  for (loomc_host_size_t i = 0; i < binding_count; ++i) {
    loomc_request_binding_t binding;
    loomc_request_binding_at(request, i, &binding);
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&array));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("binding_ordinal"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("requirement_ordinal"), binding.requirement_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &object, IREE_SV("root_ordinal"), binding.root_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&array);
}

static iree_status_t loom_benchmark_write_request_file(
    const loom_benchmark_compile_request_export_options_t* options,
    iree_host_size_t capture_index, iree_host_size_t request_index,
    loomc_byte_span_t contents, iree_string_builder_t* leaf) {
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      leaf, "capture%04" PRIhsz "-r%04" PRIhsz ".loombc", capture_index,
      request_index));
  char* path_storage = NULL;
  iree_status_t status = loom_tooling_file_path_join(
      options->artifact_bundle->compile_request_dir,
      iree_string_builder_view(leaf), options->host_allocator, &path_storage);
  if (iree_status_is_ok(status)) {
    status = loom_tooling_create_parent_directory(
        iree_make_cstring_view(path_storage), options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_make_string_view((const char*)contents.data, contents.data_length),
        options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_artifact_bundle_record_file(
        options->artifact_bundle, LOOM_BENCHMARK_BUNDLE_FILE_COMPILE_REQUEST,
        iree_make_cstring_view(path_storage));
  }
  iree_allocator_free(options->host_allocator, path_storage);
  return status;
}

static iree_status_t loom_benchmark_write_request_json(
    const loom_benchmark_compile_request_export_options_t* options,
    const loom_benchmark_compile_capture_t* capture_record,
    iree_host_size_t capture_index, const loomc_request_t* request,
    iree_host_size_t request_index, loom_output_stream_t* stream,
    bool* out_failed) {
  *out_failed = false;
  const loom_benchmark_selected_benchmark_t* selection =
      &options->work_plan->selected_benchmarks[capture_record->selection_index];
  const loomc_source_t* source = loomc_request_source(request);
  const loomc_byte_span_t contents = loomc_source_contents(source);

  iree_string_builder_t leaf;
  iree_string_builder_initialize(options->host_allocator, &leaf);
  iree_status_t status = loom_benchmark_write_request_file(
      options, capture_index, request_index, contents, &leaf);

  loomc_string_view_t* root_names = NULL;
  loomc_module_t* request_module = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_resolve_request_roots(options, request, &root_names,
                                                  &request_module);
  }

  if (iree_status_is_ok(status)) {
    loom_json_object_writer_t object;
    status = loom_json_object_begin(stream, &object);
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_host_size_field(
          &object, IREE_SV("request_index"), request_index);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &object, IREE_SV("file"), iree_string_builder_view(&leaf));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_begin_field(&object, IREE_SV("source"));
    }
    loom_json_object_writer_t source_object;
    if (iree_status_is_ok(status)) {
      status = loom_json_object_begin(stream, &source_object);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &source_object, IREE_SV("identifier"),
          iree_string_view_from_loomc(loomc_source_identifier(source)));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &source_object, IREE_SV("format"),
          loom_benchmark_source_format_name(loomc_source_format(source)));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_host_size_field(
          &source_object, IREE_SV("byte_count"), contents.data_length);
    }
    char hash[17];
    iree_snprintf(hash, sizeof(hash), "%016" PRIx64,
                  loom_benchmark_compile_request_hash(contents));
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &source_object, IREE_SV("hash_algorithm"), IREE_SV("fnv1a64"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_write_string_field(
          &source_object, IREE_SV("hash"), iree_make_cstring_view(hash));
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_end(&source_object);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_begin_field(&object, IREE_SV("roots"));
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_benchmark_write_request_roots_json(request, root_names, stream);
    }
    if (iree_status_is_ok(status)) {
      status =
          loom_json_object_begin_field(&object, IREE_SV("parent_bindings"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_write_request_bindings_json(request, stream);
    }

    loomc_product_t* product = NULL;
    loomc_result_t* result = NULL;
    if (iree_status_is_ok(status)) {
      const loomc_compile_options_t compile_options = {
          .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
          .structure_size = sizeof(compile_options),
      };
      status = iree_status_from_loomc(loomc_compile_request(
          options->compiler, options->workspace, options->pass_program, request,
          &compile_options, loomc_allocator_from_iree(options->host_allocator),
          &product, &result));
    }
    if (iree_status_is_ok(status)) {
      const bool succeeded = loomc_result_succeeded(result);
      status = loom_json_object_write_string_field(
          &object, IREE_SV("compile_state"),
          succeeded ? IREE_SV("succeeded") : IREE_SV("failed"));
      if (iree_status_is_ok(status) && succeeded) {
        status = loom_json_object_write_host_size_field(
            &object, IREE_SV("export_count"),
            loomc_product_export_count(product));
      }
      if (iree_status_is_ok(status) && succeeded) {
        status = loom_json_object_write_host_size_field(
            &object, IREE_SV("artifact_count"),
            loomc_product_artifact_count(product));
      }
      if (iree_status_is_ok(status) && succeeded) {
        status = loom_json_object_write_host_size_field(
            &object, IREE_SV("requirement_count"),
            loomc_product_requirement_count(product));
      }
      if (iree_status_is_ok(status) && !succeeded) {
        *out_failed = true;
        status = loom_benchmark_write_result_diagnostics_json(options, result,
                                                              &object);
      }
      if (iree_status_is_ok(status) && !succeeded) {
        iree_string_builder_t message;
        iree_string_builder_initialize(options->host_allocator, &message);
        status = iree_string_builder_append_format(
            &message,
            "canonical request %" PRIhsz " for benchmark `%.*s` sample %" PRIhsz
            " failed compilation",
            request_index, (int)selection->benchmark_plan->name.size,
            selection->benchmark_plan->name.data,
            capture_record->benchmark_sample_ordinal);
        if (iree_status_is_ok(status)) {
          status = loom_benchmark_emit_compile_request_failure(
              options, IREE_SV("request_compile"),
              iree_string_builder_view(&message), result);
        }
        iree_string_builder_deinitialize(&message);
      }
    }
    loomc_result_release(result);
    loomc_product_release(product);
    if (iree_status_is_ok(status)) {
      status = loom_json_object_end(&object);
    }
  }

  iree_allocator_free(options->host_allocator, root_names);
  loomc_module_release(request_module);
  iree_string_builder_deinitialize(&leaf);
  return status;
}

static iree_status_t loom_benchmark_write_capture_json(
    const loom_benchmark_compile_request_export_options_t* options,
    iree_host_size_t capture_index,
    loom_benchmark_compile_capture_t* capture_record,
    loom_output_stream_t* stream,
    loom_benchmark_compile_request_export_result_t* export_result) {
  const loom_benchmark_selected_benchmark_t* selection =
      &options->work_plan->selected_benchmarks[capture_record->selection_index];
  const loom_testbench_case_plan_t* case_plan = selection->case_plan;
  const loom_testbench_case_sample_t sample =
      loom_testbench_benchmark_case_sample(
          selection->benchmark_plan, capture_record->benchmark_sample_ordinal);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("capture_index"), capture_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("case_index"), selection->benchmark_plan->case_index));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("case"), case_plan->name));
  IREE_RETURN_IF_ERROR(loom_benchmark_write_sample_fields_json(
      options->module_view->module, case_plan, sample, &object));

  const loom_benchmark_case_request_options_t capture_options = {
      .workspace = options->workspace,
      .module = options->module,
      .case_index = selection->benchmark_plan->case_index,
      .case_sample = sample,
      .config = options->config,
      .target_profile = options->target_profile,
  };
  loom_benchmark_case_request_capture_t capture = {0};
  loomc_result_t* capture_result = NULL;
  iree_status_t status = loom_benchmark_case_request_capture(
      &capture_options, options->host_allocator, &capture, &capture_result);
  if (!iree_status_is_ok(status)) {
    loomc_result_release(capture_result);
    loom_benchmark_case_request_capture_deinitialize(&capture);
    return status;
  }
  const bool capture_succeeded = loomc_result_succeeded(capture_result);
  status = loom_json_object_write_string_field(
      &object, IREE_SV("capture_state"),
      capture_succeeded ? IREE_SV("succeeded") : IREE_SV("failed"));

  if (iree_status_is_ok(status) && !capture_succeeded) {
    capture_record->failed = true;
    ++export_result->failure_count;
    status = loom_benchmark_write_result_diagnostics_json(
        options, capture_result, &object);
    iree_string_builder_t message;
    iree_string_builder_initialize(options->host_allocator, &message);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(
          &message,
          "benchmark `%.*s` sample %" PRIhsz
          " failed canonical request capture",
          (int)selection->benchmark_plan->name.size,
          selection->benchmark_plan->name.data,
          capture_record->benchmark_sample_ordinal);
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_emit_compile_request_failure(
          options, IREE_SV("request_capture"),
          iree_string_builder_view(&message), capture_result);
    }
    iree_string_builder_deinitialize(&message);
  } else if (iree_status_is_ok(status)) {
    status = loom_json_object_write_string_field(
        &object, IREE_SV("command_root"), capture.root_symbol);
    if (iree_status_is_ok(status)) {
      status =
          loom_json_object_begin_field(&object, IREE_SV("command_bindings"));
    }
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_write_bindings_json(options->module_view,
                                                  &capture, stream);
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_object_begin_field(&object, IREE_SV("requests"));
    }
    loom_json_array_writer_t requests;
    if (iree_status_is_ok(status)) {
      status = loom_json_array_begin(stream, &requests);
    }
    for (iree_host_size_t i = 0;
         i < capture.request_count && iree_status_is_ok(status); ++i) {
      status = loom_json_array_begin_element(&requests);
      bool request_failed = false;
      if (iree_status_is_ok(status)) {
        status = loom_benchmark_write_request_json(
            options, capture_record, capture_index, capture.requests[i], i,
            stream, &request_failed);
      }
      if (iree_status_is_ok(status)) {
        ++export_result->request_count;
        if (request_failed) {
          capture_record->failed = true;
          ++export_result->failure_count;
        }
      }
    }
    if (iree_status_is_ok(status)) {
      status = loom_json_array_end(&requests);
    }
  }

  loomc_result_release(capture_result);
  loom_benchmark_case_request_capture_deinitialize(&capture);
  if (iree_status_is_ok(status)) {
    status = loom_json_object_end(&object);
  }
  return status;
}

static iree_status_t loom_benchmark_write_selections_json(
    const loom_benchmark_compile_request_export_options_t* options,
    const loom_benchmark_compile_sample_ref_t* sample_refs,
    iree_host_size_t sample_ref_count, loom_output_stream_t* stream) {
  loom_json_array_writer_t selections;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &selections));
  iree_host_size_t sample_ref_index = 0;
  for (iree_host_size_t selection_index = 0;
       selection_index < options->work_plan->selected_benchmark_count;
       ++selection_index) {
    const loom_benchmark_selected_benchmark_t* selection =
        &options->work_plan->selected_benchmarks[selection_index];
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&selections));
    loom_json_object_writer_t object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &object));
    IREE_RETURN_IF_ERROR(loom_benchmark_write_candidate_identity_json(
        &selection->identity, &object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("benchmark_index"), selection->benchmark_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("benchmark"), selection->benchmark_plan->name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("case_index"), selection->benchmark_plan->case_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &object, IREE_SV("case"), selection->case_plan->name));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("begin_benchmark_sample"),
        selection->begin_benchmark_sample));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &object, IREE_SV("end_benchmark_sample"),
        selection->end_benchmark_sample));
    IREE_RETURN_IF_ERROR(
        loom_json_object_begin_field(&object, IREE_SV("samples")));
    loom_json_array_writer_t samples;
    IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &samples));
    while (sample_ref_index < sample_ref_count &&
           sample_refs[sample_ref_index].selection_index == selection_index) {
      const loom_benchmark_compile_sample_ref_t* sample_ref =
          &sample_refs[sample_ref_index++];
      IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&samples));
      loom_json_object_writer_t sample;
      IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &sample));
      IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
          &sample, IREE_SV("benchmark_sample_ordinal"),
          sample_ref->benchmark_sample_ordinal));
      iree_host_size_t case_sample_ordinal = 0;
      if (loom_testbench_case_sample_flatten(
              selection->case_plan,
              loom_testbench_benchmark_case_sample(
                  selection->benchmark_plan,
                  sample_ref->benchmark_sample_ordinal),
              &case_sample_ordinal)) {
        IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
            &sample, IREE_SV("case_sample_ordinal"), case_sample_ordinal));
      }
      IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
          &sample, IREE_SV("capture_index"), sample_ref->capture_index));
      IREE_RETURN_IF_ERROR(loom_json_object_end(&sample));
    }
    IREE_RETURN_IF_ERROR(loom_json_array_end(&samples));
    IREE_RETURN_IF_ERROR(loom_json_object_end(&object));
  }
  return loom_json_array_end(&selections);
}

static iree_host_size_t loom_benchmark_count_failed_selections(
    const loom_benchmark_compile_request_export_options_t* options,
    const loom_benchmark_compile_sample_ref_t* sample_refs,
    iree_host_size_t sample_ref_count,
    const loom_benchmark_compile_capture_t* captures) {
  iree_host_size_t failed_selection_count = 0;
  iree_host_size_t sample_ref_index = 0;
  for (iree_host_size_t selection_index = 0;
       selection_index < options->work_plan->selected_benchmark_count;
       ++selection_index) {
    bool failed = false;
    while (sample_ref_index < sample_ref_count &&
           sample_refs[sample_ref_index].selection_index == selection_index) {
      failed |= captures[sample_refs[sample_ref_index].capture_index].failed;
      ++sample_ref_index;
    }
    failed_selection_count += failed;
  }
  return failed_selection_count;
}

static iree_status_t loom_benchmark_write_compile_request_manifest(
    const loom_benchmark_compile_request_export_options_t* options,
    const loom_benchmark_compile_sample_ref_t* sample_refs,
    iree_host_size_t sample_ref_count,
    loom_benchmark_compile_capture_t* captures, iree_host_size_t capture_count,
    loom_benchmark_compile_request_export_result_t* export_result) {
  iree_string_builder_t manifest;
  iree_string_builder_initialize(options->host_allocator, &manifest);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&manifest, &stream);

  loom_json_object_writer_t object;
  iree_status_t status = loom_json_object_begin(&stream, &object);
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_string_field(
        &object, IREE_SV("kind"), IREE_SV("loom.compile_requests"));
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_json_object_write_uint32_field(&object, IREE_SV("version"), 1);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_string_field(&object, IREE_SV("source"),
                                                 options->run->source);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_string_field(&object, IREE_SV("target"),
                                                 options->target);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t pipeline =
        iree_string_view_is_empty(iree_string_view_trim(options->pipeline))
            ? IREE_SV("default")
            : iree_string_view_trim(options->pipeline);
    status = loom_json_object_write_string_field(&object, IREE_SV("pipeline"),
                                                 pipeline);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&object, IREE_SV("sanitizer"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_write_sanitizer_options_json(options->sanitizer,
                                                         &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&object, IREE_SV("config"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_write_config_json(options->config_text, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&object, IREE_SV("captures"));
  }
  loom_json_array_writer_t capture_array;
  if (iree_status_is_ok(status)) {
    status = loom_json_array_begin(&stream, &capture_array);
  }
  for (iree_host_size_t i = 0; i < capture_count && iree_status_is_ok(status);
       ++i) {
    status = loom_json_array_begin_element(&capture_array);
    if (iree_status_is_ok(status)) {
      status = loom_benchmark_write_capture_json(options, i, &captures[i],
                                                 &stream, export_result);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_array_end(&capture_array);
  }
  if (iree_status_is_ok(status)) {
    export_result->failed_selection_count =
        loom_benchmark_count_failed_selections(options, sample_refs,
                                               sample_ref_count, captures);
    status = loom_json_object_begin_field(&object, IREE_SV("selections"));
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_write_selections_json(options, sample_refs,
                                                  sample_ref_count, &stream);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin_field(&object, IREE_SV("summary"));
  }
  loom_json_object_writer_t summary;
  if (iree_status_is_ok(status)) {
    status = loom_json_object_begin(&stream, &summary);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("selection_count"),
        options->work_plan->selected_benchmark_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("selected_sample_count"), sample_ref_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("capture_count"), export_result->capture_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("request_count"), export_result->request_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("failure_count"), export_result->failure_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_write_host_size_field(
        &summary, IREE_SV("failed_selection_count"),
        export_result->failed_selection_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_end(&summary);
  }
  if (iree_status_is_ok(status)) {
    status = loom_json_object_end(&object);
  }

  char* path_storage = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_tooling_file_path_join(
        options->artifact_bundle->compile_request_dir, IREE_SV("manifest.json"),
        options->host_allocator, &path_storage);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_create_parent_directory(
        iree_make_cstring_view(path_storage), options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_tooling_write_output_file(
        iree_make_cstring_view(path_storage),
        iree_string_builder_view(&manifest), options->host_allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_benchmark_artifact_bundle_record_file(
        options->artifact_bundle,
        LOOM_BENCHMARK_BUNDLE_FILE_COMPILE_REQUEST_MANIFEST,
        iree_make_cstring_view(path_storage));
  }
  iree_allocator_free(options->host_allocator, path_storage);
  iree_string_builder_deinitialize(&manifest);
  return status;
}

iree_status_t loom_benchmark_export_compile_requests(
    const loom_benchmark_compile_request_export_options_t* options,
    loom_benchmark_compile_request_export_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->run);
  IREE_ASSERT_ARGUMENT(options->context);
  IREE_ASSERT_ARGUMENT(options->compiler);
  IREE_ASSERT_ARGUMENT(options->workspace);
  IREE_ASSERT_ARGUMENT(options->pass_program);
  IREE_ASSERT_ARGUMENT(options->module);
  IREE_ASSERT_ARGUMENT(options->module_view);
  IREE_ASSERT_ARGUMENT(options->module_plan);
  IREE_ASSERT_ARGUMENT(options->work_plan);
  IREE_ASSERT_ARGUMENT(options->config);
  IREE_ASSERT_ARGUMENT(options->config_text);
  IREE_ASSERT_ARGUMENT(options->target_profile);
  IREE_ASSERT_ARGUMENT(options->sanitizer);
  IREE_ASSERT_ARGUMENT(options->artifact_bundle);
  IREE_ASSERT_ARGUMENT(options->event_sink);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_benchmark_compile_request_export_result_t){0};
  loom_benchmark_compile_sample_ref_t* sample_refs = NULL;
  iree_host_size_t sample_ref_count = 0;
  loom_benchmark_compile_capture_t* captures = NULL;
  iree_host_size_t capture_count = 0;
  iree_status_t status = loom_benchmark_index_compile_samples(
      options, &sample_refs, &sample_ref_count, &captures, &capture_count);
  if (iree_status_is_ok(status)) {
    out_result->capture_count = capture_count;
    status = loom_benchmark_write_compile_request_manifest(
        options, sample_refs, sample_ref_count, captures, capture_count,
        out_result);
  }
  iree_allocator_free(options->host_allocator, captures);
  iree_allocator_free(options->host_allocator, sample_refs);
  return status;
}
