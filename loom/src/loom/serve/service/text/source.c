// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/source.h"

#include <string.h>

#include "iree/tokenizer/vocab/vocab.h"
#include "loom/runtime/program/input.h"
#include "loom/runtime/program/json.h"
#include "loom/runtime/program/program.h"
#include "loom/serve/service/text/state.h"
#include "loomc/iree.h"

iree_status_t loom_serve_text_source_resolve_tokenizer(
    loom_serve_text_source_t* source) {
  iree_const_byte_span_t bytes;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      source->terminal_marker, 0,
      iree_vm_buffer_length(source->terminal_marker), &bytes));
  source->eos_token = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(loom_runtime_tokenizer_value(source->tokenizer)),
      iree_make_string_view((const char*)bytes.data, bytes.data_length));
  if (source->eos_token < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no source-declared terminal marker");
  }
  return iree_ok_status();
}

// Bootstrap returns ordinary source-owned references. Its temporary program
// is gone before device setup; the shared environment outlives warm control.
static iree_status_t loom_serve_text_prepare(
    iree_vm_module_t* module, loom_serve_text_source_t* source,
    const loom_serve_text_options_t* options,
    loom_serve_text_storage_t* storage) {
  const iree_allocator_t host_allocator = source->host_allocator;
  iree_vm_buffer_t* shapes = NULL;
  iree_status_t status =
      iree_vm_buffer_create(options->epoch_count * 2 * sizeof(int64_t),
                            iree_alignof(int64_t), host_allocator, &shapes);
  iree_byte_span_t shape_bytes = iree_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(shapes, 0, iree_vm_buffer_length(shapes),
                                      &shape_bytes);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < options->epoch_count; ++i) {
      iree_unaligned_store_le_u64(shape_bytes.data + i * 16,
                                  options->epoch_shapes[i].token_capacity);
      iree_unaligned_store_le_u64(shape_bytes.data + i * 16 + 8,
                                  options->epoch_shapes[i].span_capacity);
    }
  }
  iree_vm_buffer_t* weights = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_clone(
        IREE_VM_BUFFER_ACCESS_FLAG_READ,
        iree_make_const_byte_span(options->weights_path.data,
                                  options->weights_path.size),
        1, host_allocator, &weights);
  }
  if (iree_status_is_ok(status)) {
    const iree_string_view_t json = iree_string_view_is_empty(options->options)
                                        ? IREE_SV("{}")
                                        : options->options;
    status =
        iree_vm_buffer_clone(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                             iree_make_const_byte_span(json.data, json.size), 1,
                             host_allocator, &source->options);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64((int64_t)options->prefill_capacity),
      iree_vm_variant_from_i64((int64_t)options->context_capacity),
      iree_vm_variant_from_i64((int64_t)options->pool_capacity),
      iree_vm_variant_from_i64((int64_t)options->row_count),
      iree_vm_variant_from_i64((int64_t)options->checkpoint_capacity),
      iree_vm_variant_from_i32((int32_t)options->draft_depth),
      iree_vm_buffer_variant_from_ptr_move(&source->vm_types, &shapes),
      iree_vm_variant_from_i64((int64_t)options->epoch_count),
      iree_vm_buffer_variant_from_ptr_move(&source->vm_types, &weights),
      iree_vm_buffer_variant_from_ptr_borrowed(&source->vm_types,
                                               source->options),
  };
  iree_vm_variant_t results[4 + LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT] = {0};
  iree_vm_module_t* modules[2] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_runtime_input_module_create(source->environment, NULL,
                                              host_allocator, &modules[0]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_json_module_create(source->environment,
                                             host_allocator, &modules[1]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_preparation_evaluate(
        source->environment, module, &options->sources, IREE_SV("prepare"),
        LOOM_RUNTIME_PREPARATION_FLAG_PARAMETERS,
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        iree_vm_module_span_from_array(modules), host_allocator,
        &source->preparation);
  }
  int64_t prefill_capacity = 0;
  int32_t shared_stage_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[0], &prefill_capacity);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[1], &shared_stage_count);
  }
  if (iree_status_is_ok(status) &&
      (prefill_capacity < 1 || prefill_capacity > INT32_MAX ||
       shared_stage_count < 1 ||
       (iree_host_size_t)shared_stage_count >
           loom_runtime_preparation_stage_count(source->preparation))) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "source bootstrap exceeds token runner capacities");
  }
  if (iree_status_is_ok(status)) {
    source->prefill_capacity = (iree_host_size_t)prefill_capacity;
    source->shared_stage_count = (iree_host_size_t)shared_stage_count;
    status = iree_vm_buffer_ptr_from_variant_move(
        &source->vm_types, &results[2], &source->control_state);
  }
  if (iree_status_is_ok(status) && !source->control_state) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source control state must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_move(
        &source->vm_types, &results[3], &source->terminal_marker);
  }
  if (iree_status_is_ok(status) && !source->terminal_marker &&
      source->mode == LOOM_SERVE_TEXT_MODE_GENERATE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source terminal marker must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_storage_initialize(
        &source->vm_types, results + 4, options->row_count,
        options->context_capacity, options->pool_capacity,
        source->host_allocator, storage);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t slots =
        options->checkpoint_capacity
            ? 2 * options->row_count + options->checkpoint_capacity
            : options->row_count;
    const uint64_t carry_bytes = storage->allocations[6].length;
    if (storage->recurrent.capacity < slots ||
        (storage->carry_stride &&
         options->row_count + options->checkpoint_capacity >
             carry_bytes / storage->carry_stride)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source storage lacks configured checkpoint headroom");
    }
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(modules[1]);
  iree_vm_module_release(modules[0]);
  return status;
}

typedef struct loom_serve_text_source_sizing_t {
  // Immutable preparation bytecode shared across isolated candidate processes.
  iree_vm_module_t* module;
  // Borrowed original deployment inputs; candidates change only endpoint count.
  const loom_serve_text_options_t* options;
  // Candidate declarations, independently owned by the enclosing source.
  loom_serve_text_source_t* source;
  // Candidate storage, transferred unchanged to native allocation after sizing.
  loom_serve_text_storage_t* storage;
  // Resolved source token addressability, initially explicit or zero.
  iree_host_size_t pool_capacity;
  // Cache payload bytes per physical page, stable across candidates.
  uint64_t page_bytes;
} loom_serve_text_source_sizing_t;

static void loom_serve_text_source_reset_candidate(
    loom_serve_text_source_t* source, loom_serve_text_storage_t* storage) {
  loom_runtime_preparation_free(source->preparation);
  source->preparation = NULL;
  iree_vm_buffer_release(source->control_state);
  source->control_state = NULL;
  iree_vm_buffer_release(source->terminal_marker);
  source->terminal_marker = NULL;
  iree_vm_buffer_release(source->options);
  source->options = NULL;
  loom_serve_text_storage_deinitialize(storage);
  memset(storage, 0, sizeof(*storage));
}

static iree_status_t loom_serve_text_storage_declared_bytes(
    const loom_serve_text_storage_t* storage, uint64_t* out_bytes) {
  *out_bytes = 0;
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT;
       ++i) {
    if (!iree_checked_add_u64(*out_bytes, storage->allocations[i].length,
                              out_bytes)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source storage size exceeds byte accounting");
    }
  }
  return iree_ok_status();
}

// Resolves source page geometry against the complete shared-pool boundary.
// The context-sized candidate is only a cold geometry probe; it reserves no
// native memory and is discarded before the selected preparation survives.
static iree_status_t loom_serve_text_source_resolve_pool_capacity(
    loom_serve_text_source_sizing_t* sizing,
    iree_host_size_t checkpoint_capacity) {
  loom_serve_text_options_t probe = *sizing->options;
  probe.checkpoint_capacity = checkpoint_capacity;
  probe.pool_capacity = probe.context_capacity;
  IREE_RETURN_IF_ERROR(loom_serve_text_prepare(sizing->module, sizing->source,
                                               &probe, sizing->storage));
  const loom_serve_text_storage_t* storage = sizing->storage;
  const uint64_t probe_page_count = probe.pool_capacity / storage->block_size;
  const uint64_t limit_bytes = sizing->options->cache_limit_bytes;
  const uint64_t page_count =
      iree_min(limit_bytes / storage->page_bytes, (uint64_t)UINT32_MAX);
  if (page_count < probe_page_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "shared pool byte limit cannot address one model context");
  }
  uint64_t pool_capacity = 0;
  if (!iree_checked_mul_u64(page_count, storage->block_size, &pool_capacity) ||
      pool_capacity > IREE_HOST_SIZE_MAX || pool_capacity > INT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source cache capacity is not representable");
  }
  sizing->pool_capacity = (iree_host_size_t)pool_capacity;
  sizing->page_bytes = storage->page_bytes;
  loom_serve_text_source_reset_candidate(sizing->source, sizing->storage);
  return iree_ok_status();
}

static iree_status_t loom_serve_text_source_evaluate_cache(
    void* user_data, iree_host_size_t checkpoint_capacity,
    loom_serve_text_cache_cost_t* out_cost) {
  loom_serve_text_source_sizing_t* sizing = user_data;
  loom_serve_text_source_t* source = sizing->source;
  loom_serve_text_storage_t* storage = sizing->storage;
  loom_serve_text_source_reset_candidate(source, storage);
  if (sizing->options->cache_limit_bytes && !sizing->pool_capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_text_source_resolve_pool_capacity(
        sizing, checkpoint_capacity));
  }
  loom_serve_text_options_t options = *sizing->options;
  options.checkpoint_capacity = checkpoint_capacity;
  options.pool_capacity = sizing->pool_capacity;
  IREE_RETURN_IF_ERROR(
      loom_serve_text_prepare(sizing->module, source, &options, storage));
  if (sizing->page_bytes && storage->page_bytes != sizing->page_bytes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source cache page geometry changes with endpoint capacity");
  }
  sizing->page_bytes = storage->page_bytes;
  uint64_t bytes = 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_storage_declared_bytes(storage, &bytes));
  uint64_t checkpoint_bytes = 0;
  const uint64_t checkpoint_stride =
      (uint64_t)storage->blocks_per_row * sizeof(uint32_t) +
      sizeof(loom_serve_text_state_checkpoint_t);
  if (!iree_checked_mul_u64(checkpoint_capacity, checkpoint_stride,
                            &checkpoint_bytes) ||
      !iree_checked_add_u64(bytes, checkpoint_bytes, &bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "checkpoint storage exceeds byte accounting");
  }
  *out_cost = (loom_serve_text_cache_cost_t){.storage_bytes = bytes};
  return iree_ok_status();
}

iree_status_t loom_serve_text_source_initialize(
    loom_runtime_jit_t* jit, const loom_serve_text_options_t* options,
    iree_allocator_t host_allocator, loom_serve_text_source_t* out_source,
    loom_serve_text_storage_t* out_storage) {
  memset(out_source, 0, sizeof(*out_source));
  memset(out_storage, 0, sizeof(*out_storage));
  out_source->mode = options->mode;
  out_source->eos_token = -1;
  out_source->host_allocator = host_allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(host_allocator, &out_source->environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    out_source->environment, IREE_SV("vm")),
                                &out_source->vm_types));
  const iree_string_view_t prepare = IREE_SV("prepare");
  loom_serve_text_source_sizing_t sizing = {
      .options = options,
      .source = out_source,
      .storage = out_storage,
      .pool_capacity = options->pool_capacity};
  IREE_RETURN_IF_ERROR(loom_runtime_program_compile_service_cached(
      jit, out_source->environment, &options->sources, options->sources.prepare,
      1, &prepare, (iree_string_view_list_t){0}, host_allocator,
      &sizing.module));
  iree_status_t status = loom_serve_text_cache_select(
      &options->cache, options->checkpoint_capacity,
      loom_serve_text_source_evaluate_cache, &sizing, &out_source->cache);
  iree_vm_module_release(sizing.module);
  IREE_RETURN_IF_ERROR(status);
  out_source->pool_capacity = sizing.pool_capacity;
  const loom_model_file_list_t* paths = NULL;
  const uint32_t source_count =
      loom_runtime_preparation_local_sources(out_source->preparation, &paths);
  IREE_RETURN_IF_ERROR(loom_runtime_parameter_sources_create(
      NULL, &options->sources, source_count, paths, host_allocator,
      &out_source->parameter_sources));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vm_register_types(out_source->environment, &table));
  IREE_RETURN_IF_ERROR(iree_hal_vm_types_resolve(table, &out_source->types));
  return iree_ok_status();
}

iree_status_t loom_serve_text_source_compile_control(
    loom_runtime_jit_t* jit, const loom_serve_text_options_t* options,
    loom_serve_text_source_t* source) {
  const iree_string_view_t generation_roots[] = {IREE_SVL("step"),
                                                 IREE_SVL("epoch"),
                                                 IREE_SVL("encode_epoch"),
                                                 IREE_SVL("publish_epoch"),
                                                 IREE_SVL("encode_state"),
                                                 IREE_SVL("resize_state"),
                                                 IREE_SVL("render_tool"),
                                                 IREE_SVL("prepare_input"),
                                                 IREE_SVL("output_next"),
                                                 IREE_SVL("complete_text"),
                                                 IREE_SVL("model_name"),
                                                 IREE_SVL("chat_reasoning"),
                                                 IREE_SVL("chat_image_limits"),
                                                 IREE_SVL("chat_begin"),
                                                 IREE_SVL("chat_message"),
                                                 IREE_SVL("chat_end"),
                                                 IREE_SVL("parse_tools"),
                                                 IREE_SVL("request_defaults"),
                                                 IREE_SVL("request_options"),
                                                 IREE_SVL("bind_request")};
  const iree_string_view_t encoding_roots[] = {
      IREE_SVL("epoch"),          IREE_SVL("encode_epoch"),
      IREE_SVL("publish_epoch"),  IREE_SVL("encode_state"),
      IREE_SVL("resize_state"),   IREE_SVL("bind_request"),
      IREE_SVL("capture_epoch"),  IREE_SVL("request_defaults"),
      IREE_SVL("prepare_request")};
  const bool encodes = options->mode == LOOM_SERVE_TEXT_MODE_ENCODE;
  const iree_string_view_t group_roots[] = {
      IREE_SVL("text_prefill_group_plan_v1"),
      IREE_SVL("text_encode_prefill_group_v1"),
      IREE_SVL("text_publish_prefill_group_v1")};
  const iree_string_view_list_t additional =
      options->group_mode == LOOM_RUNTIME_GROUP_MODE_TWO_BANK
          ? (iree_string_view_list_t){IREE_ARRAYSIZE(group_roots), group_roots}
          : (iree_string_view_list_t){0};
  return loom_runtime_program_compile_service_cached(
      jit, source->environment, &options->sources, options->sources.control,
      encodes ? IREE_ARRAYSIZE(encoding_roots)
              : IREE_ARRAYSIZE(generation_roots),
      encodes ? encoding_roots : generation_roots, additional,
      source->host_allocator, &source->module);
}

void loom_serve_text_source_deinitialize(loom_serve_text_source_t* source) {
  loom_runtime_parameter_sources_release(source->parameter_sources);
  loom_runtime_preparation_free(source->preparation);
  iree_vm_module_release(source->module);
  iree_vm_buffer_release(source->control_state);
  iree_vm_buffer_release(source->options);
  loom_runtime_tokenizer_release(source->tokenizer);
  iree_vm_buffer_release(source->terminal_marker);
  iree_vm_environment_free(source->environment);
}
