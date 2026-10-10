// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/source.h"

#include <string.h>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/internal/path.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/vocab/vocab.h"

static iree_string_view_t text_file_text(iree_io_file_contents_t* contents) {
  return iree_make_string_view((const char*)contents->const_buffer.data,
                               contents->const_buffer.data_length);
}

static iree_status_t text_load_tokenizer(loom_serve_text_source_t* source,
                                         iree_string_view_t path,
                                         iree_string_view_t terminal_marker) {
  iree_io_file_contents_t* contents = NULL;
  IREE_RETURN_IF_ERROR(iree_io_file_contents_map(path, IREE_IO_FILE_ACCESS_READ,
                                                 source->allocator, &contents));
  iree_status_t status = iree_tokenizer_from_huggingface_json(
      text_file_text(contents), source->allocator, &source->tokenizer);
  iree_io_file_contents_free(contents);
  IREE_RETURN_IF_ERROR(status);
  source->eos_token = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(source->tokenizer), terminal_marker);
  if (source->eos_token < 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "tokenizer has no source-declared terminal marker");
  }
  return iree_ok_status();
}

// Bootstrap returns ordinary source-owned references. Its temporary program
// is gone before device setup; the shared environment outlives warm control.
static iree_status_t text_prepare(loom_serve_text_source_t* source,
                                  const loom_serve_text_options_t* options,
                                  loom_serve_text_storage_t* storage) {
  const iree_allocator_t allocator = source->allocator;
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(allocator, &source->environment));
  IREE_RETURN_IF_ERROR(
      iree_vm_ref_types_resolve(iree_vm_environment_lookup_ref_type_table(
                                    source->environment, IREE_SV("vm")),
                                &source->vm_types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("prepare.loom"), allocator, &path));
  iree_vm_buffer_t* shapes = NULL;
  iree_status_t status =
      iree_vm_buffer_create(options->epoch_count * 2 * sizeof(int64_t),
                            iree_alignof(int64_t), allocator, &shapes);
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
        1, allocator, &weights);
  }
  iree_vm_variant_t arguments[] = {
      iree_vm_variant_from_i64((int64_t)options->prefill_capacity),
      iree_vm_variant_from_i64((int64_t)options->context_capacity),
      iree_vm_variant_from_i64((int64_t)options->pool_capacity),
      iree_vm_variant_from_i64((int64_t)options->row_count),
      iree_vm_variant_from_i64((int64_t)options->checkpoint_capacity),
      iree_vm_variant_from_i32(options->enable_mtp),
      iree_vm_buffer_variant_from_ptr_move(&source->vm_types, &shapes),
      iree_vm_variant_from_i64((int64_t)options->epoch_count),
      iree_vm_buffer_variant_from_ptr_move(&source->vm_types, &weights),
  };
  iree_vm_variant_t results[4 + LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT] = {0};
  iree_vm_module_t* input = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(source->environment, NULL, &input,
                                            allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_create(
        source->environment, iree_make_cstring_view(path), IREE_SV("prepare"),
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(results),
        (iree_vm_module_span_t){&input, 1}, &source->compilation,
        &source->preparation, allocator);
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
      (prefill_capacity < 1 ||
       (uint64_t)prefill_capacity > options->context_capacity ||
       shared_stage_count < 1 ||
       (iree_host_size_t)shared_stage_count >
           loom_serve_preparation_stage_count(source->preparation))) {
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
  iree_vm_buffer_t* terminal = NULL;
  iree_const_byte_span_t terminal_bytes = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_ptr_from_variant_borrowed(&source->vm_types,
                                                      results[3], &terminal);
  }
  if (iree_status_is_ok(status) && !terminal) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "source terminal marker must not be null");
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(
        terminal, 0, iree_vm_buffer_length(terminal), &terminal_bytes);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_text_storage_initialize(
        &source->vm_types, results + 4, options->row_count,
        options->context_capacity, options->pool_capacity, storage,
        source->allocator);
  }
  if (iree_status_is_ok(status)) {
    const iree_host_size_t slots =
        options->checkpoint_capacity
            ? 2 * options->row_count + options->checkpoint_capacity
            : options->row_count;
    const uint64_t carry_bytes = iree_unaligned_load_le_u64(
        storage->bytes[LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS].data + 6 * 24);
    if (storage->recurrent.capacity < slots ||
        (options->enable_mtp &&
         options->row_count + options->checkpoint_capacity >
             carry_bytes / storage->carry_stride)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source storage lacks configured checkpoint headroom");
    }
  }
  if (iree_status_is_ok(status)) {
    status = text_load_tokenizer(
        source, options->tokenizer_path,
        iree_make_string_view((const char*)terminal_bytes.data,
                              terminal_bytes.data_length));
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(input);
  iree_allocator_free(allocator, path);
  return status;
}

iree_status_t loom_serve_text_source_initialize(
    const loom_serve_text_options_t* options,
    loom_serve_text_source_t* out_source,
    loom_serve_text_storage_t* out_storage, iree_allocator_t allocator) {
  memset(out_source, 0, sizeof(*out_source));
  out_source->allocator = allocator;
  IREE_RETURN_IF_ERROR(text_prepare(out_source, options, out_storage));
  const iree_vm_ref_type_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_register_types(out_source->environment, &table));
  IREE_RETURN_IF_ERROR(
      iree_hal_module_types_resolve(table, &out_source->types));
  char* path = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      options->source_directory, IREE_SV("control.loom"), allocator, &path));
  const iree_string_view_t roots[] = {
      IREE_SVL("step"),          IREE_SVL("epoch"),
      IREE_SVL("encode_epoch"),  IREE_SVL("publish_epoch"),
      IREE_SVL("encode_state"),  IREE_SVL("render_tool"),
      IREE_SVL("prepare_input"), IREE_SVL("text_end"),
      IREE_SVL("complete_text"), IREE_SVL("model_name"),
      IREE_SVL("chat_begin"),    IREE_SVL("chat_message"),
      IREE_SVL("chat_end"),      IREE_SVL("parse_tools")};
  iree_status_t status = loom_serve_program_compile(
      out_source->environment, iree_make_cstring_view(path),
      IREE_ARRAYSIZE(roots), roots, &out_source->compilation, allocator,
      &out_source->module);
  iree_allocator_free(allocator, path);
  return status;
}

void loom_serve_text_source_deinitialize(loom_serve_text_source_t* source) {
  loom_serve_preparation_destroy(source->preparation);
  iree_vm_module_release(source->module);
  iree_vm_buffer_release(source->control_state);
  iree_tokenizer_free(source->tokenizer);
  iree_vm_environment_free(source->environment);
}
