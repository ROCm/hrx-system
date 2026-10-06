// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source request observer. The image runner consumes the same VM payload in
// memory; this tool writes it for independent model-specific numerical checks.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "experimental/loom_serve/runtime/input.h"
#include "experimental/loom_serve/runtime/preparation.h"
#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/internal/json.h"
#include "iree/base/internal/path.h"
#include "iree/base/tooling/flags.h"
#include "iree/io/file_contents.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/vm/buffer.h"
#include "iree/vm/sync.h"

IREE_FLAG(string, model, "", "Source model directory.");
IREE_FLAG(string, checkpoint, "", "Checkpoint asset directory.");
IREE_FLAG(string, adapter, "", "Optional model-specific adapter asset.");
IREE_FLAG(string, prompt, "", "Prompt to encode.");
IREE_FLAG(string, seed, "0", "Unsigned decimal 64-bit seed.");
IREE_FLAG(int32_t, height, 384, "Output height in pixels.");
IREE_FLAG(int32_t, width, 384, "Output width in pixels.");
IREE_FLAG(int32_t, text_tokens, 512, "Requested text capacity.");
IREE_FLAG(float, strength, 1.0f, "Adapter strength.");
IREE_FLAG(string, output, "", "Existing directory for input-0.");

static iree_status_t text_argument(const iree_vm_ref_types_t* types,
                                   iree_string_view_t text,
                                   iree_vm_variant_t* out_value,
                                   iree_allocator_t allocator) {
  iree_vm_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_buffer_wrap(
      IREE_VM_BUFFER_ACCESS_FLAG_READ,
      iree_make_byte_span((void*)text.data, text.size),
      iree_vm_buffer_release_callback_null(), allocator, &buffer));
  *out_value = iree_vm_buffer_variant_from_ptr_move(types, &buffer);
  return iree_ok_status();
}

static iree_status_t run(iree_allocator_t allocator) {
  uint64_t seed = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_parse_uint64(iree_make_cstring_view(FLAG_seed), &seed));
  if (!FLAG_model[0] || !FLAG_checkpoint[0] || !FLAG_output[0]) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--model, --checkpoint and --output are required");
  }
  iree_vm_environment_t* environment = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_environment_allocate(allocator, &environment));
  iree_vm_ref_types_t types = {0};
  iree_status_t status = iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types);
  iree_vm_module_t* input = NULL;
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_input_module_create(environment, NULL, &input, allocator);
  }
  char* prepare_path = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_file_path_join(iree_make_cstring_view(FLAG_model),
                            IREE_SV("prepare.loom"), allocator, &prepare_path);
  }
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i64(FLAG_height),
                                   iree_vm_variant_from_i64(FLAG_width),
                                   iree_vm_variant_from_i64(FLAG_text_tokens),
                                   {0},
                                   {0}};
  if (iree_status_is_ok(status)) {
    status = text_argument(&types, iree_make_cstring_view(FLAG_checkpoint),
                           &arguments[3], allocator);
  }
  if (iree_status_is_ok(status)) {
    status = text_argument(&types, iree_make_cstring_view(FLAG_adapter),
                           &arguments[4], allocator);
  }
  iree_vm_variant_t metadata[4] = {0};
  loom_serve_preparation_t* preparation = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_serve_preparation_create(
        environment, iree_make_cstring_view(prepare_path), IREE_SV("prepare"),
        iree_vm_variant_span_from_array(arguments),
        iree_vm_variant_span_from_array(metadata),
        (iree_vm_module_span_t){&input, 1}, &preparation, allocator);
  }
  iree_allocator_free(allocator, prepare_path);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(arguments));
  iree_vm_module_release(input);
  input = NULL;
  iree_vm_buffer_t* asset = NULL;
  iree_const_byte_span_t asset_bytes = {0};
  char* tokenizer_path = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_ptr_from_variant_borrowed(&types, metadata[1], &asset);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(asset, 0, iree_vm_buffer_length(asset),
                                     &asset_bytes);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_file_path_join(iree_make_cstring_view(FLAG_checkpoint),
                            iree_make_string_view((const char*)asset_bytes.data,
                                                  asset_bytes.data_length),
                            allocator, &tokenizer_path);
  }
  iree_io_file_contents_t* contents = NULL;
  iree_tokenizer_t* tokenizer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_read(iree_make_cstring_view(tokenizer_path),
                                        allocator, &contents);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_from_huggingface_json(
        iree_make_string_view((const char*)contents->const_buffer.data,
                              contents->const_buffer.data_length),
        allocator, &tokenizer);
  }
  iree_io_file_contents_free(contents);
  iree_allocator_free(allocator, tokenizer_path);
  if (iree_status_is_ok(status)) {
    status = loom_serve_input_module_create(environment, tokenizer, &input,
                                            allocator);
  }
  char* control_path = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_file_path_join(iree_make_cstring_view(FLAG_model),
                            IREE_SV("control.loom"), allocator, &control_path);
  }
  loom_serve_program_t* program = NULL;
  const iree_string_view_t entry = IREE_SVL("prepare_request");
  if (iree_status_is_ok(status)) {
    status = loom_serve_program_create(
        environment, iree_make_cstring_view(control_path), 1, &entry,
        (iree_vm_module_span_t){&input, 1}, allocator, &program);
  }
  iree_allocator_free(allocator, control_path);
  iree_vm_buffer_t* tags = NULL;
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status)) {
    count = loom_serve_preparation_stage_count(preparation);
    status = iree_vm_buffer_create(count * 8, 8, allocator, &tags);
  }
  iree_byte_span_t tag_bytes = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_write(tags, 0, count * 8, &tag_bytes);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      iree_unaligned_store_le_u64(
          tag_bytes.data + i * 8,
          (uint64_t)loom_serve_preparation_stage(preparation, i)->tag);
    }
  }
  iree_vm_function_t function = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), entry, &function);
  }
  iree_vm_variant_t request[] = {
      iree_vm_variant_move(&metadata[3]),
      iree_vm_buffer_variant_from_ptr_move(&types, &tags),
      iree_vm_variant_from_i32((int32_t)count),
      {0},
      iree_vm_variant_from_i64((int64_t)seed),
      iree_vm_variant_from_f32(FLAG_strength)};
  if (iree_status_is_ok(status)) {
    status = text_argument(&types, iree_make_cstring_view(FLAG_prompt),
                           &request[3], allocator);
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    status = iree_vm_invoke(loom_serve_program_invocation(program), function,
                            iree_vm_variant_span_from_array(request),
                            iree_vm_variant_span_from_array(results));
  }
  int32_t stage = -1;
  if (iree_status_is_ok(status)) {
    status = iree_vm_i32_from_variant(results[0], &stage);
  }
  if (iree_status_is_ok(status) &&
      (stage < 0 || (iree_host_size_t)stage >= count)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE, "undeclared stage");
  }
  iree_vm_buffer_t* payload = NULL;
  iree_const_byte_span_t bytes = {0};
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_ptr_from_variant_borrowed(&types, results[1], &payload);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_buffer_map_read(payload, 0, iree_vm_buffer_length(payload),
                                     &bytes);
  }
  char* output_path = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_file_path_join(iree_make_cstring_view(FLAG_output),
                                 IREE_SV("input-0"), allocator, &output_path);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_contents_write(iree_make_cstring_view(output_path),
                                         bytes, allocator);
  }
  if (iree_status_is_ok(status)) {
    printf("{\"stage\":%d,\"tag\":%" PRId64 ",\"input_bytes\":%zu}\n", stage,
           loom_serve_preparation_stage(preparation, stage)->tag,
           bytes.data_length);
  }
  iree_allocator_free(allocator, output_path);
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(request));
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(metadata));
  loom_serve_program_destroy(program);
  loom_serve_preparation_destroy(preparation);
  iree_vm_module_release(input);
  iree_tokenizer_free(tokenizer);
  iree_vm_environment_free(environment);
  return status;
}

int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  iree_status_t status = run(iree_allocator_system());
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
