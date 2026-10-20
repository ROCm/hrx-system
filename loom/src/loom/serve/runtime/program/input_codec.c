// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/input_codec.h"

#include "iree/tokenizer/vocab/vocab.h"

struct loom_serve_input_encoder_t {
  // Intrusive VM ownership prefix.
  iree_vm_ref_object_t object;
  // Allocator owning this allocation and runtime model workspace.
  iree_allocator_t host_allocator;
  // Runtime encoder stored in the trailing allocation.
  iree_tokenizer_encode_state_t* state;
  // Native token staging, independent of source buffer alignment/endian order.
  int32_t* tokens;
  // Maximum tokens produced by one feed/finalize call.
  iree_host_size_t token_capacity;
};

struct loom_serve_input_decoder_t {
  // Intrusive VM ownership prefix.
  iree_vm_ref_object_t object;
  // Allocator owning this complete allocation.
  iree_allocator_t host_allocator;
  // Borrowed immutable tokenizer, outliving every call and final destruction.
  const iree_tokenizer_t* tokenizer;
  // Runtime decoder stored in state_storage.
  iree_tokenizer_decode_state_t* state;
  // Reusable runtime state storage, retained across resets.
  iree_byte_span_t state_storage;
  // Native token staging, independent of source buffer alignment/endian order.
  int32_t* tokens;
  // Maximum IDs accepted by one feed call.
  iree_host_size_t token_capacity;
  // Finalization forbids feed until reset, including predecoded fast paths.
  bool finalizing;
};

static void loom_serve_input_encoder_destroy(void* pointer) {
  loom_serve_input_encoder_t* encoder = pointer;
  iree_tokenizer_encode_state_deinitialize(encoder->state);
  iree_allocator_free(encoder->host_allocator, encoder);
}

static void loom_serve_input_decoder_destroy(void* pointer) {
  loom_serve_input_decoder_t* decoder = pointer;
  iree_tokenizer_decode_state_deinitialize(decoder->state);
  iree_allocator_free(decoder->host_allocator, decoder);
}

static const iree_vm_ref_type_descriptor_t loom_serve_input_decoder_type = {
    .destroy = loom_serve_input_decoder_destroy,
    .table = &loom_serve_input_codec_type_table,
    .type_name = IREE_SVL("decoder"),
};
static const iree_vm_ref_type_descriptor_t loom_serve_input_encoder_type = {
    .destroy = loom_serve_input_encoder_destroy,
    .table = &loom_serve_input_codec_type_table,
    .type_name = IREE_SVL("encoder"),
};
static const iree_vm_ref_type_t loom_serve_input_codec_types[] = {
    &loom_serve_input_decoder_type,
    &loom_serve_input_encoder_type,
};
const iree_vm_ref_type_table_t loom_serve_input_codec_type_table = {
    .structure_size = sizeof(iree_vm_ref_type_table_t),
    .namespace_name = IREE_SVL("input"),
    .types = {loom_serve_input_codec_types,
              IREE_ARRAYSIZE(loom_serve_input_codec_types)},
};

iree_status_t loom_serve_input_validate_encode_flags(
    iree_tokenizer_encode_flags_t flags) {
  if (flags & ~(IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS |
                IREE_TOKENIZER_ENCODE_FLAG_NO_SPECIAL_TOKEN_MATCHING)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported input encoding flags 0x%08x", flags);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_input_validate_decode_flags(
    iree_tokenizer_decode_flags_t flags) {
  if (flags & ~IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported input decoding flags 0x%08x", flags);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_input_read_token_ids(const iree_tokenizer_t* tokenizer,
                                              iree_const_byte_span_t bytes,
                                              int32_t* out_tokens) {
  const iree_tokenizer_vocab_t* vocab = iree_tokenizer_vocab(tokenizer);
  const iree_host_size_t vocab_capacity = iree_tokenizer_vocab_capacity(vocab);
  iree_status_t status = iree_ok_status();
  const iree_host_size_t count = bytes.data_length / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    const uint32_t id = iree_unaligned_load_le_u32(bytes.data + i * 4);
    if (id >= vocab_capacity ||
        (iree_tokenizer_vocab_token_attrs(vocab, (int32_t)id) &
         IREE_TOKENIZER_TOKEN_ATTR_UNUSED)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "token ID %" PRIu32 " is not in the vocabulary", id);
    } else {
      out_tokens[i] = (int32_t)id;
    }
  }
  return status;
}

iree_status_t loom_serve_input_encoder_create(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_encode_flags_t flags,
    iree_host_size_t transform_capacity, iree_host_size_t token_capacity,
    iree_allocator_t host_allocator, iree_vm_ref_t* out_ref) {
  *out_ref = iree_vm_ref_null();
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_encode_flags(flags));
  if (!token_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "encoder token capacity must be positive");
  }
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  iree_host_size_t total_size = 0;
  iree_host_size_t state_offset = 0;
  iree_host_size_t tokens_offset = 0;
  iree_host_size_t transform_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_input_encoder_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(state_size, uint8_t, iree_max_align_t,
                                &state_offset),
      IREE_STRUCT_FIELD_ALIGNED(token_capacity, int32_t, iree_alignof(int32_t),
                                &tokens_offset),
      IREE_STRUCT_FIELD(transform_capacity, uint8_t, &transform_offset)));
  loom_serve_input_encoder_t* encoder = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&encoder));
  encoder->host_allocator = host_allocator;
  encoder->tokens = (int32_t*)((uint8_t*)encoder + tokens_offset);
  encoder->token_capacity = token_capacity;
  iree_status_t status = iree_tokenizer_encode_state_initialize(
      tokenizer,
      iree_make_byte_span((uint8_t*)encoder + state_offset, state_size),
      iree_make_byte_span((uint8_t*)encoder + transform_offset,
                          transform_capacity),
      (iree_tokenizer_offset_run_list_t){0},
      flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, host_allocator,
      &encoder->state);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_object_initialize(&encoder->object);
    *out_ref = iree_vm_ref_from_ptr_move((void**)&encoder,
                                         &loom_serve_input_encoder_type);
  } else {
    iree_allocator_free(host_allocator, encoder);
  }
  return status;
}

iree_status_t loom_serve_input_encoder_reset(
    loom_serve_input_encoder_t* encoder, iree_tokenizer_encode_flags_t flags) {
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_encode_flags(flags));
  return iree_tokenizer_encode_state_reset(
      encoder->state, flags | IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);
}

static void loom_serve_input_encoder_write(loom_serve_input_encoder_t* encoder,
                                           iree_host_size_t count,
                                           iree_byte_span_t output) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(output.data + i * 4,
                                (uint32_t)encoder->tokens[i]);
  }
}

iree_status_t loom_serve_input_encoder_feed(loom_serve_input_encoder_t* encoder,
                                            iree_string_view_t input,
                                            iree_byte_span_t output,
                                            iree_host_size_t* out_consumed,
                                            iree_host_size_t* out_written) {
  const iree_host_size_t capacity =
      iree_min(encoder->token_capacity, output.data_length / sizeof(int32_t));
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_feed(
      encoder->state, input,
      iree_tokenizer_make_token_output(encoder->tokens, NULL, NULL, capacity),
      out_consumed, out_written));
  loom_serve_input_encoder_write(encoder, *out_written, output);
  return iree_ok_status();
}

iree_status_t loom_serve_input_encoder_finalize(
    loom_serve_input_encoder_t* encoder, iree_byte_span_t output,
    iree_host_size_t* out_written, bool* out_complete) {
  const iree_host_size_t capacity =
      iree_min(encoder->token_capacity, output.data_length / sizeof(int32_t));
  IREE_RETURN_IF_ERROR(iree_tokenizer_encode_state_finalize(
      encoder->state,
      iree_tokenizer_make_token_output(encoder->tokens, NULL, NULL, capacity),
      out_written));
  loom_serve_input_encoder_write(encoder, *out_written, output);
  *out_complete = !iree_tokenizer_encode_state_has_pending(encoder->state);
  return iree_ok_status();
}

iree_status_t loom_serve_input_decoder_create(
    const iree_tokenizer_t* tokenizer, iree_tokenizer_decode_flags_t flags,
    iree_host_size_t token_capacity, iree_allocator_t host_allocator,
    iree_vm_ref_t* out_ref) {
  *out_ref = iree_vm_ref_null();
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_decode_flags(flags));
  if (!token_capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "decoder token capacity must be positive");
  }
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_decode_state_calculate_size(tokenizer, &state_size));
  iree_host_size_t total_size = 0;
  iree_host_size_t state_offset = 0;
  iree_host_size_t tokens_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_input_decoder_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(state_size, uint8_t, iree_max_align_t,
                                &state_offset),
      IREE_STRUCT_FIELD_ALIGNED(token_capacity, int32_t, iree_alignof(int32_t),
                                &tokens_offset)));
  loom_serve_input_decoder_t* decoder = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&decoder));
  decoder->host_allocator = host_allocator;
  decoder->tokenizer = tokenizer;
  decoder->state_storage =
      iree_make_byte_span((uint8_t*)decoder + state_offset, state_size);
  decoder->tokens = (int32_t*)((uint8_t*)decoder + tokens_offset);
  decoder->token_capacity = token_capacity;
  iree_status_t status = iree_tokenizer_decode_state_initialize(
      tokenizer, flags, decoder->state_storage, &decoder->state);
  if (iree_status_is_ok(status)) {
    iree_vm_ref_object_initialize(&decoder->object);
    *out_ref = iree_vm_ref_from_ptr_move((void**)&decoder,
                                         &loom_serve_input_decoder_type);
  } else {
    iree_allocator_free(host_allocator, decoder);
  }
  return status;
}

iree_status_t loom_serve_input_decoder_reset(
    loom_serve_input_decoder_t* decoder, iree_tokenizer_decode_flags_t flags) {
  IREE_RETURN_IF_ERROR(loom_serve_input_validate_decode_flags(flags));
  iree_tokenizer_decode_state_deinitialize(decoder->state);
  decoder->state = NULL;
  decoder->finalizing = false;
  return iree_tokenizer_decode_state_initialize(
      decoder->tokenizer, flags, decoder->state_storage, &decoder->state);
}

iree_status_t loom_serve_input_decoder_feed(loom_serve_input_decoder_t* decoder,
                                            iree_const_byte_span_t input,
                                            iree_byte_span_t output,
                                            iree_host_size_t* out_consumed,
                                            iree_host_size_t* out_written) {
  if (decoder->finalizing) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "decoder feed requires reset after finalization");
  }
  const iree_host_size_t count =
      iree_min(decoder->token_capacity, input.data_length / sizeof(int32_t));
  IREE_RETURN_IF_ERROR(loom_serve_input_read_token_ids(
      decoder->tokenizer, iree_make_const_byte_span(input.data, count * 4),
      decoder->tokens));
  char empty_output = 0;
  return iree_tokenizer_decode_state_feed(
      decoder->state, iree_tokenizer_make_token_id_list(decoder->tokens, count),
      iree_make_mutable_string_view(
          output.data ? (char*)output.data : &empty_output, output.data_length),
      out_consumed, out_written);
}

iree_status_t loom_serve_input_decoder_finalize(
    loom_serve_input_decoder_t* decoder, iree_byte_span_t output,
    iree_host_size_t* out_written, bool* out_complete) {
  decoder->finalizing = true;
  char empty_output = 0;
  iree_status_t status = iree_tokenizer_decode_state_finalize(
      decoder->state,
      iree_make_mutable_string_view(
          output.data ? (char*)output.data : &empty_output, output.data_length),
      out_written);
  *out_complete = iree_status_is_ok(status);
  if (iree_status_is_resource_exhausted(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }
  return status;
}
