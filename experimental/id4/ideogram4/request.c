// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/request.h"

#include <string.h>

#include "iree/base/internal/json.h"

static const char id4_ideogram4_qwen_prompt_prefix[] = "<|im_start|>user\n";
static const char id4_ideogram4_qwen_prompt_suffix[] =
    "<|im_end|>\n<|im_start|>assistant\n";

static iree_status_t id4_ideogram4_request_count_member(
    void* user_data, iree_string_view_t key, iree_string_view_t value) {
  (void)key;
  (void)value;
  iree_host_size_t* count = (iree_host_size_t*)user_data;
  ++*count;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_validate_lowering_options(
    const id4_ideogram4_qwen_lowering_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering options structure size "
                            "%" PRIhsz " is smaller than expected %" PRIhsz,
                            options->structure_size, sizeof(*options));
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Qwen request lowering extension structures are not supported");
  }
  if (!options->tokenizer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering tokenizer is required");
  }
  if (!options->request) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering request is required");
  }
  if (iree_string_view_is_empty(options->request->qwen_prompt)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering wrapped prompt is empty");
  }
  if (options->max_token_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Qwen request lowering max token count is zero");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_copy_string(
    iree_string_view_t value, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  *out_value = iree_string_view_empty();
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, value.size, (void**)&storage));
  memcpy(storage, value.data, value.size);
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_wrap_qwen_prompt(
    iree_string_view_t raw_prompt_json, iree_allocator_t host_allocator,
    iree_string_view_t* out_qwen_prompt) {
  *out_qwen_prompt = iree_string_view_empty();
  const iree_host_size_t prefix_length =
      sizeof(id4_ideogram4_qwen_prompt_prefix) - 1;
  const iree_host_size_t suffix_length =
      sizeof(id4_ideogram4_qwen_prompt_suffix) - 1;
  if (raw_prompt_json.size > IREE_HOST_SIZE_MAX - prefix_length ||
      raw_prompt_json.size + prefix_length >
          IREE_HOST_SIZE_MAX - suffix_length) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "Ideogram 4 Qwen prompt is too large");
  }
  const iree_host_size_t prompt_length =
      prefix_length + raw_prompt_json.size + suffix_length;
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, prompt_length, (void**)&storage));
  memcpy(storage, id4_ideogram4_qwen_prompt_prefix, prefix_length);
  memcpy(storage + prefix_length, raw_prompt_json.data, raw_prompt_json.size);
  memcpy(storage + prefix_length + raw_prompt_json.size,
         id4_ideogram4_qwen_prompt_suffix, suffix_length);
  *out_qwen_prompt = iree_make_string_view(storage, prompt_length);
  return iree_ok_status();
}

iree_status_t id4_ideogram4_request_parse_json(
    iree_string_view_t json, iree_allocator_t host_allocator,
    id4_ideogram4_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(out_request);
  memset(out_request, 0, sizeof(*out_request));

  iree_string_view_t remaining = json;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  iree_string_view_t object = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&remaining, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&remaining));
  if (!iree_string_view_is_empty(remaining)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing data after Ideogram 4 request object");
  }

  iree_host_size_t member_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object(
      object, id4_ideogram4_request_count_member, &member_count));
  if (member_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Ideogram 4 request object is empty");
  }

  iree_status_t status = id4_ideogram4_request_copy_string(
      object, host_allocator, &out_request->raw_prompt_json);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_wrap_qwen_prompt(
        out_request->raw_prompt_json, host_allocator,
        &out_request->qwen_prompt);
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_request_deinitialize(out_request, host_allocator);
  }
  return status;
}

void id4_ideogram4_request_deinitialize(id4_ideogram4_request_t* request,
                                        iree_allocator_t host_allocator) {
  if (!request) return;
  iree_allocator_free(host_allocator, (void*)request->qwen_prompt.data);
  iree_allocator_free(host_allocator, (void*)request->raw_prompt_json.data);
  memset(request, 0, sizeof(*request));
}

static iree_status_t id4_ideogram4_request_allocate_qwen_inputs(
    uint32_t token_count, iree_allocator_t host_allocator,
    id4_ideogram4_qwen_inputs_t* out_inputs) {
  out_inputs->token_count = token_count;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_count, sizeof(out_inputs->token_ids[0]),
      (void**)&out_inputs->token_ids));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, token_count, sizeof(out_inputs->token_weights[0]),
      (void**)&out_inputs->token_weights));
  const iree_host_size_t attention_element_count =
      token_count * (iree_host_size_t)token_count;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, attention_element_count,
                                  sizeof(out_inputs->attention_mask[0]),
                                  (void**)&out_inputs->attention_mask));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_request_encode_qwen_tokens(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator,
    iree_tokenizer_token_id_t** out_token_storage, uint32_t* out_token_count) {
  *out_token_storage = NULL;
  *out_token_count = 0;

  iree_tokenizer_token_id_t* token_storage = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, options->max_token_count, sizeof(token_storage[0]),
      (void**)&token_storage);
  iree_host_size_t token_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_tokenizer_encode(
        options->tokenizer, options->request->qwen_prompt,
        options->tokenizer_flags,
        iree_tokenizer_make_token_output(token_storage, NULL, NULL,
                                         options->max_token_count),
        host_allocator, &token_count);
  }
  if (iree_status_is_ok(status) && token_count == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Ideogram 4 request prompt produced no tokens");
  }
  if (iree_status_is_ok(status) && token_count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Ideogram 4 request token count %" PRIhsz
                              " exceeds uint32 range",
                              token_count);
  }
  if (iree_status_is_ok(status)) {
    *out_token_storage = token_storage;
    *out_token_count = (uint32_t)token_count;
  } else {
    iree_allocator_free(host_allocator, token_storage);
  }
  return status;
}

iree_status_t id4_ideogram4_request_lower_qwen_inputs(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, id4_ideogram4_qwen_inputs_t* out_inputs) {
  IREE_ASSERT_ARGUMENT(out_inputs);
  memset(out_inputs, 0, sizeof(*out_inputs));
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_lowering_options(options));

  iree_tokenizer_token_id_t* token_storage = NULL;
  uint32_t token_count = 0;
  iree_status_t status = id4_ideogram4_request_encode_qwen_tokens(
      options, host_allocator, &token_storage, &token_count);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_request_allocate_qwen_inputs(
        token_count, host_allocator, out_inputs);
  }
  if (iree_status_is_ok(status)) {
    memcpy(out_inputs->token_ids, token_storage,
           token_count * sizeof(out_inputs->token_ids[0]));
    for (iree_host_size_t i = 0; i < token_count; ++i) {
      out_inputs->token_weights[i] = 1.0f;
    }
    const iree_host_size_t attention_element_count =
        token_count * (iree_host_size_t)token_count;
    for (iree_host_size_t i = 0; i < attention_element_count; ++i) {
      out_inputs->attention_mask[i] = 0.0f;
    }
  } else {
    id4_ideogram4_qwen_inputs_deinitialize(out_inputs, host_allocator);
  }
  iree_allocator_free(host_allocator, token_storage);
  return status;
}

iree_status_t id4_ideogram4_request_count_qwen_tokens(
    const id4_ideogram4_qwen_lowering_options_t* options,
    iree_allocator_t host_allocator, uint32_t* out_token_count) {
  IREE_ASSERT_ARGUMENT(out_token_count);
  *out_token_count = 0;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_request_validate_lowering_options(options));

  iree_tokenizer_token_id_t* token_storage = NULL;
  iree_status_t status = id4_ideogram4_request_encode_qwen_tokens(
      options, host_allocator, &token_storage, out_token_count);
  iree_allocator_free(host_allocator, token_storage);
  return status;
}

void id4_ideogram4_qwen_inputs_deinitialize(id4_ideogram4_qwen_inputs_t* inputs,
                                            iree_allocator_t host_allocator) {
  if (!inputs) return;
  iree_allocator_free(host_allocator, inputs->attention_mask);
  iree_allocator_free(host_allocator, inputs->token_weights);
  iree_allocator_free(host_allocator, inputs->token_ids);
  memset(inputs, 0, sizeof(*inputs));
}
