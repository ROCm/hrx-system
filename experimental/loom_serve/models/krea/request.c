// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/models/krea/request.h"

#include <math.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "iree/base/string_builder.h"
#include "iree/tokenizer/vocab/vocab.h"

struct loom_serve_krea2_prompt_t {
  // Allocator owning this object and its trailing native token IDs.
  iree_allocator_t allocator;
  // Actual retained count of the combined framing prefix and prompt.
  uint32_t token_count;
  // Validated live suffix IDs, placed at the end of the selected text extent.
  int32_t suffix[5];
  // Retained prefix+prompt IDs in native byte order, without padding or suffix.
  int32_t token_ids[];
};

struct loom_serve_krea2_request_t {
  // Allocator owning this object and its trailing slab.
  iree_allocator_t allocator;
  // Immutable views into the trailing slab, in the source command's order.
  iree_const_byte_span_t inputs[LOOM_SERVE_KREA2_INPUT_COUNT];
};

static const char kPrefix[] =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, "
    "texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n";
static const char kSuffix[] = "<|im_end|>\n<|im_start|>assistant\n";

// Pull only the requested prefix of the encoded stream. Finalize is
// destructive, so its independent pending bound, not the remaining destination
// capacity, determines scratch size when the entire text fits in the input
// feed.
static iree_status_t krea2_encode_prefix(const iree_tokenizer_t* tokenizer,
                                         iree_string_view_t text,
                                         iree_host_size_t capacity,
                                         int32_t* output,
                                         iree_host_size_t* out_count,
                                         iree_allocator_t allocator) {
  *out_count = 0;
  iree_host_size_t state_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_tokenizer_encode_state_calculate_size(tokenizer, &state_size));
  const iree_host_size_t transform_size =
      iree_tokenizer_transform_buffer_oneshot_size(text.size);
  if (!transform_size || state_size > SIZE_MAX - transform_size) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "prompt tokenizer storage size overflow");
  }
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, state_size + transform_size, (void**)&storage));
  iree_tokenizer_encode_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_encode_state_initialize(
      tokenizer, iree_make_byte_span(storage, state_size),
      iree_make_byte_span(storage + state_size, transform_size),
      (iree_tokenizer_offset_run_list_t){0},
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START |
          IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS,
      &state);
  iree_host_size_t count = 0;
  while (iree_status_is_ok(status) && text.size && count < capacity) {
    iree_host_size_t consumed = 0;
    iree_host_size_t produced = 0;
    status = iree_tokenizer_encode_state_feed(
        state, text,
        iree_tokenizer_make_token_output(output + count, NULL, NULL,
                                         capacity - count),
        &consumed, &produced);
    text = iree_string_view_substr(text, consumed, IREE_STRING_VIEW_NPOS);
    count += produced;
  }
  int32_t* pending = NULL;
  if (iree_status_is_ok(status) && count < capacity) {
    // Finalize requires non-null output storage even for a zero token bound.
    const iree_host_size_t bound =
        iree_max(1, iree_tokenizer_encode_state_pending_token_bound(state));
    status = iree_allocator_malloc_array(allocator, bound, sizeof(*pending),
                                         (void**)&pending);
    iree_host_size_t produced = 0;
    if (iree_status_is_ok(status)) {
      status = iree_tokenizer_encode_state_finalize(
          state, iree_tokenizer_make_token_output(pending, NULL, NULL, bound),
          &produced);
    }
    if (iree_status_is_ok(status)) {
      produced = iree_min(produced, capacity - count);
      if (produced) {
        memcpy(output + count, pending, produced * sizeof(*output));
      }
      count += produced;
    }
  }
  if (state) {
    iree_tokenizer_encode_state_deinitialize(state);
  }
  iree_allocator_free(allocator, pending);
  iree_allocator_free(allocator, storage);
  if (iree_status_is_ok(status)) {
    *out_count = count;
  }
  return status;
}

static iree_status_t krea2_encode_prompt(const iree_tokenizer_t* tokenizer,
                                         iree_string_view_t text,
                                         uint32_t capacity,
                                         loom_serve_krea2_prompt_t* prompt) {
  const iree_allocator_t allocator = prompt->allocator;
  int32_t prefix[35], suffix[6];
  iree_host_size_t prefix_count = 0, suffix_count = 0;
  IREE_RETURN_IF_ERROR(krea2_encode_prefix(
      tokenizer, iree_make_cstring_view(kPrefix), IREE_ARRAYSIZE(prefix),
      prefix, &prefix_count, allocator));
  IREE_RETURN_IF_ERROR(krea2_encode_prefix(
      tokenizer, iree_make_cstring_view(kSuffix), IREE_ARRAYSIZE(suffix),
      suffix, &suffix_count, allocator));
  if (prefix_count != 34 || suffix_count != 5) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokenizer does not match the 34/5 Krea template");
  }
  const int32_t padding = iree_tokenizer_vocab_lookup(
      iree_tokenizer_vocab(tokenizer), IREE_SV("<|endoftext|>"));
  if (padding != 151643) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tokenizer has no Krea padding token 151643");
  }
  iree_string_builder_t framed;
  iree_string_builder_initialize(allocator, &framed);
  iree_status_t status = iree_string_builder_append_cstring(&framed, kPrefix);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_string(&framed, text);
  }
  iree_host_size_t count = 0;
  if (iree_status_is_ok(status)) {
    status =
        krea2_encode_prefix(tokenizer, iree_string_builder_view(&framed),
                            capacity, prompt->token_ids, &count, allocator);
  }
  iree_string_builder_deinitialize(&framed);
  IREE_RETURN_IF_ERROR(status);
  for (iree_host_size_t i = 0; i < count + 5; ++i) {
    const int32_t id = i < count ? prompt->token_ids[i] : suffix[i - count];
    if (id < 0 || id >= 151936) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "prepared token %zu has no encoder embedding: %d",
                              i, id);
    }
  }
  prompt->token_count = (uint32_t)count;
  memcpy(prompt->suffix, suffix, sizeof(prompt->suffix));
  return iree_ok_status();
}

iree_status_t loom_serve_krea2_prompt_create(
    const iree_tokenizer_t* tokenizer, uint32_t maximum_text_tokens,
    iree_string_view_t text, loom_serve_krea2_prompt_t** out_prompt,
    iree_allocator_t host_allocator) {
  *out_prompt = NULL;
  if (!maximum_text_tokens || maximum_text_tokens > 65536 ||
      maximum_text_tokens % 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected supported 16-aligned maximum text extent");
  }
  const uint32_t capacity = maximum_text_tokens + 29;
  loom_serve_krea2_prompt_t* prompt = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*prompt) + capacity * sizeof(prompt->token_ids[0]),
      (void**)&prompt));
  prompt->allocator = host_allocator;
  iree_status_t status = krea2_encode_prompt(tokenizer, text, capacity, prompt);
  if (iree_status_is_ok(status)) {
    *out_prompt = prompt;
  } else {
    loom_serve_krea2_prompt_destroy(prompt);
  }
  return status;
}

uint32_t loom_serve_krea2_prompt_token_count(
    const loom_serve_krea2_prompt_t* prompt) {
  return prompt->token_count;
}

void loom_serve_krea2_prompt_destroy(loom_serve_krea2_prompt_t* prompt) {
  if (prompt) {
    iree_allocator_free(prompt->allocator, prompt);
  }
}

iree_status_t loom_serve_krea2_request_measure(
    uint32_t height, uint32_t width, uint32_t text_tokens,
    iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT]) {
  if (!height || height > 8192 || height % 16 || !width || width > 8192 ||
      width % 16 || !text_tokens || text_tokens > 65536 || text_tokens % 16) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected supported 16-aligned pixel/text extents");
  }
  const uint32_t images = (height / 16) * (width / 16);
  const uint32_t texts = text_tokens;
  const uint32_t tokens = images + texts;
  if (images < 16 || images % 16 || tokens > 65536) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "image patches must be a multiple of 16; combined "
                            "text/image extent must fit 65536 tokens");
  }
  const iree_host_size_t lengths[LOOM_SERVE_KREA2_INPUT_COUNT] = {
      16, (texts + 34) * 4};
  memcpy(sizes, lengths, sizeof(lengths));
  return iree_ok_status();
}

iree_status_t loom_serve_krea2_request_create(
    const loom_serve_krea2_prompt_t* prompt,
    loom_serve_krea2_request_options_t options,
    loom_serve_krea2_request_t** out_request, iree_allocator_t host_allocator) {
  *out_request = NULL;
  if (!isfinite(options.strength)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "adapter strength must be finite");
  }
  iree_host_size_t sizes[LOOM_SERVE_KREA2_INPUT_COUNT];
  IREE_RETURN_IF_ERROR(loom_serve_krea2_request_measure(
      options.height, options.width, options.text_tokens, sizes));
  if (prompt->token_count > options.text_tokens + 29) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "prepared prompt has %u tokens; text extent %u "
                            "retains at most %u",
                            prompt->token_count, options.text_tokens,
                            options.text_tokens + 29);
  }
  iree_host_size_t total =
      iree_host_align(sizeof(loom_serve_krea2_request_t), 16);
  for (int i = 0; i < LOOM_SERVE_KREA2_INPUT_COUNT; ++i) {
    total += iree_host_align(sizes[i], 16);
  }
  loom_serve_krea2_request_t* request = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total, (void**)&request));
  request->allocator = host_allocator;
  uint8_t* storage = (uint8_t*)request +
                     iree_host_align(sizeof(loom_serve_krea2_request_t), 16);
  for (int i = 0; i < LOOM_SERVE_KREA2_INPUT_COUNT; ++i) {
    request->inputs[i] = iree_make_const_byte_span(storage, sizes[i]);
    storage += iree_host_align(sizes[i], 16);
  }
  uint8_t* header =
      (uint8_t*)request->inputs[LOOM_SERVE_KREA2_INPUT_HEADER].data;
  iree_unaligned_store_le_u64(header, options.seed);
  iree_unaligned_store_le_u32(header + 8, prompt->token_count);
  iree_unaligned_store_le_f32(header + 12, options.strength);
  uint8_t* ids =
      (uint8_t*)request->inputs[LOOM_SERVE_KREA2_INPUT_TOKEN_IDS].data;
  for (uint32_t i = 0; i < prompt->token_count; ++i) {
    iree_unaligned_store_le_u32(ids + i * 4, (uint32_t)prompt->token_ids[i]);
  }
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(prompt->suffix); ++i) {
    iree_unaligned_store_le_u32(ids + (prompt->token_count + i) * 4,
                                (uint32_t)prompt->suffix[i]);
  }
  *out_request = request;
  return iree_ok_status();
}

void loom_serve_krea2_request_destroy(loom_serve_krea2_request_t* request) {
  if (request) {
    iree_allocator_free(request->allocator, request);
  }
}

iree_const_byte_span_t loom_serve_krea2_request_input(
    const loom_serve_krea2_request_t* request, loom_serve_krea2_input_t input) {
  return request->inputs[input];
}
