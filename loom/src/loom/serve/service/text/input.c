// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/input.h"

#include <string.h>

static iree_vm_buffer_t* loom_serve_text_input_history(
    const loom_serve_text_chat_t* chat, loom_serve_text_input_prefix_t prefix) {
  return prefix.completion && prefix.completion->reusable &&
                 loom_serve_text_chat_completion_is_prefix(prefix.completion,
                                                           chat)
             ? prefix.completion->storage
             : NULL;
}

void loom_serve_text_input_deinitialize(loom_serve_text_input_t* input,
                                        iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, input->media_spans);
  iree_allocator_free(host_allocator, input->tokens);
  iree_vm_buffer_release(input->history);
  memset(input, 0, sizeof(*input));
}

bool loom_serve_text_input_is_current(const loom_serve_text_input_t* input,
                                      const loom_serve_text_chat_t* chat,
                                      loom_serve_text_input_prefix_t prefix) {
  return input->tokens &&
         input->history == loom_serve_text_input_history(chat, prefix);
}

iree_status_t loom_serve_text_input_prepare(
    const loom_serve_text_chat_t* chat, loom_serve_text_input_prefix_t prefix,
    bool require_prefix, iree_host_size_t context_capacity, int32_t* scratch,
    loom_serve_text_input_t* input, iree_allocator_t host_allocator) {
  iree_vm_buffer_t* history = loom_serve_text_input_history(chat, prefix);
  const bool retained = history != NULL;
  if (require_prefix && !retained) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "request history does not extend the named checkpoint");
  }
  if (input->tokens && input->history == history) {
    return iree_ok_status();
  }
  const iree_host_size_t retained_count = retained ? prefix.position : 0;
  const iree_host_size_t prefix_count = retained ? 1 : 0;
  if (retained) {
    scratch[0] = prefix.token;
  }
  iree_host_size_t count = 0;
  loom_serve_text_media_span_t* media_spans = NULL;
  iree_host_size_t media_span_count = 0;
  const iree_host_size_t prompt_offset =
      retained ? prefix.completion->transcript.size : 0;
  IREE_RETURN_IF_ERROR(loom_serve_text_chat_prepare_input(
      chat, prompt_offset, LOOM_SERVE_TEXT_CHAT_INPUT_RENDERED,
      retained ? prefix.boundary : LOOM_SERVE_TEXT_CHAT_BOUNDARY_FRESH,
      context_capacity - prefix_count, prefix_count, host_allocator,
      scratch + prefix_count, &count, &media_spans, &media_span_count));
  count += prefix_count;
  if (count + chat->max_tokens - 1 > context_capacity - retained_count) {
    iree_allocator_free(host_allocator, media_spans);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "request input and output exceed remaining %zu-token context; compact "
        "history or lower max_tokens",
        context_capacity);
  }
  int32_t* tokens = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, iree_max(count, 1), sizeof(*tokens), (void**)&tokens);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, media_spans);
    return status;
  }
  memcpy(tokens, scratch, count * sizeof(*tokens));
  iree_vm_buffer_retain(history);
  loom_serve_text_input_deinitialize(input, host_allocator);
  *input = (loom_serve_text_input_t){.tokens = tokens,
                                     .count = count,
                                     .media_spans = media_spans,
                                     .media_span_count = media_span_count,
                                     .history = history,
                                     .retained_count = retained_count};
  return iree_ok_status();
}
