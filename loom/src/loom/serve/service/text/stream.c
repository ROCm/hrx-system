// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/stream.h"

#include <string.h>

typedef struct loom_serve_text_stream_span_range_t {
  // Source-assigned visible output channel.
  loom_serve_text_chat_output_kind_t kind;
  // First byte in the block's owned storage.
  iree_host_size_t offset;
  // Number of bytes in this span.
  iree_host_size_t length;
} loom_serve_text_stream_span_range_t;

struct loom_serve_text_stream_block_t {
  // Owned bytes, addressed by offsets while the builder grows.
  iree_string_builder_t storage;
  // Ordered source spans; allocation remains reusable across publications.
  loom_serve_text_stream_span_range_t* ranges;
  // Number of complete spans in this block.
  iree_host_size_t count;
  // Allocated span capacity; the source response byte limit bounds growth.
  iree_host_size_t capacity;
  // Final output, including an empty successful or failed completion.
  bool complete;
};

struct loom_serve_text_stream_t {
  // Allocator owning this receipt, its name tail, and output blocks.
  iree_allocator_t host_allocator;
  // Moved request body and copied selectors, released at execution retirement.
  loom_serve_text_request_t request;
  // Caller ownership, consumed only by close on the serialized owner.
  bool caller_open;
  // Execution ownership, consumed after every source/CPU borrower retires.
  bool service_open;
  // Owner-visible delivery credit; workers never access this field.
  bool published;
  // Terminal status visible to the caller, written only on the owner.
  iree_status_t status;
  // Translation failure written by the CPU loan, moved on owner retirement.
  iree_status_t staged_status;
  // CPU-written output, exchanged with delivery only after its loan retires.
  loom_serve_text_stream_block_t staging;
  // Caller-visible block, independently consumable during CPU collection.
  loom_serve_text_stream_block_t delivery;
  // Moved final answer storage, independent of retained canonical history.
  iree_string_builder_t content;
  // Moved final reasoning storage, independent of the model's source VM.
  iree_string_builder_t reasoning;
  // Moved complete tool transaction; no partial calls are observable.
  loom_serve_text_tool_calls_t calls;
  // CPU-written final view, exposed only after successful owner retirement.
  loom_serve_text_stream_result_t result;
};

static void loom_serve_text_stream_block_reset(
    loom_serve_text_stream_block_t* block) {
  iree_string_builder_reset(&block->storage);
  block->count = 0;
  block->complete = false;
}

static void loom_serve_text_stream_block_deinitialize(
    loom_serve_text_stream_block_t* block) {
  iree_allocator_free(block->storage.allocator, block->ranges);
  iree_string_builder_deinitialize(&block->storage);
}

static void loom_serve_text_stream_release(loom_serve_text_stream_t* stream) {
  if (!stream->caller_open && !stream->service_open) {
    iree_status_free(stream->status);
    iree_status_free(stream->staged_status);
    loom_serve_text_stream_block_deinitialize(&stream->staging);
    loom_serve_text_stream_block_deinitialize(&stream->delivery);
    iree_string_builder_deinitialize(&stream->content);
    iree_string_builder_deinitialize(&stream->reasoning);
    loom_serve_text_tool_calls_deinitialize(&stream->calls);
    iree_allocator_free(stream->host_allocator, stream);
  }
}

static bool loom_serve_text_stream_open(void* self) {
  const loom_serve_text_stream_t* stream = self;
  return stream->caller_open;
}

static bool loom_serve_text_stream_blocked(void* self) {
  const loom_serve_text_stream_t* stream = self;
  return stream->published &&
         (stream->staging.count || stream->staging.complete);
}

static iree_status_t loom_serve_text_stream_begin(
    void* self, const loom_serve_text_chat_response_t* response) {
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stream_append(
    void* self, const loom_serve_text_chat_response_t* response,
    const loom_serve_text_chat_output_t* output,
    iree_string_builder_t* scratch) {
  loom_serve_text_stream_t* stream = self;
  loom_serve_text_stream_block_t* block = &stream->staging;
  if (block->count == block->capacity) {
    const iree_host_size_t capacity = block->capacity ? block->capacity * 2 : 4;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc(
        stream->host_allocator, capacity * sizeof(*block->ranges),
        (void**)&block->ranges));
    block->capacity = capacity;
  }
  const loom_serve_text_stream_span_range_t range = {
      .kind = output->kind,
      .offset = iree_string_builder_size(&block->storage),
      .length = output->text.size};
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&block->storage, output->text));
  block->ranges[block->count++] = range;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stream_complete(
    void* self, const loom_serve_text_chat_response_t* response,
    iree_string_builder_t* content, iree_string_builder_t* reasoning,
    loom_serve_text_tool_calls_t* calls, iree_string_view_t finish_reason,
    iree_string_builder_t* scratch) {
  loom_serve_text_stream_t* stream = self;
  const iree_string_view_t model_name = stream->result.response.model_name;
  stream->content = *content;
  iree_string_builder_initialize(content->allocator, content);
  stream->reasoning = *reasoning;
  iree_string_builder_initialize(reasoning->allocator, reasoning);
  stream->calls = *calls;
  loom_serve_text_tool_calls_initialize(calls->storage.allocator, calls);
  stream->result = (loom_serve_text_stream_result_t){
      .response = *response,
      .content = iree_string_builder_view(&stream->content),
      .reasoning = iree_string_builder_view(&stream->reasoning),
      .calls = &stream->calls,
      .finish_reason = finish_reason};
  stream->result.response.model_name = model_name;
  stream->staging.complete = true;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stream_fail(
    void* self, loom_serve_text_chat_flags_t flags, iree_status_t failure) {
  loom_serve_text_stream_t* stream = self;
  loom_serve_text_stream_block_reset(&stream->staging);
  stream->staged_status = failure;
  stream->staging.complete = true;
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stream_publish(void* self, bool* out_ready,
                                                    bool* out_progress) {
  loom_serve_text_stream_t* stream = self;
  *out_progress = false;
  *out_ready = !(stream->staging.count || stream->staging.complete);
  if (!*out_ready && !stream->published) {
    const loom_serve_text_stream_block_t reusable = stream->delivery;
    stream->delivery = stream->staging;
    stream->staging = reusable;
    stream->published = true;
    *out_ready = true;
    *out_progress = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_text_stream_reject(void* self,
                                                   iree_status_t failure) {
  loom_serve_text_stream_t* stream = self;
  stream->status = failure;
  stream->service_open = false;
  iree_string_builder_deinitialize(&stream->request.body);
  loom_serve_text_stream_release(stream);
  return iree_ok_status();
}

static void loom_serve_text_stream_end(void* self, bool complete,
                                       iree_status_code_t code) {
  loom_serve_text_stream_t* stream = self;
  stream->status = stream->staged_status;
  stream->staged_status = iree_ok_status();
  if (iree_status_is_ok(stream->status)) {
    stream->status = iree_status_from_code(code);
  }
  if (!complete) {
    stream->published = false;
  }
  stream->service_open = false;
  iree_string_builder_deinitialize(&stream->request.body);
  loom_serve_text_stream_release(stream);
}

iree_status_t loom_serve_text_stream_allocate(
    iree_string_view_t model_name, loom_serve_text_request_t* request,
    iree_allocator_t host_allocator, loom_serve_text_stream_t** out_stream,
    loom_serve_text_consumer_t* out_consumer) {
  *out_stream = NULL;
  *out_consumer = (loom_serve_text_consumer_t){0};
  if (iree_allocator_is_null(request->body.allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native text input requires an owning builder");
  }
  iree_host_size_t allocation_size = 0;
  iree_host_size_t model_offset = 0, session_offset = 0, checkpoint_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_text_stream_t), &allocation_size,
      IREE_STRUCT_FIELD(model_name.size, char, &model_offset),
      IREE_STRUCT_FIELD(request->session.size, char, &session_offset),
      IREE_STRUCT_FIELD(request->checkpoint.size, char, &checkpoint_offset)));
  loom_serve_text_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&stream));
  stream->host_allocator = host_allocator;
  stream->caller_open = true;
  stream->service_open = true;
  stream->request = *request;
  char* storage = (char*)stream;
  memcpy(storage + model_offset, model_name.data, model_name.size);
  if (request->session.size) {
    memcpy(storage + session_offset, request->session.data,
           request->session.size);
  }
  if (request->checkpoint.size) {
    memcpy(storage + checkpoint_offset, request->checkpoint.data,
           request->checkpoint.size);
  }
  stream->result.response.model_name =
      iree_make_string_view(storage + model_offset, model_name.size);
  stream->request.session =
      iree_make_string_view(storage + session_offset, request->session.size);
  stream->request.checkpoint = iree_make_string_view(
      storage + checkpoint_offset, request->checkpoint.size);
  iree_string_builder_initialize(host_allocator, &stream->staging.storage);
  iree_string_builder_initialize(host_allocator, &stream->delivery.storage);
  iree_string_builder_initialize(host_allocator, &stream->content);
  iree_string_builder_initialize(host_allocator, &stream->reasoning);
  loom_serve_text_tool_calls_initialize(host_allocator, &stream->calls);
  *request = (loom_serve_text_request_t){0};
  *out_stream = stream;
  static const loom_serve_text_consumer_vtable_t vtable = {
      .is_open = loom_serve_text_stream_open,
      .is_blocked = loom_serve_text_stream_blocked,
      .begin = loom_serve_text_stream_begin,
      .append = loom_serve_text_stream_append,
      .complete = loom_serve_text_stream_complete,
      .fail = loom_serve_text_stream_fail,
      .publish = loom_serve_text_stream_publish,
      .reject = loom_serve_text_stream_reject,
      .end = loom_serve_text_stream_end};
  *out_consumer =
      (loom_serve_text_consumer_t){.self = stream, .vtable = &vtable};
  return iree_ok_status();
}

const loom_serve_text_request_t* loom_serve_text_stream_request(
    const loom_serve_text_stream_t* stream) {
  return &stream->request;
}

const loom_serve_text_stream_block_t* loom_serve_text_stream_peek(
    const loom_serve_text_stream_t* stream) {
  return stream->published ? &stream->delivery : NULL;
}

iree_host_size_t loom_serve_text_stream_block_count(
    const loom_serve_text_stream_block_t* block) {
  return block->count;
}

loom_serve_text_stream_span_t loom_serve_text_stream_block_at(
    const loom_serve_text_stream_block_t* block, iree_host_size_t index) {
  const loom_serve_text_stream_span_range_t range = block->ranges[index];
  return (loom_serve_text_stream_span_t){
      .kind = range.kind,
      .text = iree_make_string_view(
          iree_string_builder_buffer(&block->storage) + range.offset,
          range.length)};
}

void loom_serve_text_stream_consume(loom_serve_text_stream_t* stream) {
  loom_serve_text_stream_block_reset(&stream->delivery);
  stream->published = false;
}

bool loom_serve_text_stream_is_complete(
    const loom_serve_text_stream_t* stream) {
  return !stream->service_open && !stream->published;
}

iree_status_t loom_serve_text_stream_status(
    const loom_serve_text_stream_t* stream) {
  return iree_status_clone(stream->status);
}

const loom_serve_text_stream_result_t* loom_serve_text_stream_result(
    const loom_serve_text_stream_t* stream) {
  return !stream->service_open && iree_status_is_ok(stream->status)
             ? &stream->result
             : NULL;
}

void loom_serve_text_stream_close(loom_serve_text_stream_t* stream) {
  stream->caller_open = false;
  stream->published = false;
  loom_serve_text_stream_release(stream);
}
