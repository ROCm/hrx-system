// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CONSUMER_H_
#define LOOM_SERVE_SERVICE_TEXT_CONSUMER_H_

#include "loom/serve/service/text/chat.h"

#ifdef __cplusplus
extern "C" {
#endif

// Operations for one request's input lifetime and bounded output delivery.
// CPU encoding may overlap owner interest queries and therefore never reads
// interest or delivery-slot fields. No operation waits or submits model work.
typedef struct loom_serve_text_consumer_vtable_t {
  // Owner-only interest query; false cannot become true again.
  bool (*is_open)(void* self);
  // Owner-only observation between CPU loans: staged output is waiting behind
  // an occupied delivery slot. An empty stage still permits another epoch.
  bool (*is_blocked)(void* self);
  // Starts admitted output on the owner; metadata is borrowed for this call.
  iree_status_t (*begin)(void* self,
                         const loom_serve_text_chat_response_t* response);
  // Encodes one source-classified span on the finite CPU output loan. Scratch
  // is shared by that loan; no source view or scratch borrow escapes this call.
  iree_status_t (*append)(void* self,
                          const loom_serve_text_chat_response_t* response,
                          const loom_serve_text_chat_output_t* output,
                          iree_string_builder_t* scratch);
  // Encodes the complete successful source transaction on the same CPU loan.
  // May move content, reasoning, and calls, leaving their owners initialized
  // and empty. Canonical retained history is independent of these outputs.
  // finish_reason is a process-lifetime classification, safe to retain.
  iree_status_t (*complete)(void* self,
                            const loom_serve_text_chat_response_t* response,
                            iree_string_builder_t* content,
                            iree_string_builder_t* reasoning,
                            loom_serve_text_tool_calls_t* calls,
                            iree_string_view_t finish_reason,
                            iree_string_builder_t* scratch);
  // Consumes translation failure on the CPU loan, discarding partial staged
  // output and retaining a terminal result. Only encoding allocation can fail.
  iree_status_t (*fail)(void* self, loom_serve_text_chat_flags_t flags,
                        iree_status_t failure);
  // Owner-only publication after CPU retirement. ready means the stage is
  // empty and another epoch may run; progress means a stage was transferred.
  // Output pressure returns OK/false without changing either slot.
  iree_status_t (*publish)(void* self, bool* out_ready, bool* out_progress);
  // Consumes a rejected receipt and its failure after preparation borrowers
  // retire. Error-response allocation may fail; ownership is always consumed.
  iree_status_t (*reject)(void* self, iree_status_t failure);
  // Consumes the receipt after request storage and every borrower retire.
  // complete means final output was delivered, including a translated error;
  // false discards undelivered output. code preserves failure/cancellation.
  void (*end)(void* self, bool complete, iree_status_code_t code);
} loom_serve_text_consumer_vtable_t;

// Compact movable ownership. Immutable operations are shared by each delivery
// implementation instead of copied into every queue entry and execution row.
// Pending and admitted requests move this value without moving the receipt at
// self. The service retains it through every accepted CPU/device borrow and
// consumes it exactly once by reject or end.
typedef struct loom_serve_text_consumer_t {
  // Owned receipt keeping borrowed request body/name/tool/image views alive.
  void* self;
  // Process-lifetime operations; nonnull exactly when self is nonnull.
  const loom_serve_text_consumer_vtable_t* vtable;
} loom_serve_text_consumer_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CONSUMER_H_
