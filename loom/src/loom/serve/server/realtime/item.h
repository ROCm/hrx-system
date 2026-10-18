// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVER_REALTIME_ITEM_H_
#define LOOM_SERVE_SERVER_REALTIME_ITEM_H_

#include "loom/serve/server/realtime/conversation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Decodes a client item into reserved, owned conversation storage. Message
// text, function calls, and free-text function results have disjoint field
// contracts. Client status is descriptive and has no effect on native history;
// accepted input is complete. Success does not insert history; failure releases
// the reservation and leaves out_item NULL.
iree_status_t loom_serve_realtime_item_parse(
    loom_serve_realtime_conversation_t* conversation, iree_string_view_t object,
    loom_serve_realtime_item_t** out_item);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVER_REALTIME_ITEM_H_
