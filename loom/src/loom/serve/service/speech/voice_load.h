// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_VOICE_LOAD_H_
#define LOOM_SERVE_SERVICE_SPEECH_VOICE_LOAD_H_

#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/speech/service.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers original model-card recordings through the ordinary native upload
// lifecycle. File reads run on preparation workers; the serialized owner drives
// service advancement. At most one original recording is retained at a time.
typedef struct loom_serve_speech_voice_load_t loom_serve_speech_voice_load_t;

// Captures an already-resolved source namespace and borrows host/service
// through completion. Installs observer on the service; the caller must not
// independently advance or publish that service until this operation completes.
// No I/O or asynchronous work is accepted here. A card with no voices returns
// NULL.
iree_status_t loom_serve_speech_voice_load_allocate(
    loom_serve_host_t* host, const loom_serve_sources_t* sources,
    loom_serve_speech_service_t* service, loom_serve_wake_callback_t observer,
    iree_allocator_t host_allocator, loom_serve_speech_voice_load_t** out_load);

// Advances one bounded owner turn. Cancellation is irreversible and stops the
// service, joining accepted reads, CPU preparation, and device work before
// reporting completion. Successful completion leaves every declared ID ready
// and transfers service advancement back to the caller. NULL is complete.
iree_status_t loom_serve_speech_voice_load_advance(
    loom_serve_speech_voice_load_t* load, bool cancel, bool* out_complete);

// Releases a completed operation. The service and host remain caller-owned.
void loom_serve_speech_voice_load_free(loom_serve_speech_voice_load_t* load);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_VOICE_LOAD_H_
