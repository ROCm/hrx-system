// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_
#define LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_

#include "iree/async/proactor.h"
#include "iree/task/executor.h"
#include "loom/serve/service/finite/component.h"
#include "loom/serve/service/text/projected_media.h"
#include "loom/serve/service/text/readiness.h"

#ifdef __cplusplus
extern "C" {
#endif

// One selected text admission candidate's serial image projector. This owns a
// separate CPU/device publication pair so ordinary text epochs can progress
// while image preparation or execution is outstanding.
typedef struct loom_serve_text_media_projector_t
    loom_serve_text_media_projector_t;

// Allocates a dormant projector around one borrowed deployment component. The
// executor, proactor and component outlive the projector. No component demand,
// image decode, result table or residency admission begins here.
iree_status_t loom_serve_text_media_projector_allocate(
    iree_task_executor_t* executor, iree_async_proactor_t* proactor,
    loom_serve_finite_component_t* component, iree_allocator_t host_allocator,
    loom_serve_text_media_projector_t** out_projector);

// Installs the shared owner observer before work begins and detaches it only
// after every accepted callback has been consumed.
void loom_serve_text_media_projector_set_observer(
    loom_serve_text_media_projector_t* projector,
    loom_serve_wake_callback_t observer);

// Selects all complete source media spans at or after |input_origin|. The input
// is consumed synchronously; its image indices, spans, and metadata offsets are
// copied before return. The chat remains borrowed until completion is taken.
// A prefix covering every image leaves the projector dormant without preparing
// the component. Exactly one candidate may otherwise be active; the caller
// prevents admission overtaking until it is taken.
iree_status_t loom_serve_text_media_projector_begin(
    loom_serve_text_media_projector_t* projector,
    const loom_serve_text_chat_t* chat, const loom_serve_text_input_t* input,
    iree_host_size_t input_origin);

// Marks the active candidate cancelled. Shared component preparation continues;
// accepted CPU/device work and failed-model retirement are joined before the
// borrowed request may be removed. An idle projector is inert.
void loom_serve_text_media_projector_cancel(
    loom_serve_text_media_projector_t* projector);

// Advances immediate owner work. shared_activity distinguishes temporary
// capacity pressure from an impossible idle deployment. Request/source errors
// are retained for take instead of failing the parent text service.
bool loom_serve_text_media_projector_advance(
    loom_serve_text_media_projector_t* projector, bool shared_activity);

// Whether a candidate still owns this projector, whether an accepted callback
// still needs collection, and whether its terminal result is ready to take.
bool loom_serve_text_media_projector_is_active(
    const loom_serve_text_media_projector_t* projector);
bool loom_serve_text_media_projector_is_pending(
    const loom_serve_text_media_projector_t* projector);
bool loom_serve_text_media_projector_is_complete(
    const loom_serve_text_media_projector_t* projector);

// Consumes one terminal candidate. Success transfers projected rows unless the
// candidate was cancelled; failure transfers its request-local diagnostic.
// The projector returns to dormant state and no longer borrows chat or input.
iree_status_t loom_serve_text_media_projector_take(
    loom_serve_text_media_projector_t* projector, bool* out_cancelled,
    loom_serve_text_projected_media_t* out_media);

// Releases a dormant projector. NULL is accepted.
void loom_serve_text_media_projector_free(
    loom_serve_text_media_projector_t* projector);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_MEDIA_PROJECTOR_H_
