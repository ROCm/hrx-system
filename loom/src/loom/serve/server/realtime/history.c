// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/realtime/history.h"

void loom_serve_realtime_history_initialize(
    loom_serve_registry_t* registry, loom_serve_realtime_binding_t binding,
    iree_allocator_t host_allocator, loom_serve_realtime_history_t* history) {
  *history =
      (loom_serve_realtime_history_t){.registry = registry, .binding = binding};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    iree_string_builder_initialize(host_allocator, &history->pins[i].name);
    iree_string_builder_initialize(host_allocator, &history->pins[i].session);
  }
}

iree_status_t loom_serve_realtime_history_capture(
    loom_serve_realtime_history_t* history, iree_string_view_t session,
    uint64_t through, bool* accepted) {
  *accepted = false;
  loom_serve_realtime_history_pin_t* empty = NULL;
  loom_serve_realtime_history_pin_t* oldest = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    loom_serve_realtime_history_pin_t* pin = &history->pins[i];
    if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE ||
        pin->phase == LOOM_SERVE_REALTIME_HISTORY_RELEASE) {
      return iree_ok_status();
    }
    if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_EMPTY) {
      empty = pin;
    } else if (!oldest || pin->through < oldest->through) {
      oldest = pin;
    }
  }
  if (!empty) {
    oldest->phase = LOOM_SERVE_REALTIME_HISTORY_RELEASE;
    return iree_ok_status();
  }
  iree_string_builder_reset(&empty->name);
  iree_string_builder_reset(&empty->session);
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(&empty->name, "realtime-"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&empty->name, session));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_string(&empty->session, session));
  empty->through = through;
  empty->invalidated = false;
  empty->phase = LOOM_SERVE_REALTIME_HISTORY_CAPTURE;
  *accepted = true;
  return iree_ok_status();
}

iree_string_view_t loom_serve_realtime_history_checkpoint(
    const loom_serve_realtime_history_t* history) {
  const loom_serve_realtime_history_pin_t* latest = NULL;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    const loom_serve_realtime_history_pin_t* pin = &history->pins[i];
    if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_PINNED && !pin->invalidated &&
        (!latest || pin->through > latest->through)) {
      latest = pin;
    }
  }
  return latest ? iree_string_builder_view(&latest->name)
                : iree_string_view_empty();
}

void loom_serve_realtime_history_invalidate(
    loom_serve_realtime_history_t* history, uint64_t sequence) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    loom_serve_realtime_history_pin_t* pin = &history->pins[i];
    if (pin->through >= sequence) {
      pin->invalidated = true;
      if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_PINNED) {
        pin->phase = LOOM_SERVE_REALTIME_HISTORY_RELEASE;
      }
    }
  }
}

iree_status_t loom_serve_realtime_history_advance(
    loom_serve_realtime_history_t* history, bool stopping, bool* out_progress,
    bool* out_cleanup_failed) {
  *out_cleanup_failed = false;
  if (stopping) {
    loom_serve_realtime_history_invalidate(history, 0);
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(history->pins) && iree_status_is_ok(status); ++i) {
    loom_serve_realtime_history_pin_t* pin = &history->pins[i];
    if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_EMPTY) {
      continue;
    }
    if (!loom_serve_realtime_binding_present(history->registry,
                                             &history->binding)) {
      // Original model teardown owns its accepted resources. A namesake is
      // not evidence that an old checkpoint may be released through it.
      loom_serve_text_checkpoint_control_close(pin->control);
      pin->control = NULL;
      pin->phase = LOOM_SERVE_REALTIME_HISTORY_EMPTY;
      *out_progress = true;
      continue;
    }
    if (pin->control) {
      if (!loom_serve_text_checkpoint_control_is_complete(pin->control)) {
        continue;
      }
      status = loom_serve_text_checkpoint_control_status(pin->control);
      const bool success = iree_status_is_ok(status);
      if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_RELEASE &&
          iree_status_is_not_found(status)) {
        // RELEASE establishes absence. An already-absent logical resource
        // satisfies cleanup without another operation or hidden retry.
        iree_status_ignore(status);
        status = iree_ok_status();
      }
      loom_serve_text_checkpoint_control_close(pin->control);
      pin->control = NULL;
      *out_cleanup_failed = !iree_status_is_ok(status) &&
                            pin->phase == LOOM_SERVE_REALTIME_HISTORY_RELEASE;
      if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE) {
        pin->phase =
            success ? (pin->invalidated ? LOOM_SERVE_REALTIME_HISTORY_RELEASE
                                        : LOOM_SERVE_REALTIME_HISTORY_PINNED)
                    : LOOM_SERVE_REALTIME_HISTORY_EMPTY;
      } else if (iree_status_is_ok(status)) {
        pin->phase = LOOM_SERVE_REALTIME_HISTORY_EMPTY;
      }
      *out_progress = true;
    }
    if (!iree_status_is_ok(status) ||
        pin->phase == LOOM_SERVE_REALTIME_HISTORY_EMPTY ||
        pin->phase == LOOM_SERVE_REALTIME_HISTORY_PINNED) {
      continue;
    }
    if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE && pin->invalidated) {
      // No capture has been admitted. There is no side effect to undo.
      pin->phase = LOOM_SERVE_REALTIME_HISTORY_EMPTY;
      *out_progress = true;
      continue;
    }
    const loom_serve_text_checkpoint_request_t request = {
        .action = pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE
                      ? LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN
                      : LOOM_SERVE_TEXT_CHECKPOINT_ACTION_RELEASE,
        .name = iree_string_builder_view(&pin->name),
        .session = pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE
                       ? iree_string_builder_view(&pin->session)
                       : iree_string_view_empty()};
    status = loom_serve_registry_try_open_text_checkpoint(
        history->registry, history->binding.name, &request, &pin->control);
    *out_progress |= pin->control != NULL;
    if (!iree_status_is_ok(status)) {
      if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_RELEASE &&
          iree_status_is_not_found(status)) {
        iree_status_ignore(status);
        status = iree_ok_status();
      }
      if (pin->phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE ||
          iree_status_is_ok(status)) {
        pin->phase = LOOM_SERVE_REALTIME_HISTORY_EMPTY;
      }
      *out_cleanup_failed = !iree_status_is_ok(status) &&
                            pin->phase == LOOM_SERVE_REALTIME_HISTORY_RELEASE;
    }
  }
  return status;
}

bool loom_serve_realtime_history_pending(
    const loom_serve_realtime_history_t* history) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    if (history->pins[i].phase != LOOM_SERVE_REALTIME_HISTORY_EMPTY) {
      return true;
    }
  }
  return false;
}

bool loom_serve_realtime_history_capturing(
    const loom_serve_realtime_history_t* history) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    if (history->pins[i].phase == LOOM_SERVE_REALTIME_HISTORY_CAPTURE) {
      return true;
    }
  }
  return false;
}

void loom_serve_realtime_history_deinitialize(
    loom_serve_realtime_history_t* history) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(history->pins); ++i) {
    iree_string_builder_deinitialize(&history->pins[i].name);
    iree_string_builder_deinitialize(&history->pins[i].session);
  }
}
