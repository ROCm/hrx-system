// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_CONSUMER_H_
#define LOOM_SERVE_SERVICE_SPEECH_CONSUMER_H_

#include "loom/serve/service/speech/generator.h"
#include "loom/serve/telemetry/record.h"

#ifdef __cplusplus
extern "C" {
#endif

// One request's output owner. The service retains this receipt through every
// accepted CPU borrower and source release. Only encode runs on a worker; all
// other operations run on the serialized serving owner. encode may overlap
// interest/credit queries or cancel and therefore never accesses their fields.
typedef struct loom_serve_speech_consumer_t {
  // Owned consumer receipt, consumed exactly once by end.
  void* self;
  // Whether the caller still wants output. False cannot become true again.
  bool (*is_open)(void* self);
  // Whether another bounded block can be accepted. Queries never wait.
  bool (*can_write)(void* self);
  // Copies/converts one retired source block into bounded consumer storage.
  // Source views stay borrowed only through this finite CPU call. No owner
  // field is touched and no transport or device operation is submitted.
  // Failure terminates this request; the partially encoded block is never
  // published. end consumes the receipt after all accepted borrowers retire.
  iree_status_t (*encode)(void* self, const loom_serve_speech_block_t* block);
  // Publishes the encoded block after its CPU borrower retires. False is
  // output pressure; no bytes/state were accepted. Success with true consumes
  // this publication even for a source step producing no waveform.
  iree_status_t (*publish)(void* self, bool* out_published);
  // Stops interest and releases transport ownership without freeing storage
  // still borrowed by encode. Idempotent; end still consumes the receipt.
  void (*cancel)(void* self);
  // Consumes the receipt after every CPU borrower joins. The source has either
  // released its request or retained it for terminal model teardown; neither
  // path can access this consumer again. Status is borrowed during the call.
  // Independently owned final output/status may survive this operation.
  iree_status_t (*end)(void* self, loom_serve_telemetry_outcome_t outcome,
                       const iree_status_t status);
} loom_serve_speech_consumer_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_SPEECH_CONSUMER_H_
