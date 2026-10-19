// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_RECOGNITION_GENERATOR_H_
#define LOOM_SERVE_SERVICE_RECOGNITION_GENERATOR_H_

#include "iree/tokenizer/tokenizer.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/service/recognition/events.h"
#include "loom/serve/service/recognition/input.h"
#include "loom/serve/service/recognition/request.h"

// Source-established compatibility at a retired model cut. A zero-unit offer
// awaits input without consuming it; positive units fit the declared batch
// capacity. Different progress alone need not produce different tags.
typedef struct loom_serve_recognition_work_t {
  // Additive numerical work, or zero when additional input is required.
  uint64_t units;
  // Equal tags permit one source batch when the sum of units fits.
  uint64_t tag;
} loom_serve_recognition_work_t;

typedef struct loom_serve_recognition_result_t {
  // Acknowledged prefix of the accepted input snapshot, excluding padding.
  // Requested activity observations partition this prefix exactly; lookahead
  // remains unconsumed until its probability is established.
  iree_host_size_t consumed_frames;
  // No further work or output events remain for this request.
  bool complete;
} loom_serve_recognition_result_t;

// Independently resident recognition requests over the shared execution and
// residency owners. All CPU/model calls are serialized; none waits for device
// work. Accepted rows, model-owned PCM, and original ordinals survive through
// actual retirement, including cancellation and failed readiness.
typedef struct loom_serve_recognition_generator_t {
  // Borrowed model owner, outliving all requests and completion waits.
  void* self;
  // Fixed normalized input geometry, independent of browser/audio containers.
  loom_serve_recognition_format_t format;
  // Maximum PCM frames in one source input snapshot, including lookahead.
  iree_host_size_t input_capacity;
  // Maximum total input frames in an admitted request.
  uint64_t frame_limit;
  // Maximum number of source events produced by one collected ordinal.
  iree_host_size_t event_capacity;
  // Maximum encoded bytes in that ordinal's complete source event packet.
  iree_host_size_t event_bytes;
  // Output families supported by the source, selected explicitly per request.
  loom_serve_recognition_output_flags_t outputs;
  // Stable resident rows, independent of executable batch shape.
  iree_host_size_t request_capacity;
  // Maximum sum of compatible work units in one numerical batch.
  uint64_t batch_units;
  // CPU-only request preparation. Failure retains no handle. row_index remains
  // reserved by the service until discard/release consumes the handle.
  iree_status_t (*prepare)(void* self, iree_host_size_t row_index,
                           const loom_serve_recognition_request_t* request,
                           void** out_invocation);
  // Owner-side release of an unadmitted request and its admission interest.
  void (*discard)(void* self, void* invocation);
  // ACQUIRED owns an activation pin/frontier, including a row-capacity denial.
  iree_status_t (*try_admit)(void* self, void* invocation,
                             loom_serve_residency_admission_t* out_admission,
                             loom_serve_execution_frontier_t* out_frontier);
  // Consumes successful admission readiness. False releases its activation pin.
  bool (*complete_admit)(void* self, void* invocation);
  // CPU-only offer over a captured input snapshot. Neither source progress nor
  // the input clock advances. Snapshot bytes are borrowed only through this
  // call; source owns any PCM needed by subsequent preparation or execution.
  // Input credit remains occupied until successful collection acknowledges it.
  // Requery after every retired cut.
  iree_status_t (*prepare_work)(
      void* self, void* invocation,
      const loom_serve_recognition_input_block_t* input,
      loom_serve_recognition_work_t* out_work);
  // Packs a nonempty set of distinct admitted handles with compatible positive
  // work. The source owns shape/phase selection. Failure retains no batch.
  iree_status_t (*prepare_batch)(void* self, iree_host_size_t count,
                                 void* const* invocations);
  // Queues one prepared numerical batch, retaining all original members.
  iree_status_t (*submit)(void* self,
                          loom_serve_execution_frontier_t* out_frontier);
  // Interprets successfully retired feedback and checks source event metadata.
  // Every original ordinal is collected, including cancelled members. Copies
  // into caller-owned event storage before another source call can overwrite
  // its packet. Storage has this generator's event count and byte bounds and
  // outlives the collected views. No source submission occurs during
  // collection. Failure terminates only this ordinal's request; its outputs
  // are ignored. The batch and remaining ordinals stay collectible, and the
  // failed handle remains releasable at the retired batch cut.
  iree_status_t (*collect)(void* self, iree_host_size_t ordinal,
                           loom_serve_recognition_events_t* events,
                           loom_serve_recognition_result_t* out_result);
  // Releases an unsubmitted plan or a retired batch after all CPU borrowers.
  void (*release_batch)(void* self);
  // Releases an admitted request at a successfully retired cut, after its
  // batch/CPU borrowers stop. True consumes the pin/handle. False retains it
  // through the returned borrowed frontier; call again after successful ALL
  // readiness, excluding other model mutation. Backing release never blocks
  // the owner. Platform failure retains the handle for model teardown.
  iree_status_t (*release)(void* self, void* invocation, bool* out_complete,
                           loom_serve_execution_frontier_t* out_frontier);
  // Optional immutable tokenizer, borrowed through model retirement.
  const iree_tokenizer_t* tokenizer;
  // Shared capacity identity, borrowed through service retirement.
  const void* capacity_domain;
  // Optional copied parameter activity, queried only on serialized owner turns.
  loom_serve_weight_activity_t (*weight_activity)(void* self);
} loom_serve_recognition_generator_t;

#endif  // LOOM_SERVE_SERVICE_RECOGNITION_GENERATOR_H_
