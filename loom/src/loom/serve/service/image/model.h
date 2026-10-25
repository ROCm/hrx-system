// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_IMAGE_MODEL_H_
#define LOOM_SERVE_SERVICE_IMAGE_MODEL_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/serve/service/image/generator.h"

#ifdef __cplusplus
extern "C" {
#endif

// One source-defined diffusion residency with serialized cohort storage.
// The application owns serialization; simultaneous generate calls are invalid.
typedef struct loom_serve_image_model_t loom_serve_image_model_t;

typedef struct loom_serve_image_model_options_t {
  // Explicit program inputs and local base, borrowed through load completion.
  loom_serve_sources_t sources;
  // Checkpoint asset directory, passed to the source bootstrap.
  iree_string_view_t checkpoint_directory;
  // Optional adapter asset, interpreted by the source bootstrap.
  iree_string_view_t adapter_path;
  // Source-owned JSON options, borrowed through load completion; empty is {}.
  iree_string_view_t options;
  // Default output height in pixels, used for cold preparation.
  uint32_t height;
  // Default output width in pixels, used for cold preparation.
  uint32_t width;
  // Text capacity requested from the source; its meaning is model-defined.
  uint32_t text_tokens;
  // Soft command-count target per submission; zero records the whole program.
  // Cuts occur only at full barriers, never within an indivisible wave.
  uint32_t command_quantum;
  // Bounds reusable request code without restricting one active image.
  loom_serve_variant_cache_options_t specialization_cache;
} loom_serve_image_model_options_t;

// Owner-driven preparation. Private CPU/JIT work runs on the shared host;
// advance alone binds shared memory, residency and runtime capabilities.
typedef struct loom_serve_image_load_t loom_serve_image_load_t;

// Called after accepted preparation work retires. May only notify/schedule
// the serialized owner; it must not call back into the load or block. Payload
// and borrowed device/options remain live through the load's terminal advance.
typedef void (*loom_serve_image_load_wake_fn_t)(void* user_data);

// Allocates an unstarted operation without model storage reservation, source
// I/O or work submission. Options' views are borrowed, not copied. A registered
// but drained deployment can defer creating this operation altogether.
iree_status_t loom_serve_image_load_allocate(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    loom_serve_image_load_wake_fn_t wake, void* user_data,
    iree_allocator_t host_allocator, loom_serve_image_load_t** out_load);

enum loom_serve_image_load_flag_bits_e {
  // Stops subsequent phases, retiring any accepted CPU/JIT work before the
  // terminal CANCELLED result. No callback tears down shared resources.
  LOOM_SERVE_IMAGE_LOAD_FLAG_CANCEL = 1u << 0,
};
typedef uint32_t loom_serve_image_load_flags_t;

// Serialized pool-owner turn. Never waits for tasks or device work. After a
// nonterminal result, wake schedules another turn. A terminal result transfers
// exclusive access to the owner, including on failure; no callbacks remain.
// Call until *out_complete, then inspect evidence and take or destroy the load.
iree_status_t loom_serve_image_load_advance(loom_serve_image_load_t* load,
                                            loom_serve_image_load_flags_t flags,
                                            bool* out_complete);

// Terminal-only queries. Evidence is borrowed until take_model or destruction;
// take_model transfers a successfully prepared, still-unactivated model.
const loom_serve_compilation_report_t* loom_serve_image_load_report(
    const loom_serve_image_load_t* load);
loom_serve_image_model_t* loom_serve_image_load_take_model(
    loom_serve_image_load_t* load);

// Destroys a terminal operation and any untaken model. Preparation submits no
// queue work, so partial-load cleanup requires no asynchronous retirement.
// A platform release failure preserves the operation and remaining ownership.
iree_status_t loom_serve_image_load_free(loom_serve_image_load_t* load);

// Source ABI:
// prepare.loom: prepare(i64 height, i64 width, i64 text_capacity,
//                      buffer checkpoint, buffer adapter, buffer options)
//   -> (i64 batch_units, i64 request_capacity, buffer tokenizer_asset,
//       buffer name, buffer state)
// The source declares commands/configuration/parameter domains through prepare
// imports. tokenizer_asset is relative to checkpoint. Returned state is opaque
// and retained independently of the cold program; its VM environment stays
// live. control.loom:
//   prepare_request(buffer state, i64 height, i64 width, buffer prompt,
//                   i64 seed, buffer loras) -> (buffer bytes, i64 units, i64
//                   tag)
//   prepare_batch(buffer state, buffer i64_stage_tags, buffer ranges,
//                 buffer bytes) -> (i32 stage, buffer payload)
//   bind_parameters(buffer state, buffer payload, buffer bank_offsets)
//   generate(i32 first_part, i32 part_count, i32 cursor, i64 output_bytes,
//            hal.buffer payload, hal.buffer rgb, hal.buffer workspace) -> i32
// The cursor starts at zero. Each invocation submits its next part and returns
// a cursor for continuation, or a negative value after final feedback. The
// owner invokes it again only after both accepted frontiers retire. State is
// immutable across requests. loras contains the validated records documented
// by loom_serve_image_request_t; source policy resolves registered identities
// and rejects unsupported selections before publishing prepared bytes.
// Positive units bound cohort work/storage; equal tags permit batching.
// Ranges are little-endian [offset, length] i64 pairs
// into concatenated prepared bytes. Output is concatenated F32 CHW RGB in that
// same stable request order. A cancelled peer never changes this layout. The
// warm program imports input/JSON capabilities, invocation-only prepare
// declarations, and runner.execute_N/feedback. Request stage ordinals follow
// the immutable cold table. Explicit checkpoint declarations select retained
// domains or prepare independent banks on demand; omitted declarations preserve
// the cold layout. Stages declared by one cohort share a parameter binding
// map, while retained cohort choices can bind different checkpoint banks. Each
// cold-declared parameter arena can supply independently selected banks through
// one fixed binding. Such a model declares a bind_parameters control entry at
// registration. The owner invokes it once after successful bank resolution and
// before the existing payload upload; bank_offsets is an owned read-only buffer
// of little-endian u64 arena offsets in parameter_bank declaration order. The
// source patches its opaque payload, retaining item order and strengths. The
// invocation retains selected banks through actual execution retirement. Each
// command has three dynamic bindings: opaque request bytes, output F32 CHW RGB
// [-1,1], and workspace. External input/output roots promise only byte
// alignment; stronger private storage and parameter alignments come from their
// compiler-reflected contracts. Input capacity comes from source; compiler
// reflection supplies the workspace extent and alignment. One queue allocation
// transaction acquires independent output and private workspace roots.
// execute_3 receives both. Final work or cancellation queue-releases workspace;
// feedback retains only output through its actual retirement. Private
// storage belongs to the shared workspace pool, not this model's residency.
// This model owns independent work and feedback timelines over the device
// queues. The cold constructor drives the asynchronous load and joins on its
// outer application caller, never a shared worker. Live registration uses
// load_advance. Creation indexes parameters and records commands without
// reading weight payloads. The device owner outlives the model and serializes
// all model calls. Its physical pool backs parameter roots independently of
// cached commands. First generation or explicit activation streams/prepares
// weights. Active weights remain shared across request specializations; an
// exact code-cache hit does no JIT or parameter preparation.
// Failure releases partial ownership and leaves *out_model NULL. Compilation
// evidence is available through load_report; this wrapper performs no
// process-global output.
iree_status_t loom_serve_image_model_allocate(
    loom_serve_device_t* device,
    const loom_serve_image_model_options_t* options,
    iree_allocator_t host_allocator, loom_serve_image_model_t** out_model);

// Begins irreversible removal after service CPU tasks, waits and external
// tokenizer borrowers retire. No further invocation/admission is allowed.
// Attaches nonblocking ownership wakes before releasing cached VM/command/root
// consumers. Payloads, feedback and residency protection remain until actual
// queue ownership retires, independently of success/failure readiness. The wake
// only schedules the serialized owner; it must not reenter or destroy
// resources.
void loom_serve_image_model_retire(loom_serve_image_model_t* model,
                                   void (*wake)(void* user_data),
                                   void* user_data);

// Query after retire. A true result authorizes free_retired; no device wait,
// new submission or retry is needed. NULL is already retired.
bool loom_serve_image_model_is_retired(loom_serve_image_model_t* model);

// Final owner-side destruction after is_retired. Never waits for worker/device
// work. On platform cleanup failure the model and remaining reservations stay
// owned by the caller; otherwise the model is consumed. NULL is accepted.
iree_status_t loom_serve_image_model_free_retired(
    loom_serve_image_model_t* model);

// Cold outer-owner teardown after service CPU tasks and waits retire. Begins
// retirement and joins actual queue ownership before free_retired. Live
// model removal uses the wake/query path instead of blocking a shared worker.
iree_status_t loom_serve_image_model_free(loom_serve_image_model_t* model);

// Reloads or releases every checkpoint domain while preserving compiled
// commands. Deactivation requires unpinned elastic backing. Next generation
// pins every domain together and activates on demand, reclaiming eligible idle
// weights of other models if needed. Activation is an outer-thread join;
// deactivation advances without waiting until out_complete joins native
// release. Execution/I/O failure is terminal.
iree_status_t loom_serve_image_model_activate(loom_serve_image_model_t* model);
iree_status_t loom_serve_image_model_deactivate(loom_serve_image_model_t* model,
                                                bool* out_complete);
// Borrowed admission for the selected invocation, valid until its release.
// Without a selected invocation this returns the original cold working set.
// Model-wide retention policy is applied with model_set_pinned below.
loom_serve_residency_t* loom_serve_image_model_residency(
    const loom_serve_image_model_t* model);

// Model-wide checkpoint accounting and policy, including idle cached banks.
loom_serve_residency_snapshot_t loom_serve_image_model_parameters(
    const loom_serve_image_model_t* model);
void loom_serve_image_model_set_pinned(loom_serve_image_model_t* model,
                                       bool pinned);
// Owner-side idle retirement; native failures retain ownership for teardown.
iree_status_t loom_serve_image_model_query_status(
    loom_serve_image_model_t* model);
bool loom_serve_image_model_is_pending(const loom_serve_image_model_t* model);
loom_serve_memory_usage_t loom_serve_image_model_weight_statistics(
    const loom_serve_image_model_t* model);
// Copies cumulative submission activity across the model's parameter domains.
loom_serve_weight_activity_t loom_serve_image_model_weight_activity(
    const loom_serve_image_model_t* model);

// Borrows the source-declared public identifier for this residency.
iree_string_view_t loom_serve_image_model_name(
    const loom_serve_image_model_t* model);

// Borrows this model's finite invocation callbacks for a serialized service.
// The model and its shared device owner outlive the returned interface.
loom_serve_image_generator_t loom_serve_image_model_generator(
    loom_serve_image_model_t* model);

// CPU-only source preparation, serialized with other model calls. Suitable for
// a CPU task: no device submission, wait or shared residency mutation occurs.
// Success transfers opaque prepared bytes to the caller. These may borrow the
// request storage, which remains live through batch preparation. Multiple
// prepared requests may coexist without capturing executables or checkpoint
// banks. INVALID_ARGUMENT rejects the request without retaining ownership.
// Width/height are positive and their F32 RGB extent fits host storage and i64;
// the public request parser establishes this representational boundary. Source
// preparation validates model-specific geometry before any output allocation.
// Returns source work/compatibility facts and owned bytes, or zeros on failure.
iree_status_t loom_serve_image_model_prepare_request(
    loom_serve_image_model_t* model, const loom_serve_image_request_t* request,
    loom_serve_prepared_request_t* out_prepared);

// CPU-only preparation of one compatible cohort in stable client order. The
// caller supplies protocol-validated F32 RGB byte lengths in the same order.
// Captures one executable generation and the complete checkpoint bank union;
// there must be no preceding invocation. Failure leaves no prepared cohort.
// Prepared request bytes remain caller-owned and can be released on return.
iree_status_t loom_serve_image_model_prepare_batch(
    loom_serve_image_model_t* model, iree_host_size_t count,
    const loom_serve_prepared_request_t* requests,
    const iree_host_size_t* output_lengths,
    loom_serve_image_prepared_t* out_prepared);

// Installs the serialized owner's nonblocking compilation wake. Detachment
// follows terminal advance of all accepted preparation work.
void loom_serve_image_model_set_observer(loom_serve_image_model_t* model,
                                         loom_serve_wake_callback_t observer);

// Resolves the prepared invocation's captured executables before admission.
// Cancellation retires accepted compilation but submits no device work. The
// owner calls until complete, then admits or discards the prepared cohort.
iree_status_t loom_serve_image_model_advance_batch(
    loom_serve_image_model_t* model, bool cancel, bool* out_complete);

// Releases a successfully prepared cohort that has not been submitted. This
// does not release residency pins or retire device work. The model is reusable.
void loom_serve_image_model_discard_batch(loom_serve_image_model_t* model);

// Queues one part and returns its work/feedback frontiers and terminal state.
// The first call follows residency acquisition; continuation or cancellation
// follows successful retirement of both previous frontiers. Only one cohort
// may be outstanding. Failure preserves payload and residency protection for
// outer-owner destruction; no host wait occurs.
iree_status_t loom_serve_image_model_submit(
    loom_serve_image_model_t* model, loom_serve_image_submit_flags_t flags,
    loom_serve_execution_frontier_t* out_frontier, bool* out_complete);

// Consumes successful completion of both terminal frontiers on the scheduler
// owner. Releases the request payload and residency pin. Returned F32 CHW RGB
// remains borrowed until another preparation or model destruction. Failed
// readiness never calls this function: teardown joins actual queue ownership.
// Cancelled invocations return storage without valid output; callers discard
// it.
iree_const_byte_span_t loom_serve_image_model_complete(
    loom_serve_image_model_t* model);

// Source prepares one opaque payload and selects a retained command. Its
// payload is kept alive through both accepted execution frontiers, including
// failures after partial submission. Request errors before submission leave
// the model reusable; execution failure is terminal. Request is borrowed only
// for this call. Success returns little-endian F32 CHW RGB, borrowed until the
// next generate call or destruction. Failure leaves *out_rgb empty.
// out_prepared retains source-selected facts if preparation succeeded, even
// when subsequent admission or device work fails.
iree_status_t loom_serve_image_model_generate(
    loom_serve_image_model_t* model, const loom_serve_image_request_t* request,
    loom_serve_image_prepared_t* out_prepared, iree_const_byte_span_t* out_rgb);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_IMAGE_MODEL_H_
