// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_SOURCE_H_
#define LOOM_SERVE_SERVICE_TEXT_SOURCE_H_

#include "iree/module/hal/types.h"
#include "iree/vm/buffer.h"
#include "iree/vm/environment.h"
#include "iree/vm/module.h"
#include "loom/serve/runtime/parameter/sources.h"
#include "loom/serve/runtime/program/preparation.h"
#include "loom/serve/runtime/program/tokenizer_cache.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/storage.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_serve_jit_t loom_serve_jit_t;

// Source-owned preparation, independent of device bindings and invocation.
// Private construction may run on a shared worker; the model retains this
// representation and its descriptor tables through all warm program consumers.
typedef struct loom_serve_text_source_t {
  // Coherent source contract selected before compilation and tokenizer work.
  loom_serve_text_mode_t mode;
  // Allocation policy for source-owned metadata and bytecode.
  iree_allocator_t host_allocator;
  // Construction registry for canonical VM/HAL reference descriptors.
  iree_vm_environment_t* environment;
  // Resolved byte-buffer reference descriptors.
  iree_vm_ref_types_t vm_types;
  // Resolved HAL reference descriptors, without device capabilities.
  iree_hal_module_types_t types;
  // Owned source declarations and parameter policies.
  loom_serve_preparation_t* preparation;
  // Owned immutable checkpoint indexes shared by resident and demand reads.
  loom_serve_parameter_sources_t* parameter_sources;
  // Compiled control bytecode awaiting native capability linking.
  iree_vm_module_t* module;
  // Opaque source state retained independently of the bootstrap invocation.
  iree_vm_buffer_t* control_state;
  // Owned deployment JSON shared with bootstrap and cold request policy.
  iree_vm_buffer_t* options;
  // Owned parsed vocabulary reused across requests and tokenizer borrowers.
  loom_serve_tokenizer_t* tokenizer;
  // Source terminal marker retained until independent tokenizer construction.
  iree_vm_buffer_t* terminal_marker;
  // Source terminal marker resolved once against the tokenizer.
  int32_t eos_token;
  // Effective prefill capacity selected by source preparation.
  iree_host_size_t prefill_capacity;
  // Leading stages sharing one immutable parameter placement.
  iree_host_size_t shared_stage_count;
  // Resolved service caches and exact cold storage sizing evidence.
  loom_serve_text_cache_selection_t cache;
} loom_serve_text_source_t;

// Consumes validated model options and produces source metadata plus the
// caller-owned storage description. Both outputs retain partial ownership on
// failure and must be deinitialized by their owners. No device memory, queue
// submission or shared pool mutation occurs. Option views are borrowed only
// through this call; source-returned state may retain its own copied inputs.
// jit supplies the source grammar, model-local report, and immutable VM cache.
iree_status_t loom_serve_text_source_initialize(
    loom_serve_jit_t* jit, const loom_serve_text_options_t* options,
    iree_allocator_t host_allocator, loom_serve_text_source_t* out_source,
    loom_serve_text_storage_t* out_storage);

// Resolves source terminal semantics after the model owner joins tokenizer
// construction and transfers the tokenizer into source. No device work occurs.
iree_status_t loom_serve_text_source_resolve_tokenizer(
    loom_serve_text_source_t* source);

// Releases partial or complete construction after all native model/program
// consumers have retired. Storage returned by initialize has its own owner.
void loom_serve_text_source_deinitialize(loom_serve_text_source_t* source);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_SOURCE_H_
