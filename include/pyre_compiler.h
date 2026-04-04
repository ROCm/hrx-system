// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef PYRE_COMPILER_H_
#define PYRE_COMPILER_H_

#include "pyre_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pyre_compiler_s* pyre_compiler_t;
typedef struct pyre_compiler_session_s* pyre_compiler_session_t;

typedef enum pyre_compiler_backend_t {
  // Auto-select the preferred available backend. This prefers the IREE
  // compiler dylib and falls back to the CLI backend only when
  // PYRE_IREE_COMPILER_CLI is explicitly configured.
  PYRE_COMPILER_BACKEND_AUTO = 0,
  PYRE_COMPILER_BACKEND_DEFAULT = PYRE_COMPILER_BACKEND_AUTO,
  PYRE_COMPILER_BACKEND_DYLIB = 1,
  PYRE_COMPILER_BACKEND_CLI = 2,
} pyre_compiler_backend_t;

// Creates a compiler frontend for the selected backend.
//
// DYLIB/AUTO load libIREECompiler.so from PYRE_IREE_COMPILE or the default
// dynamic linker search path. CLI uses PYRE_IREE_COMPILER_CLI and is an
// explicit debug fallback.
PYRE_API pyre_status_t pyre_compiler_create(
    pyre_compiler_backend_t backend, pyre_compiler_t* compiler);

PYRE_API void pyre_compiler_retain(pyre_compiler_t compiler);
PYRE_API void pyre_compiler_release(pyre_compiler_t compiler);

// Returns the backend selected at create time.
PYRE_API pyre_compiler_backend_t pyre_compiler_backend(
    pyre_compiler_t compiler);

// Creates a short-lived compiler session. For the dylib backend this owns the
// session-local MLIRContext and flags.
PYRE_API pyre_status_t pyre_compiler_session_create(
    pyre_compiler_t compiler, pyre_compiler_session_t* session);

PYRE_API void pyre_compiler_session_retain(
    pyre_compiler_session_t session);
PYRE_API void pyre_compiler_session_release(
    pyre_compiler_session_t session);

// Replaces session-local compiler flags.
PYRE_API pyre_status_t pyre_compiler_session_set_flags(
    pyre_compiler_session_t session, const char* const* flags,
    size_t flag_count);

// Compiles textual MLIR to a VMFB bytecode module artifact.
PYRE_API pyre_status_t pyre_compiler_session_compile_mlir(
    pyre_compiler_session_t session, const char* mlir_data, size_t mlir_size,
    pyre_compiler_output_t* output);

PYRE_API void pyre_compiler_output_retain(
    pyre_compiler_output_t output);
PYRE_API void pyre_compiler_output_release(
    pyre_compiler_output_t output);

// Returns borrowed VMFB bytes owned by |output|.
PYRE_API const uint8_t* pyre_compiler_output_data(
    pyre_compiler_output_t output);
PYRE_API size_t pyre_compiler_output_size(
    pyre_compiler_output_t output);

#ifdef __cplusplus
}
#endif

#endif  // PYRE_COMPILER_H_
