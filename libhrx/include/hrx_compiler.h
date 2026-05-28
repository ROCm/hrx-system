// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef HRX_COMPILER_H_
#define HRX_COMPILER_H_

#include "hrx_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_compiler_s *hrx_compiler_t;
typedef struct hrx_compiler_session_s *hrx_compiler_session_t;

typedef enum hrx_compiler_backend_t {
  // Auto-select the preferred available backend. This prefers the IREE
  // compiler dylib and falls back to the CLI backend only when
  // HRX_IREE_COMPILER_CLI is explicitly configured.
  HRX_COMPILER_BACKEND_AUTO = 0,
  HRX_COMPILER_BACKEND_DEFAULT = HRX_COMPILER_BACKEND_AUTO,
  HRX_COMPILER_BACKEND_DYLIB = 1,
  HRX_COMPILER_BACKEND_CLI = 2,
} hrx_compiler_backend_t;

// Creates a compiler frontend for the selected backend.
//
// DYLIB/AUTO load libIREECompiler.so from HRX_IREE_COMPILE or the default
// dynamic linker search path. CLI uses HRX_IREE_COMPILER_CLI and is an
// explicit debug fallback.
HRX_API hrx_status_t hrx_compiler_create(hrx_compiler_backend_t backend,
                                         hrx_compiler_t *compiler);

HRX_API void hrx_compiler_retain(hrx_compiler_t compiler);
HRX_API void hrx_compiler_release(hrx_compiler_t compiler);

// Returns the backend selected at create time.
HRX_API hrx_compiler_backend_t hrx_compiler_backend(hrx_compiler_t compiler);

// Creates a short-lived compiler session. For the dylib backend this owns the
// session-local MLIRContext and flags.
HRX_API hrx_status_t hrx_compiler_session_create(
    hrx_compiler_t compiler, hrx_compiler_session_t *session);

HRX_API void hrx_compiler_session_retain(hrx_compiler_session_t session);
HRX_API void hrx_compiler_session_release(hrx_compiler_session_t session);

// Replaces session-local compiler flags.
HRX_API hrx_status_t
hrx_compiler_session_set_flags(hrx_compiler_session_t session,
                               const char *const *flags, size_t flag_count);

// Compiles textual MLIR to a VMFB bytecode module artifact.
HRX_API hrx_status_t hrx_compiler_session_compile_mlir(
    hrx_compiler_session_t session, const char *mlir_data, size_t mlir_size,
    hrx_compiler_output_t *output);

HRX_API void hrx_compiler_output_retain(hrx_compiler_output_t output);
HRX_API void hrx_compiler_output_release(hrx_compiler_output_t output);

// Returns borrowed VMFB bytes owned by |output|.
HRX_API const uint8_t *hrx_compiler_output_data(hrx_compiler_output_t output);
HRX_API size_t hrx_compiler_output_size(hrx_compiler_output_t output);

#ifdef __cplusplus
}
#endif

#endif // HRX_COMPILER_H_
