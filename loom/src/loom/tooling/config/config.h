// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Config materialization helpers shared by Loom command-line tools and
// programmatic compiler entry points.
//
// Config bindings are explicit invocation state. Tools append direct
// assignments:
//
//   --config=model36.model.hidden_size=4096
//
// and JSON/JSONC config files:
//
//   {"model36":{"model":{"hidden_size":4096}}}
//
// Both forms produce the same flattened key/value binding. Materialization
// then replaces matching config.decl/config.def symbols in the loaded module,
// leaving unreferenced caller bindings in the config set so one invocation can
// specialize whichever declarations are present after later linking.

#ifndef LOOM_TOOLING_CONFIG_CONFIG_H_
#define LOOM_TOOLING_CONFIG_CONFIG_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/util/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// One caller-provided config binding. The key is the symbol name with or
// without the textual '@' sigil. The value is parsed according to the matched
// config.decl/config.def result type.
typedef struct loom_tooling_config_binding_t {
  // Config symbol name, without ownership transfer.
  iree_string_view_t key;
  // Textual config value, without ownership transfer.
  iree_string_view_t value;
} loom_tooling_config_binding_t;

// Owned config bindings for one compiler operation.
//
// Config sets are explicit invocation state. They are intentionally separate
// from sessions, contexts, and pass managers so callers can reuse long-lived
// compiler infrastructure across many independent compilations without ambient
// config leakage.
typedef struct loom_tooling_config_set_t {
  // Host allocator owning the binding array and copied key/value strings.
  iree_allocator_t host_allocator;
  // Owned normalized bindings. Callers must not mutate this array directly.
  loom_tooling_config_binding_t* bindings;
  // Number of entries in |bindings|.
  iree_host_size_t binding_count;
  // Allocated capacity of |bindings|.
  iree_host_size_t binding_capacity;
} loom_tooling_config_set_t;

// Materialization policy flags.
enum loom_tooling_config_materialize_flag_bits_e {
  // Reject bindings whose keys do not name config symbols in the module.
  // Programmatic callers compiling many modules with a shared config map can
  // leave this unset so non-sensitive bindings are ignored.
  LOOM_TOOLING_CONFIG_MATERIALIZE_REQUIRE_MATCHES = 1u << 0,
};
typedef uint32_t loom_tooling_config_materialize_flags_t;

// Options for materializing config values into a module.
typedef struct loom_tooling_config_materialize_options_t {
  // Policy flags controlling typo handling and module sensitivity.
  loom_tooling_config_materialize_flags_t flags;
  // Borrowed config set for the current compiler operation. NULL is accepted
  // and treated as an empty set.
  const loom_tooling_config_set_t* config_set;
} loom_tooling_config_materialize_options_t;

// Summary of a materialization run.
typedef struct loom_tooling_config_materialize_result_t {
  // Number of bindings that replaced config symbols with config.def ops.
  iree_host_size_t materialized_count;
  // Number of bindings ignored because the module has no matching config
  // symbol and REQUIRE_MATCHES was unset.
  iree_host_size_t ignored_count;
} loom_tooling_config_materialize_result_t;

// Summary of a config resolution check.
typedef struct loom_tooling_config_resolution_result_t {
  // Number of unresolved config.decl symbols found in the module.
  iree_host_size_t unresolved_count;
} loom_tooling_config_resolution_result_t;

// Initializes options to a safe default: no bindings and non-strict matching.
void loom_tooling_config_materialize_options_initialize(
    loom_tooling_config_materialize_options_t* out_options);

// Initializes |out_config_set| as an empty owned config set.
void loom_tooling_config_set_initialize(
    iree_allocator_t host_allocator, loom_tooling_config_set_t* out_config_set);

// Releases all strings and storage owned by |config_set|.
void loom_tooling_config_set_deinitialize(
    loom_tooling_config_set_t* config_set);

// Appends a single config binding to |config_set|.
//
// The key is normalized in the same way as command-line assignments:
// surrounding whitespace is trimmed and one leading '@' sigil is removed. The
// value is trimmed and stored as text to parse against each matching IR symbol
// type. Duplicate normalized keys are rejected so precedence remains explicit.
iree_status_t loom_tooling_config_set_append(
    loom_tooling_config_set_t* config_set, iree_string_view_t key,
    iree_string_view_t value);

// Parses a single `key=value` command-line assignment into borrowed views.
//
// The split happens at the first '='. Both sides are trimmed. A leading '@' on
// the key is accepted and removed so command-line spelling matches IR spelling
// without forcing shell users to quote sigils unnecessarily.
iree_status_t loom_tooling_config_parse_assignment(
    iree_string_view_t assignment, loom_tooling_config_binding_t* out_binding);

// Parses and appends one `key=value` command-line assignment to |config_set|.
iree_status_t loom_tooling_config_set_append_assignment(
    loom_tooling_config_set_t* config_set, iree_string_view_t assignment);

// Parses a JSON/JSONC object and appends flattened config bindings.
//
// Nested object keys are joined with '.' so `{ "model36": { "model": {
// "hidden_size": 4096 }}}` appends the same binding as
// `--config=model36.model.hidden_size=4096`. Leaf values may be JSON booleans,
// numbers, or strings. String leaves are unescaped and then parsed later using
// the matched config symbol type, which lets structured files carry textual
// Loom values such as encodings without inventing a second value grammar.
iree_status_t loom_tooling_config_set_append_json_object(
    loom_tooling_config_set_t* config_set, iree_string_view_t json_object);

// Reads a JSON/JSONC config object file and appends flattened bindings.
//
// Empty paths and "-" are rejected. Command-line tools reserve those spellings
// for stdin/stdout, but config file loading is intentionally a filesystem-path
// operation so accidental stdin consumption cannot race the module input.
iree_status_t loom_tooling_config_set_append_json_file(
    loom_tooling_config_set_t* config_set, iree_string_view_t path,
    iree_allocator_t host_allocator);

// Replaces matching config.decl/config.def symbol ops with config.def ops whose
// initializer attributes are parsed from |options->config_set|. Bindings
// without matching config symbols are ignored unless REQUIRE_MATCHES is set.
//
// This is intentionally a direct module operation rather than a pass. Tooling
// should call it immediately after loading and, when requested, initially
// verifying a module. The resulting IR then behaves exactly like linked or
// user-authored config.def IR for verifiers, dependency analysis, and passes.
iree_status_t loom_tooling_config_materialize_module(
    loom_module_t* module,
    const loom_tooling_config_materialize_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result);

// Requires that |module| contains no remaining config.decl symbols.
//
// Linkable/library outputs should not call this: unresolved declarations are
// valid IR and keep config sensitivity visible to symbol dependency/index
// consumers. Final compilation drivers should call this after config
// materialization and any pruning passes that remove unused declarations.
iree_status_t loom_tooling_config_require_resolved_module(
    const loom_module_t* module,
    loom_tooling_config_resolution_result_t* out_result);

// Writes a stable JSON description of config symbols in |module|.
//
// The report includes both unresolved config.decl symbols and resolved
// config.def defaults/overrides. Consumers should treat the textual type and
// default fields as display/round-trip strings and the constraints array as the
// structured contract for programmatic validation.
iree_status_t loom_tooling_config_format_schema_json(
    const loom_module_t* module, loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_CONFIG_CONFIG_H_
