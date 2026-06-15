// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured IR generator for producing valid loom modules.
//
// Generates syntactically and semantically valid loom IR from either a
// deterministic PRNG seed or raw fuzzer bytes. The generator serves three
// use cases:
//
//   Benchmarks: seeded generation produces reproducible modules of
//   controlled size for measuring parser, printer, verifier, and pass
//   throughput. Scale factors control op count, nesting depth, and
//   function count independently.
//
//   Property tests: seeded generation with varying seeds explores the
//   IR space systematically. The value tracking set maintains SSA
//   validity (every operand references a dominating definition), and
//   the type palette controls the distribution of scalar types to
//   stress specific code paths.
//
//   Fuzzing: raw byte consumption mode consumes fuzzer-provided data
//   byte-by-byte, falling back to zeros when the input is exhausted.
//   This lets libFuzzer/AFL guide generation toward interesting IR
//   structures without requiring a custom mutator.
//
// The generator uses a hook-based architecture for extensibility.
// Op generation is driven by an array of weighted hooks: each hook
// is a function pointer paired with a weight. At each generation
// step, the generator selects a hook by weighted random choice and
// calls it to emit one or more ops. This decouples the generation
// core from dialect-specific op knowledge.
//
// The default presets use only the synthetic test dialect so generator
// infrastructure tests stay independent of production dialects. Production
// dialect hook sets are exposed as explicit profiles; callers opt into them by
// registering those dialects and installing the corresponding hook table.
//
// Preset configs bundle hooks, palettes, and sizing parameters for
// common scenarios (representative IR, CSE stress, DCE stress, deep
// nesting, format stress). Each preset takes a scale factor that
// controls approximate op count.

#ifndef LOOM_TESTING_GEN_H_
#define LOOM_TESTING_GEN_H_

#include "iree/base/api.h"
#include "iree/base/internal/prng.h"
#include "loom/ir/types.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

// IR construction handle. Bundles the target module, allocation
// arena, and insertion point into one object.
typedef struct loom_builder_t loom_builder_t;

// Global compilation context. Owns dialect registrations, op vtable
// lookup tables, and encoding registrations. Immutable after
// finalization; shared across modules and threads.
typedef struct loom_context_t loom_context_t;

// Top-level IR container. Owns the arena, string table, type table,
// value table, and symbol table for one compilation unit.
typedef struct loom_module_t loom_module_t;

// A single IR operation node. Carries op kind, operands, results,
// attributes, regions, and optional tied result bindings.
typedef struct loom_op_t loom_op_t;

//===----------------------------------------------------------------------===//
// Randomness source
//===----------------------------------------------------------------------===//

// Dual-mode randomness source: PRNG for deterministic generation,
// raw byte consumption for fuzzing.
//
// In PRNG mode (fuzz_data == NULL), all randomness comes from the
// xoroshiro128 state seeded at initialization. Generation is fully
// deterministic for a given seed.
//
// In fuzz mode (fuzz_data != NULL), bytes are consumed from the
// fuzzer-provided buffer. When the buffer is exhausted, fuzz_data
// is NULLed and subsequent calls fall through to the PRNG, which
// was seeded from the first 8 bytes of the fuzz input.
typedef struct loom_test_gen_t {
  // Xoroshiro128 PRNG state. Used in seeded mode and as the sole
  // randomness source when fuzz_data is NULL.
  iree_prng_xoroshiro128_state_t prng;
  // Pointer to the next unconsumed byte of fuzzer input. NULL in
  // seeded mode. Advances as bytes are consumed.
  const uint8_t* fuzz_data;
  // Number of bytes remaining in the fuzzer input buffer.
  iree_host_size_t fuzz_remaining;
} loom_test_gen_t;

// Initializes a generator in deterministic PRNG mode. All randomness
// derives from |seed|. The same seed always produces the same IR.
void loom_test_gen_initialize_seeded(uint64_t seed, loom_test_gen_t* out_gen);

// Initializes a generator in fuzz mode. Bytes are consumed from
// |data| (|size| bytes). When exhausted, returns zeros. The data
// buffer must remain valid for the lifetime of the generator.
void loom_test_gen_initialize_fuzz(const uint8_t* data, iree_host_size_t size,
                                   loom_test_gen_t* out_gen);

// Returns the next 64-bit random value. In PRNG mode, steps the
// xoroshiro128** generator. In fuzz mode, consumes 8 bytes from
// the input buffer (zero-padded if fewer than 8 bytes remain).
uint64_t loom_test_gen_next_uint64(loom_test_gen_t* gen);

// Returns the next 32-bit random value. In PRNG mode, uses the
// upper 32 bits of a 64-bit step. In fuzz mode, consumes 4 bytes.
uint32_t loom_test_gen_next_uint32(loom_test_gen_t* gen);

// Returns a uniformly distributed value in [0, upper_exclusive).
// Uses rejection sampling to avoid modulo bias: draws 32-bit values
// and rejects those in the biased range. For power-of-two bounds,
// a single draw with masking suffices.
uint32_t loom_test_gen_next_range(loom_test_gen_t* gen,
                                  uint32_t upper_exclusive);

// Returns a random boolean with approximately 50/50 distribution.
bool loom_test_gen_next_bool(loom_test_gen_t* gen);

// Returns true with approximately |percent| probability (0-100).
// A percent of 0 always returns false; 100 always returns true.
// Values outside [0, 100] are clamped.
bool loom_test_gen_next_probability(loom_test_gen_t* gen, uint8_t percent);

//===----------------------------------------------------------------------===//
// Type palette
//===----------------------------------------------------------------------===//

// A weighted set of scalar types controlling the distribution of
// generated types. Each entry pairs a scalar type with a weight.
// When a type is needed, the generator draws from the palette using
// weighted random selection: types with higher weights appear more
// frequently in the generated IR.
//
// Fixed capacity (LOOM_SCALAR_TYPE_COUNT_) avoids allocation. The
// total_weight field caches the sum for fast weighted selection.
typedef struct loom_test_gen_type_palette_t {
  // Scalar types in this palette. Only the first |count| entries
  // are valid.
  loom_scalar_type_t types[LOOM_SCALAR_TYPE_COUNT_];
  // Per-type weights. Higher weight means the type appears more
  // frequently in generated IR.
  uint16_t weights[LOOM_SCALAR_TYPE_COUNT_];
  // Number of valid entries in types[] and weights[].
  uint16_t count;
  // Cached sum of all weights. Must equal the sum of
  // weights[0..count-1]. Updated by palette construction functions.
  uint16_t total_weight;
} loom_test_gen_type_palette_t;

// Fills |out_palette| with a representative distribution of scalar
// types for general-purpose IR generation. The default weights
// emphasize common compute types while including less frequent types
// for coverage:
//
//   i8:3  i16:3  i32:5  i64:3  f16:1  bf16:1  f32:2  f64:1  index:2
void loom_test_gen_type_palette_default(
    loom_test_gen_type_palette_t* out_palette);

// Selects a scalar type from |palette| by weighted random choice and
// returns it wrapped as a loom_type_t via loom_type_scalar(). The
// palette must have at least one entry (count > 0).
loom_type_t loom_test_gen_type_palette_pick(
    loom_test_gen_t* gen, const loom_test_gen_type_palette_t* palette);

// Constructs a present constant attribute compatible with |type|'s element
// scalar. Integer-like scalars use I64 attributes, boolean i1 uses BOOL, and
// float-like scalars use F64.
loom_attribute_t loom_test_gen_constant_attr(loom_type_t type, int64_t value);

// Selects a scalar type from |palette| that satisfies |constraint|.
// Uses weighted random choice among matching entries only. Returns
// true and writes the type to |out_type| on success, false if no
// palette entry satisfies the constraint.
bool loom_test_gen_type_palette_pick_constrained(
    loom_test_gen_t* gen, const loom_test_gen_type_palette_t* palette,
    loom_type_constraint_t constraint, loom_type_t* out_type);

//===----------------------------------------------------------------------===//
// Live value set
//===----------------------------------------------------------------------===//

// Maximum number of live values the generator tracks. This bounds
// the number of SSA values that can be in scope at any point during
// generation. 4096 is sufficient for modules with thousands of ops
// (each op typically produces 1-2 results).
#define LOOM_TEST_GEN_VALUES_MAX_CAPACITY 4096

// Tracks live SSA values during generation, enabling the generator
// to select valid operands for new ops.
//
// Values are stored in insertion order in entries[] and types[].
// For type-matched lookups (e.g., "pick an integer-typed value"),
// per-scalar-type buckets provide O(1) selection after a lazy
// rebuild. The buckets_dirty flag tracks whether the bucket indices
// need rebuilding after new values have been added.
//
// Bucket layout: bucket_indices[] is a permutation of [0..count-1]
// grouped by scalar type. bucket_starts[t] and bucket_counts[t]
// give the range within bucket_indices[] for scalar type t. This
// avoids scanning all values when a specific type is requested.
typedef struct loom_test_gen_values_t {
  // Value IDs in insertion order.
  loom_value_id_t entries[LOOM_TEST_GEN_VALUES_MAX_CAPACITY];
  // Types corresponding to each entry, parallel array.
  loom_type_t types[LOOM_TEST_GEN_VALUES_MAX_CAPACITY];
  // Number of valid entries.
  uint16_t count;
  // Number of values offered to this set, including entries omitted after
  // reaching the fixed storage capacity.
  iree_host_size_t total_count;
  // Permutation of [0..count-1] sorted by scalar type for bucketed
  // lookup. Only valid when buckets_dirty is false.
  uint16_t bucket_indices[LOOM_TEST_GEN_VALUES_MAX_CAPACITY];
  // Start offset within bucket_indices[] for each scalar type.
  uint16_t bucket_starts[LOOM_SCALAR_TYPE_COUNT_];
  // Number of entries in each scalar type bucket.
  uint16_t bucket_counts[LOOM_SCALAR_TYPE_COUNT_];
  // True when entries have been added since the last bucket rebuild.
  // Bucket queries trigger a rebuild when this flag is set.
  bool buckets_dirty;
} loom_test_gen_values_t;

// Initializes a value set to empty. All counts and bucket data are
// zeroed.
void loom_test_gen_values_initialize(loom_test_gen_values_t* values);

// Adds a value to the set. Marks the bucket indices as dirty (they
// will be rebuilt on the next type-matched query). Silently drops
// the value if the set is at capacity.
void loom_test_gen_values_add(loom_test_gen_values_t* values,
                              loom_value_id_t id, loom_type_t type);

// Picks a random value from the entire set. Returns
// LOOM_VALUE_ID_INVALID if the set is empty.
loom_value_id_t loom_test_gen_values_pick_any(
    loom_test_gen_t* gen, const loom_test_gen_values_t* values);

// Picks a random value whose scalar type matches |type|. Rebuilds
// bucket indices if dirty. Only scalar values are eligible; shaped values with
// the same element type are intentionally excluded. Returns
// LOOM_VALUE_ID_INVALID if no values of the requested scalar type exist.
loom_value_id_t loom_test_gen_values_pick_typed(loom_test_gen_t* gen,
                                                loom_test_gen_values_t* values,
                                                loom_scalar_type_t type);

// Picks a random value whose type structurally equals |type|. This is the
// shape-preserving companion to loom_test_gen_values_pick_typed() and is used
// by hooks that need same-shaped vector/tile/tensor operands. Returns
// LOOM_VALUE_ID_INVALID if no value of the exact type exists.
loom_value_id_t loom_test_gen_values_pick_exact_type(
    loom_test_gen_t* gen, const loom_test_gen_values_t* values,
    loom_type_t type);

// Picks a random value with any integer scalar type (i1, i8, i16,
// i32, i64). Rebuilds bucket indices if dirty. Returns
// LOOM_VALUE_ID_INVALID if no integer-typed values exist.
loom_value_id_t loom_test_gen_values_pick_integer(
    loom_test_gen_t* gen, loom_test_gen_values_t* values);

// Picks a random value with any float scalar type (f8e4m3, f8e5m2,
// f16, bf16, f32, f64). Rebuilds bucket indices if dirty. Returns
// LOOM_VALUE_ID_INVALID if no float-typed values exist.
loom_value_id_t loom_test_gen_values_pick_float(loom_test_gen_t* gen,
                                                loom_test_gen_values_t* values);

// Returns the type of a value in the live set by scanning for its ID.
// Falls back to loom_type_scalar(I32) if not found (defensive for fuzzing).
loom_type_t loom_test_gen_values_type_of(const loom_test_gen_values_t* values,
                                         loom_value_id_t id);

// Picks two values of the same integer scalar type suitable for a binary
// op. Returns false if no integer type has at least one value.
bool loom_test_gen_values_pick_binary_integer(loom_test_gen_t* gen,
                                              loom_test_gen_values_t* values,
                                              loom_value_id_t* out_lhs,
                                              loom_value_id_t* out_rhs,
                                              loom_type_t* out_type);

// Picks two values of the same float scalar type suitable for a binary
// op. Returns false if no float type has at least one value.
bool loom_test_gen_values_pick_binary_float(loom_test_gen_t* gen,
                                            loom_test_gen_values_t* values,
                                            loom_value_id_t* out_lhs,
                                            loom_value_id_t* out_rhs,
                                            loom_type_t* out_type);

// Picks a value satisfying |constraint| from the live value set.
// Dispatches to the appropriate type-specific picker based on the
// constraint enum. Returns LOOM_VALUE_ID_INVALID if no matching value
// exists. Also writes the chosen value's type to |out_type| on success.
loom_value_id_t loom_test_gen_values_pick_for_constraint(
    loom_test_gen_t* gen, loom_test_gen_values_t* values,
    loom_type_constraint_t constraint, loom_type_t* out_type);

// Picks two values of the same type satisfying |constraint|, suitable
// for a binary op. Returns false if no type satisfying the constraint
// has at least one value.
bool loom_test_gen_values_pick_binary_for_constraint(
    loom_test_gen_t* gen, loom_test_gen_values_t* values,
    loom_type_constraint_t constraint, loom_value_id_t* out_lhs,
    loom_value_id_t* out_rhs, loom_type_t* out_type);

//===----------------------------------------------------------------------===//
// Hook interface
//===----------------------------------------------------------------------===//

// Configuration for generating a function body. Defined below in
// the configs section. Passed to hooks so they can inspect nesting
// limits, probabilities, and palette settings.
typedef struct loom_test_gen_body_config_t loom_test_gen_body_config_t;

// Context passed to every op generation hook. Provides the hook with
// everything it needs to emit ops: the randomness source, builder
// for IR construction, live values for operand selection, type
// palette for type generation, body config for nesting decisions,
// and current generation state (depth, remaining op budget).
typedef struct loom_test_gen_hook_context_t {
  // Randomness source for all generation decisions.
  loom_test_gen_t* gen;
  // Builder positioned at the insertion point for new ops.
  loom_builder_t* builder;
  // Live values in scope. Hooks add results of created ops here.
  loom_test_gen_values_t* values;
  // Type palette for generating operand/result types.
  const loom_test_gen_type_palette_t* palette;
  // Body config governing nesting, probabilities, and limits.
  const loom_test_gen_body_config_t* body_config;
  // Current nesting depth. Hooks that create nested regions should
  // check this against body_config->max_nesting_depth before
  // recursing.
  uint16_t current_depth;
  // Number of ops remaining in the current generation budget.
  // Hooks that emit multiple ops should respect this limit.
  uint16_t ops_remaining;
} loom_test_gen_hook_context_t;

// Result of an op generation hook attempt.
typedef enum loom_test_gen_hook_result_e {
  LOOM_TEST_GEN_HOOK_EMITTED = 0,  // Op(s) emitted successfully.
  LOOM_TEST_GEN_HOOK_SKIPPED = 1,  // Can't emit (no suitable values, etc).
} loom_test_gen_hook_result_t;

// Signature for an op generation hook. Receives the generation
// context and hook-specific user data. The hook emits one or more
// ops via the builder, adds any results to the value set, and
// writes the result to |out_result|. Returns iree_ok_status() on
// both EMITTED and SKIPPED — only returns an error status on a
// real failure (builder error, allocation failure, etc.).
typedef iree_status_t (*loom_test_gen_hook_fn_t)(
    const loom_test_gen_hook_context_t* context, void* user_data,
    loom_test_gen_hook_result_t* out_result);

// Signature for a type registration hook. Called during palette
// initialization to allow hooks to add dialect-specific types to
// the palette. May be NULL if the hook uses only standard types.
typedef void (*loom_test_gen_hook_register_types_fn_t)(
    loom_test_gen_type_palette_t* palette, void* user_data);

// A single op generation hook with its selection weight and optional
// type registration callback.
typedef struct loom_test_gen_op_hook_t {
  // Relative weight for weighted random selection. Higher weight
  // means this hook fires more often. A weight of 0 disables the
  // hook without removing it from the array.
  uint16_t weight;
  // Called to generate one or more ops. Must not be NULL.
  loom_test_gen_hook_fn_t generate;
  // Called during palette setup to register types this hook needs.
  // May be NULL if the hook uses only the default palette types.
  loom_test_gen_hook_register_types_fn_t register_types;
  // Opaque user data passed to both callbacks.
  void* user_data;
} loom_test_gen_op_hook_t;

// Maximum number of hooks in a body config's inline hooks array.
// Sufficient to hold any concatenation of built-in dialect hook sets.
#define LOOM_TEST_GEN_MAX_BUILTIN_HOOKS 32

//===----------------------------------------------------------------------===//
// Configs
//===----------------------------------------------------------------------===//

// Configuration for generating a function body (the ops within a
// single function). Controls the number and shape of generated ops,
// nesting behavior, and the distribution of interesting IR patterns
// (dead code, duplicate ops, casts).
struct loom_test_gen_body_config_t {
  // Target number of ops to generate in the body. The actual count
  // may vary slightly due to multi-op hooks and nesting.
  uint16_t op_count;
  // Maximum nesting depth for ops that contain regions (loops,
  // conditionals). Depth 0 means no nested regions are generated.
  uint16_t max_nesting_depth;
  // Probability (0-100) that a generation step produces a nested
  // op (one containing a region) instead of a flat op, when the
  // current depth is below max_nesting_depth.
  uint8_t nesting_probability;
  // Probability (0-100) that a generated op's results are not used
  // by subsequent ops, creating dead code for DCE to eliminate.
  uint8_t dead_op_probability;
  // Probability (0-100) that a generated op duplicates an existing
  // op (same opcode and operands), creating redundancy for CSE.
  uint8_t duplicate_probability;
  // Probability (0-100) that a generated op performs a write to a
  // resource. Reserved for future use when write-effect ops are
  // supported in the generator.
  uint8_t write_probability;
  // Probability (0-100) that a generated op is a type cast between
  // compatible scalar types. Reserved for future use.
  uint8_t cast_probability;
  // Number of block arguments to create at the function entry block.
  // These serve as the initial live values for op operand selection.
  uint16_t block_arg_count;
  // Maximum number of times a single value may be used as an operand
  // across generated ops. Limits fan-out to control use-list sizes.
  // Range: 1-8.
  uint8_t value_fan_out;
  // Type palette controlling the distribution of scalar types in
  // generated block arguments, op results, and constants.
  loom_test_gen_type_palette_t palette;
  // Op generation hooks. The generator selects hooks by weighted
  // random choice at each generation step. Inline storage so configs
  // are self-contained and copyable by value.
  loom_test_gen_op_hook_t hooks[LOOM_TEST_GEN_MAX_BUILTIN_HOOKS];
  // Number of valid entries in the hooks array.
  iree_host_size_t hook_count;
};

// Configuration for generating a complete module with multiple
// functions, declarations, and inter-function calls.
typedef struct loom_test_gen_module_config_t {
  // Number of function definitions (functions with bodies) to
  // generate.
  uint16_t function_count;
  // Number of function declarations (functions without bodies) to
  // generate. These can be referenced as call targets.
  uint16_t declaration_count;
  // Number of call ops to insert per function body. Calls target
  // other functions in the module (both definitions and declarations).
  uint16_t calls_per_function;
  // Body config applied to each generated function.
  loom_test_gen_body_config_t body_config;
} loom_test_gen_module_config_t;

// Body generation with explicit depth tracking. Region hooks call
// this to generate nested bodies at depth+1.
iree_status_t loom_test_gen_body_internal(
    loom_test_gen_t* gen, const loom_test_gen_body_config_t* config,
    loom_builder_t* builder, loom_test_gen_values_t* values,
    uint16_t current_depth);

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

// Generates a function body. The builder must be positioned at a
// function entry block. Creates |config->block_arg_count| block
// arguments with types drawn from the palette, then fills the block
// with ops according to the config and hooks.
//
// |values| is populated with all generated values (block args and
// op results). The caller may pre-populate it with values from an
// enclosing scope if generating a nested region body.
iree_status_t loom_test_gen_body(loom_test_gen_t* gen,
                                 const loom_test_gen_body_config_t* config,
                                 loom_builder_t* builder,
                                 loom_test_gen_values_t* values);

// Generates a complete module with functions, declarations, and
// inter-function calls. Allocates a new module from |block_pool|
// using the context's allocator. The module is returned through
// |out_module| and must be freed with loom_module_free().
iree_status_t loom_test_gen_module(loom_test_gen_t* gen,
                                   const loom_test_gen_module_config_t* config,
                                   loom_context_t* context,
                                   iree_arena_block_pool_t* block_pool,
                                   loom_module_t** out_module);

//===----------------------------------------------------------------------===//
// Built-in hooks
//===----------------------------------------------------------------------===//

// Returns the built-in scalar dialect op hooks. The returned array
// is in static storage and valid for the lifetime of the program.
// Scalar hooks generate arithmetic, comparison, conversion, and
// constant ops. |out_hook_count| receives the number of hooks.
const loom_test_gen_op_hook_t* loom_test_gen_scalar_hooks(
    iree_host_size_t* out_hook_count);

// Returns the built-in test dialect op hooks. The returned array
// is in static storage and valid for the lifetime of the program.
// Test hooks generate binary/unary ops, side-effect ops, and
// region-bearing ops from the test dialect. |out_hook_count|
// receives the number of hooks.
const loom_test_gen_op_hook_t* loom_test_gen_test_hooks(
    iree_host_size_t* out_hook_count);

// Returns the built-in vector dialect op hooks. The returned array
// is in static storage and valid for the lifetime of the program.
// Vector hooks generate target-agnostic vector construction, arithmetic, and
// extraction ops. |out_hook_count| receives the number of hooks.
const loom_test_gen_op_hook_t* loom_test_gen_vector_hooks(
    iree_host_size_t* out_hook_count);

//===----------------------------------------------------------------------===//
// Presets
//===----------------------------------------------------------------------===//

// Body config for representative IR. Produces a balanced mix of op
// kinds, moderate nesting, some dead code, and occasional duplicates.
// Scale 1 produces tens of ops; scale 10 produces thousands. The returned
// config owns an inline copy of the selected hook table.
loom_test_gen_body_config_t loom_test_gen_body_config_representative(
    uint32_t scale);

// Body config that maximizes CSE opportunities. High duplicate
// probability, low nesting, and repeated use of the same operands.
// Useful for benchmarking and testing the CSE pass.
loom_test_gen_body_config_t loom_test_gen_body_config_cse_stress(
    uint32_t scale);

// Body config that maximizes DCE opportunities. High dead-op
// probability and low fan-out, creating many unused results for
// the DCE pass to eliminate.
loom_test_gen_body_config_t loom_test_gen_body_config_dce_stress(
    uint32_t scale);

// Body config that maximizes nesting depth. High nesting probability
// and generous max depth, creating deeply nested region structures
// for testing recursive pass traversal and printer indentation.
loom_test_gen_body_config_t loom_test_gen_body_config_nesting_stress(
    uint32_t scale);

// Body config that stresses the text format parser and printer.
// High op count with many block arguments, large fan-out, and
// varied types to exercise all format element kinds.
loom_test_gen_body_config_t loom_test_gen_body_config_format_stress(
    uint32_t scale);

// Module config for representative IR. Produces a module with
// multiple functions of varying signatures, some declarations,
// and inter-function calls. Scale controls both function count
// and per-function body size.
loom_test_gen_module_config_t loom_test_gen_module_config_representative(
    uint32_t scale);

// Module config selected from the built-in fuzz presets. The preset index is
// modulo the available preset count. Scale 0 selects 1, and oversized values
// are clamped to keep fuzz iterations bounded.
loom_test_gen_module_config_t loom_test_gen_module_config_fuzz_preset(
    uint8_t preset_index, uint32_t scale);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TESTING_GEN_H_
