// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core generator infrastructure: randomness source, type palette, live
// value set, body generation loop, module generation, and presets.

#include "loom/testing/gen.h"

#include "iree/base/internal/prng.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

//===----------------------------------------------------------------------===//
// Randomness source
//===----------------------------------------------------------------------===//

void loom_test_gen_initialize_seeded(uint64_t seed, loom_test_gen_t* out_gen) {
  memset(out_gen, 0, sizeof(*out_gen));
  iree_prng_xoroshiro128_initialize(seed, &out_gen->prng);
  out_gen->fuzz_data = NULL;
  out_gen->fuzz_remaining = 0;
}

void loom_test_gen_initialize_fuzz(const uint8_t* data, iree_host_size_t size,
                                   loom_test_gen_t* out_gen) {
  memset(out_gen, 0, sizeof(*out_gen));
  // Seed PRNG from first 8 bytes so the generator has deterministic
  // fallback behavior when fuzz bytes are exhausted.
  uint64_t seed = 0;
  iree_host_size_t seed_bytes = size < 8 ? size : 8;
  if (seed_bytes > 0) {
    memcpy(&seed, data, seed_bytes);
  }
  iree_prng_xoroshiro128_initialize(seed, &out_gen->prng);
  out_gen->fuzz_data = data;
  out_gen->fuzz_remaining = size;
}

// Consumes |count| bytes from the fuzz buffer, zero-padding if fewer
// remain. Returns the bytes packed into a uint64_t (little-endian).
// When the buffer is fully exhausted, NULLs fuzz_data so subsequent
// calls fall through to the PRNG — this prevents infinite loops in
// rejection sampling which requires varying input to make progress.
static uint64_t loom_test_gen_consume_fuzz(loom_test_gen_t* gen,
                                           iree_host_size_t count) {
  uint64_t result = 0;
  iree_host_size_t available =
      gen->fuzz_remaining < count ? gen->fuzz_remaining : count;
  if (available > 0) {
    memcpy(&result, gen->fuzz_data, available);
  }
  gen->fuzz_data += available;
  gen->fuzz_remaining -= available;
  if (gen->fuzz_remaining == 0) {
    gen->fuzz_data = NULL;
  }
  return result;
}

uint64_t loom_test_gen_next_uint64(loom_test_gen_t* gen) {
  if (gen->fuzz_data) {
    return loom_test_gen_consume_fuzz(gen, 8);
  }
  // xoroshiro128plus returns 60 usable bits. We use all 64 bits of the
  // state output — the low bits have adequate quality for our
  // non-cryptographic use case.
  return iree_prng_xoroshiro128plus_next_uint60(&gen->prng);
}

uint32_t loom_test_gen_next_uint32(loom_test_gen_t* gen) {
  if (gen->fuzz_data) {
    return (uint32_t)loom_test_gen_consume_fuzz(gen, 4);
  }
  return (uint32_t)(iree_prng_xoroshiro128plus_next_uint60(&gen->prng) >> 28);
}

uint32_t loom_test_gen_next_range(loom_test_gen_t* gen,
                                  uint32_t upper_exclusive) {
  if (upper_exclusive <= 1) return 0;
  // Fast path for powers of two.
  if ((upper_exclusive & (upper_exclusive - 1)) == 0) {
    return loom_test_gen_next_uint32(gen) & (upper_exclusive - 1);
  }
  // Rejection sampling to avoid modulo bias. The biased region is
  // [0, 2^32 % upper_exclusive). We reject values in that range.
  uint32_t threshold = (UINT32_MAX - upper_exclusive + 1) % upper_exclusive;
  for (;;) {
    uint32_t value = loom_test_gen_next_uint32(gen);
    if (value >= threshold) {
      return value % upper_exclusive;
    }
  }
}

bool loom_test_gen_next_bool(loom_test_gen_t* gen) {
  return (loom_test_gen_next_uint32(gen) & 1) != 0;
}

bool loom_test_gen_next_probability(loom_test_gen_t* gen, uint8_t percent) {
  if (percent == 0) return false;
  if (percent >= 100) return true;
  return loom_test_gen_next_range(gen, 100) < percent;
}

//===----------------------------------------------------------------------===//
// Type palette
//===----------------------------------------------------------------------===//

void loom_test_gen_type_palette_default(
    loom_test_gen_type_palette_t* out_palette) {
  memset(out_palette, 0, sizeof(*out_palette));
  // Default distribution emphasizing common compute types.
  // Excludes OFFSET (rare), I1 (boolean, special-case), and F8 types
  // (exotic, narrow support). These can be added by hooks via
  // register_types if needed.
  static const struct {
    loom_scalar_type_t type;
    uint16_t weight;
  } defaults[] = {
      {LOOM_SCALAR_TYPE_I8, 3},    {LOOM_SCALAR_TYPE_I16, 3},
      {LOOM_SCALAR_TYPE_I32, 5},   {LOOM_SCALAR_TYPE_I64, 3},
      {LOOM_SCALAR_TYPE_F16, 1},   {LOOM_SCALAR_TYPE_BF16, 1},
      {LOOM_SCALAR_TYPE_F32, 2},   {LOOM_SCALAR_TYPE_F64, 1},
      {LOOM_SCALAR_TYPE_INDEX, 2},
  };
  uint16_t count = (uint16_t)IREE_ARRAYSIZE(defaults);
  uint16_t total = 0;
  for (uint16_t i = 0; i < count; ++i) {
    out_palette->types[i] = defaults[i].type;
    out_palette->weights[i] = defaults[i].weight;
    total += defaults[i].weight;
  }
  out_palette->count = count;
  out_palette->total_weight = total;
}

loom_type_t loom_test_gen_type_palette_pick(
    loom_test_gen_t* gen, const loom_test_gen_type_palette_t* palette) {
  IREE_ASSERT(palette->count > 0);
  uint32_t target = loom_test_gen_next_range(gen, palette->total_weight);
  uint32_t cumulative = 0;
  for (uint16_t i = 0; i < palette->count; ++i) {
    cumulative += palette->weights[i];
    if (target < cumulative) {
      return loom_type_scalar(palette->types[i]);
    }
  }
  return loom_type_scalar(palette->types[palette->count - 1]);
}

loom_attribute_t loom_test_gen_constant_attr(loom_type_t type, int64_t value) {
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    return loom_attr_bool((value & 1) != 0);
  } else if (loom_scalar_type_is_float(scalar_type)) {
    return loom_attr_f64((double)value * 0.1);
  }
  return loom_attr_i64(value);
}

bool loom_test_gen_type_palette_pick_constrained(
    loom_test_gen_t* gen, const loom_test_gen_type_palette_t* palette,
    loom_type_constraint_t constraint, loom_type_t* out_type) {
  // Compute the total weight of matching entries.
  uint32_t matching_weight = 0;
  for (uint16_t i = 0; i < palette->count; ++i) {
    loom_type_t type = loom_type_scalar(palette->types[i]);
    if (loom_type_satisfies_constraint(type, constraint)) {
      matching_weight += palette->weights[i];
    }
  }
  if (matching_weight == 0) return false;
  // Weighted pick among matching entries only.
  uint32_t target = loom_test_gen_next_range(gen, matching_weight);
  uint32_t cumulative = 0;
  for (uint16_t i = 0; i < palette->count; ++i) {
    loom_type_t type = loom_type_scalar(palette->types[i]);
    if (!loom_type_satisfies_constraint(type, constraint)) continue;
    cumulative += palette->weights[i];
    if (target < cumulative) {
      *out_type = type;
      return true;
    }
  }
  // Should not reach here, but handle gracefully.
  for (uint16_t i = 0; i < palette->count; ++i) {
    loom_type_t type = loom_type_scalar(palette->types[i]);
    if (loom_type_satisfies_constraint(type, constraint)) {
      *out_type = type;
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Live value set
//===----------------------------------------------------------------------===//

void loom_test_gen_values_initialize(loom_test_gen_values_t* values) {
  memset(values, 0, sizeof(*values));
  values->buckets_dirty = true;
}

void loom_test_gen_values_add(loom_test_gen_values_t* values,
                              loom_value_id_t id, loom_type_t type) {
  ++values->total_count;
  if (values->count >= LOOM_TEST_GEN_VALUES_MAX_CAPACITY) return;
  values->entries[values->count] = id;
  values->types[values->count] = type;
  values->count++;
  values->buckets_dirty = true;
}

loom_value_id_t loom_test_gen_values_pick_any(
    loom_test_gen_t* gen, const loom_test_gen_values_t* values) {
  if (values->count == 0) return LOOM_VALUE_ID_INVALID;
  uint32_t index = loom_test_gen_next_range(gen, values->count);
  return values->entries[index];
}

// Rebuilds per-scalar-type bucket indices from the entries array.
static void loom_test_gen_values_rebuild_buckets(
    loom_test_gen_values_t* values) {
  if (!values->buckets_dirty) return;
  memset(values->bucket_counts, 0, sizeof(values->bucket_counts));
  // Count entries per scalar type.
  for (uint16_t i = 0; i < values->count; ++i) {
    loom_type_t type = values->types[i];
    if (loom_type_is_scalar(type)) {
      loom_scalar_type_t scalar = loom_type_element_type(type);
      if (scalar < LOOM_SCALAR_TYPE_COUNT_) {
        values->bucket_counts[scalar]++;
      }
    }
  }
  // Compute start offsets (exclusive prefix sum).
  uint16_t offset = 0;
  for (uint16_t t = 0; t < LOOM_SCALAR_TYPE_COUNT_; ++t) {
    values->bucket_starts[t] = offset;
    offset += values->bucket_counts[t];
  }
  // Fill bucket_indices by second pass.
  uint16_t fill_counts[LOOM_SCALAR_TYPE_COUNT_] = {0};
  for (uint16_t i = 0; i < values->count; ++i) {
    loom_type_t type = values->types[i];
    if (loom_type_is_scalar(type)) {
      loom_scalar_type_t scalar = loom_type_element_type(type);
      if (scalar < LOOM_SCALAR_TYPE_COUNT_) {
        uint16_t bucket_offset =
            values->bucket_starts[scalar] + fill_counts[scalar];
        values->bucket_indices[bucket_offset] = i;
        fill_counts[scalar]++;
      }
    }
  }
  values->buckets_dirty = false;
}

loom_value_id_t loom_test_gen_values_pick_typed(loom_test_gen_t* gen,
                                                loom_test_gen_values_t* values,
                                                loom_scalar_type_t type) {
  loom_test_gen_values_rebuild_buckets(values);
  if (type >= LOOM_SCALAR_TYPE_COUNT_) return LOOM_VALUE_ID_INVALID;
  uint16_t bucket_count = values->bucket_counts[type];
  if (bucket_count == 0) return LOOM_VALUE_ID_INVALID;
  uint32_t pick = loom_test_gen_next_range(gen, bucket_count);
  uint16_t entry_index =
      values->bucket_indices[values->bucket_starts[type] + pick];
  return values->entries[entry_index];
}

loom_value_id_t loom_test_gen_values_pick_exact_type(
    loom_test_gen_t* gen, const loom_test_gen_values_t* values,
    loom_type_t type) {
  uint16_t candidate_count = 0;
  for (uint16_t i = 0; i < values->count; ++i) {
    if (loom_type_equal(values->types[i], type)) {
      ++candidate_count;
    }
  }
  if (candidate_count == 0) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint32_t pick = loom_test_gen_next_range(gen, candidate_count);
  for (uint16_t i = 0; i < values->count; ++i) {
    if (!loom_type_equal(values->types[i], type)) {
      continue;
    }
    if (pick == 0) {
      return values->entries[i];
    }
    --pick;
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_test_gen_values_pick_integer(
    loom_test_gen_t* gen, loom_test_gen_values_t* values) {
  loom_test_gen_values_rebuild_buckets(values);
  // Count total integer values across I1, I8, I16, I32, I64.
  uint16_t total = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_I1; t <= LOOM_SCALAR_TYPE_I64;
       ++t) {
    total += values->bucket_counts[t];
  }
  if (total == 0) return LOOM_VALUE_ID_INVALID;
  uint32_t pick = loom_test_gen_next_range(gen, total);
  uint16_t cumulative = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_I1; t <= LOOM_SCALAR_TYPE_I64;
       ++t) {
    uint16_t bucket_count = values->bucket_counts[t];
    if (pick < cumulative + bucket_count) {
      uint16_t entry_index =
          values->bucket_indices[values->bucket_starts[t] + pick - cumulative];
      return values->entries[entry_index];
    }
    cumulative += bucket_count;
  }
  return LOOM_VALUE_ID_INVALID;
}

loom_value_id_t loom_test_gen_values_pick_float(
    loom_test_gen_t* gen, loom_test_gen_values_t* values) {
  loom_test_gen_values_rebuild_buckets(values);
  // Count total float values across F8E4M3..F64.
  uint16_t total = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_F8E4M3;
       t <= LOOM_SCALAR_TYPE_F64; ++t) {
    total += values->bucket_counts[t];
  }
  if (total == 0) return LOOM_VALUE_ID_INVALID;
  uint32_t pick = loom_test_gen_next_range(gen, total);
  uint16_t cumulative = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_F8E4M3;
       t <= LOOM_SCALAR_TYPE_F64; ++t) {
    uint16_t bucket_count = values->bucket_counts[t];
    if (pick < cumulative + bucket_count) {
      uint16_t entry_index =
          values->bucket_indices[values->bucket_starts[t] + pick - cumulative];
      return values->entries[entry_index];
    }
    cumulative += bucket_count;
  }
  return LOOM_VALUE_ID_INVALID;
}

//===----------------------------------------------------------------------===//
// Value lookup helpers
//===----------------------------------------------------------------------===//

loom_type_t loom_test_gen_values_type_of(const loom_test_gen_values_t* values,
                                         loom_value_id_t id) {
  for (uint16_t i = 0; i < values->count; ++i) {
    if (values->entries[i] == id) return values->types[i];
  }
  return loom_type_scalar(LOOM_SCALAR_TYPE_I32);
}

// Picks a value from the set whose type satisfies |constraint|.
// Scans the live value set and collects candidates, then picks one
// randomly. Returns LOOM_VALUE_ID_INVALID if none match.
loom_value_id_t loom_test_gen_values_pick_for_constraint(
    loom_test_gen_t* gen, loom_test_gen_values_t* values,
    loom_type_constraint_t constraint, loom_type_t* out_type) {
  // Fast paths for common scalar constraints that have bucket support.
  switch (constraint) {
    case LOOM_TYPE_CONSTRAINT_INTEGER: {
      loom_value_id_t id = loom_test_gen_values_pick_integer(gen, values);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_test_gen_values_type_of(values, id);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_FLOAT: {
      loom_value_id_t id = loom_test_gen_values_pick_float(gen, values);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_test_gen_values_type_of(values, id);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_INDEX: {
      loom_value_id_t id =
          loom_test_gen_values_pick_typed(gen, values, LOOM_SCALAR_TYPE_INDEX);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_OFFSET: {
      loom_value_id_t id =
          loom_test_gen_values_pick_typed(gen, values, LOOM_SCALAR_TYPE_OFFSET);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_ADDRESS: {
      loom_scalar_type_t scalar_type = loom_test_gen_next_bool(gen)
                                           ? LOOM_SCALAR_TYPE_INDEX
                                           : LOOM_SCALAR_TYPE_OFFSET;
      loom_value_id_t id =
          loom_test_gen_values_pick_typed(gen, values, scalar_type);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_type_scalar(scalar_type);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_I1: {
      loom_value_id_t id =
          loom_test_gen_values_pick_typed(gen, values, LOOM_SCALAR_TYPE_I1);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_type_scalar(LOOM_SCALAR_TYPE_I1);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_ANY: {
      loom_value_id_t id = loom_test_gen_values_pick_any(gen, values);
      if (id != LOOM_VALUE_ID_INVALID && out_type) {
        *out_type = loom_test_gen_values_type_of(values, id);
      }
      return id;
    }
    case LOOM_TYPE_CONSTRAINT_SCALAR:
      break;
    default:
      break;
  }
  // General path: scan all values for those satisfying the constraint.
  // This handles SCALAR, VECTOR, TILE, TENSOR, POOL, GROUP, ENCODING, and any
  // future constraints without needing per-constraint bucket infrastructure.
  uint16_t candidates[LOOM_TEST_GEN_VALUES_MAX_CAPACITY];
  uint16_t candidate_count = 0;
  for (uint16_t i = 0; i < values->count; ++i) {
    if (loom_type_satisfies_constraint(values->types[i], constraint)) {
      candidates[candidate_count++] = i;
    }
  }
  if (candidate_count == 0) {
    return LOOM_VALUE_ID_INVALID;
  }
  uint16_t chosen = candidates[loom_test_gen_next_range(gen, candidate_count)];
  if (out_type) {
    *out_type = values->types[chosen];
  }
  return values->entries[chosen];
}

bool loom_test_gen_values_pick_binary_for_constraint(
    loom_test_gen_t* gen, loom_test_gen_values_t* values,
    loom_type_constraint_t constraint, loom_value_id_t* out_lhs,
    loom_value_id_t* out_rhs, loom_type_t* out_type) {
  // Fast paths for integer and float — these use the bucket-based
  // same-type pairing that ensures both operands match.
  switch (constraint) {
    case LOOM_TYPE_CONSTRAINT_INTEGER:
      return loom_test_gen_values_pick_binary_integer(gen, values, out_lhs,
                                                      out_rhs, out_type);
    case LOOM_TYPE_CONSTRAINT_FLOAT:
      return loom_test_gen_values_pick_binary_float(gen, values, out_lhs,
                                                    out_rhs, out_type);
    default:
      break;
  }
  // General path: pick first, then pick second of the same exact type.
  loom_type_t type = {0};
  loom_value_id_t first =
      loom_test_gen_values_pick_for_constraint(gen, values, constraint, &type);
  if (first == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  loom_value_id_t second =
      loom_test_gen_values_pick_exact_type(gen, values, type);
  if (second == LOOM_VALUE_ID_INVALID) {
    second = first;
  }
  *out_lhs = first;
  *out_rhs = second;
  *out_type = type;
  return true;
}

bool loom_test_gen_values_pick_binary_integer(loom_test_gen_t* gen,
                                              loom_test_gen_values_t* values,
                                              loom_value_id_t* out_lhs,
                                              loom_value_id_t* out_rhs,
                                              loom_type_t* out_type) {
  loom_test_gen_values_rebuild_buckets(values);
  loom_scalar_type_t candidates[LOOM_SCALAR_TYPE_I64 - LOOM_SCALAR_TYPE_I1 + 1];
  uint16_t candidate_count = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_I1; t <= LOOM_SCALAR_TYPE_I64;
       ++t) {
    if (values->bucket_counts[t] > 0) {
      candidates[candidate_count++] = t;
    }
  }
  if (candidate_count == 0) return false;
  loom_scalar_type_t chosen =
      candidates[loom_test_gen_next_range(gen, candidate_count)];
  *out_lhs = loom_test_gen_values_pick_typed(gen, values, chosen);
  *out_rhs = loom_test_gen_values_pick_typed(gen, values, chosen);
  *out_type = loom_type_scalar(chosen);
  return true;
}

bool loom_test_gen_values_pick_binary_float(loom_test_gen_t* gen,
                                            loom_test_gen_values_t* values,
                                            loom_value_id_t* out_lhs,
                                            loom_value_id_t* out_rhs,
                                            loom_type_t* out_type) {
  loom_test_gen_values_rebuild_buckets(values);
  loom_scalar_type_t
      candidates[LOOM_SCALAR_TYPE_F64 - LOOM_SCALAR_TYPE_F8E4M3 + 1];
  uint16_t candidate_count = 0;
  for (loom_scalar_type_t t = LOOM_SCALAR_TYPE_F8E4M3;
       t <= LOOM_SCALAR_TYPE_F64; ++t) {
    if (values->bucket_counts[t] > 0) {
      candidates[candidate_count++] = t;
    }
  }
  if (candidate_count == 0) return false;
  loom_scalar_type_t chosen =
      candidates[loom_test_gen_next_range(gen, candidate_count)];
  *out_lhs = loom_test_gen_values_pick_typed(gen, values, chosen);
  *out_rhs = loom_test_gen_values_pick_typed(gen, values, chosen);
  *out_type = loom_type_scalar(chosen);
  return true;
}

//===----------------------------------------------------------------------===//
// Body generation
//===----------------------------------------------------------------------===//

// Selects a hook by weighted random choice. Returns the hook index.
static iree_host_size_t loom_test_gen_select_hook(
    loom_test_gen_t* gen, const loom_test_gen_op_hook_t* hooks,
    iree_host_size_t hook_count) {
  uint32_t total_weight = 0;
  for (iree_host_size_t i = 0; i < hook_count; ++i) {
    total_weight += hooks[i].weight;
  }
  if (total_weight == 0) return 0;
  uint32_t target = loom_test_gen_next_range(gen, total_weight);
  uint32_t cumulative = 0;
  for (iree_host_size_t i = 0; i < hook_count; ++i) {
    cumulative += hooks[i].weight;
    if (target < cumulative) return i;
  }
  return hook_count - 1;
}

iree_status_t loom_test_gen_body_internal(
    loom_test_gen_t* gen, const loom_test_gen_body_config_t* config,
    loom_builder_t* builder, loom_test_gen_values_t* values,
    uint16_t current_depth) {
  if (config->hook_count == 0) return iree_ok_status();

  uint16_t consecutive_failures = 0;
  uint16_t max_consecutive = (uint16_t)(config->hook_count * 3);

  for (uint16_t i = 0; i < config->op_count; ++i) {
    uint16_t ops_remaining = config->op_count - i;

    loom_test_gen_hook_context_t hook_context = {
        .gen = gen,
        .builder = builder,
        .values = values,
        .palette = &config->palette,
        .body_config = config,
        .current_depth = current_depth,
        .ops_remaining = ops_remaining,
    };

    // Select and invoke a hook.
    iree_host_size_t hook_index =
        loom_test_gen_select_hook(gen, config->hooks, config->hook_count);
    loom_test_gen_hook_result_t hook_result = LOOM_TEST_GEN_HOOK_SKIPPED;
    IREE_RETURN_IF_ERROR(config->hooks[hook_index].generate(
        &hook_context, config->hooks[hook_index].user_data, &hook_result));

    if (hook_result == LOOM_TEST_GEN_HOOK_SKIPPED) {
      consecutive_failures++;

      // After too many failures, force a constant to broaden the value set.
      if (consecutive_failures >= max_consecutive) {
        loom_type_t type =
            loom_test_gen_type_palette_pick(gen, &config->palette);
        loom_attribute_t const_val = loom_test_gen_constant_attr(
            type, (int64_t)loom_test_gen_next_range(gen, 100));
        loom_op_t* const_op = NULL;
        IREE_RETURN_IF_ERROR(loom_test_constant_build(
            builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
        loom_test_gen_values_add(values, loom_op_results(const_op)[0], type);
        consecutive_failures = 0;
      } else {
        --i;  // Retry this iteration with a different hook.
      }
      continue;
    }

    consecutive_failures = 0;

    // Maybe mark the last result as dead (don't add to values — already
    // added by the hook, so we remove it).
    if (loom_test_gen_next_probability(gen, config->dead_op_probability)) {
      if (values->count > 0) {
        values->count--;
        values->buckets_dirty = true;
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_test_gen_body(loom_test_gen_t* gen,
                                 const loom_test_gen_body_config_t* config,
                                 loom_builder_t* builder,
                                 loom_test_gen_values_t* values) {
  // Create block arguments with types from the palette.
  loom_block_t* block = builder->ip.block;
  for (uint16_t i = 0; i < config->block_arg_count; ++i) {
    loom_type_t type = loom_test_gen_type_palette_pick(gen, &config->palette);
    loom_value_id_t arg_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_builder_define_block_arg(builder, block, type, &arg_id));
    loom_test_gen_values_add(values, arg_id, type);
  }

  // Seed the value set if sparse: one constant per palette type.
  if (values->count < config->palette.count) {
    for (uint16_t i = 0; i < config->palette.count; ++i) {
      loom_type_t type = loom_type_scalar(config->palette.types[i]);
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(type, (int64_t)i);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_constant_build(
          builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
      loom_test_gen_values_add(values, loom_op_results(const_op)[0], type);
    }
  }

  return loom_test_gen_body_internal(gen, config, builder, values, 0);
}

//===----------------------------------------------------------------------===//
// Module generation
//===----------------------------------------------------------------------===//

// Generates a deterministic symbol name from index.
static iree_status_t loom_test_gen_symbol_name(loom_builder_t* builder,
                                               const char* prefix,
                                               uint16_t index,
                                               loom_symbol_ref_t* out_ref) {
  char name[64];
  iree_snprintf(name, sizeof(name), "%s%u", prefix, index);
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      builder, iree_make_cstring_view(name), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_add_symbol(builder->module, name_id, &symbol_id));
  *out_ref = (loom_symbol_ref_t){
      .module_id = 0,
      .symbol_id = symbol_id,
  };
  return iree_ok_status();
}

// Maximum number of arguments and results for generated function signatures.
#define LOOM_TEST_GEN_MAX_FUNC_ARGS 8
#define LOOM_TEST_GEN_MAX_FUNC_RESULTS 4

// Signature record for tracking generated functions and their types.
typedef struct loom_test_gen_func_sig_t {
  loom_symbol_ref_t ref;
  loom_type_t arg_types[LOOM_TEST_GEN_MAX_FUNC_ARGS];
  loom_type_t result_types[LOOM_TEST_GEN_MAX_FUNC_RESULTS];
  uint16_t arg_count;
  uint16_t result_count;
} loom_test_gen_func_sig_t;

iree_status_t loom_test_gen_module(loom_test_gen_t* gen,
                                   const loom_test_gen_module_config_t* config,
                                   loom_context_t* context,
                                   iree_arena_block_pool_t* block_pool,
                                   loom_module_t** out_module) {
  loom_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      loom_module_allocate(context, iree_make_cstring_view("gen"), block_pool,
                           NULL, context->allocator, &module));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  uint16_t total_functions = config->function_count + config->declaration_count;
  loom_test_gen_func_sig_t* signatures = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &module->arena, total_functions, sizeof(loom_test_gen_func_sig_t),
      (void**)&signatures));
  uint16_t signature_count = 0;

  // Generate function declarations (external symbols).
  for (uint16_t i = 0; i < config->declaration_count; ++i) {
    loom_test_gen_func_sig_t* sig = &signatures[signature_count];
    IREE_RETURN_IF_ERROR(
        loom_test_gen_symbol_name(&builder, "gen_decl_", i, &sig->ref));

    sig->arg_count = (uint16_t)(1 + loom_test_gen_next_range(gen, 4));
    sig->result_count = (uint16_t)loom_test_gen_next_range(gen, 3);
    for (uint16_t j = 0; j < sig->arg_count; ++j) {
      sig->arg_types[j] =
          loom_test_gen_type_palette_pick(gen, &config->body_config.palette);
    }
    for (uint16_t j = 0; j < sig->result_count; ++j) {
      sig->result_types[j] =
          loom_test_gen_type_palette_pick(gen, &config->body_config.palette);
    }

    loom_op_t* decl_op = NULL;
    IREE_RETURN_IF_ERROR(loom_test_decl_build(
        &builder, 0, 0, 0, sig->ref, sig->arg_types, sig->arg_count,
        sig->result_types, sig->result_count, NULL, 0, LOOM_LOCATION_UNKNOWN,
        &decl_op));
    signature_count++;
  }

  // Generate function definitions.
  for (uint16_t i = 0; i < config->function_count; ++i) {
    loom_test_gen_func_sig_t* sig = &signatures[signature_count];
    IREE_RETURN_IF_ERROR(
        loom_test_gen_symbol_name(&builder, "gen_func_", i, &sig->ref));

    sig->arg_count = (uint16_t)(1 + loom_test_gen_next_range(gen, 4));
    sig->result_count = (uint16_t)(1 + loom_test_gen_next_range(gen, 3));
    for (uint16_t j = 0; j < sig->arg_count; ++j) {
      sig->arg_types[j] =
          loom_test_gen_type_palette_pick(gen, &config->body_config.palette);
    }
    for (uint16_t j = 0; j < sig->result_count; ++j) {
      sig->result_types[j] =
          loom_test_gen_type_palette_pick(gen, &config->body_config.palette);
    }

    loom_op_t* def_op = NULL;
    IREE_RETURN_IF_ERROR(loom_test_func_build(
        &builder, 0, 0, 0, sig->ref, sig->arg_types, sig->arg_count,
        sig->result_types, sig->result_count, NULL, 0, NULL, 0,
        LOOM_LOCATION_UNKNOWN, &def_op));
    signature_count++;

    // Generate function body.
    loom_region_t* body = loom_test_func_body(def_op);
    loom_builder_ip_t saved = loom_builder_enter_region(&builder, def_op, body);

    loom_test_gen_values_t values;
    loom_test_gen_values_initialize(&values);

    // Add function block args to the value set.
    loom_block_t* entry_block = loom_region_entry_block(body);
    for (uint16_t j = 0; j < entry_block->arg_count; ++j) {
      loom_value_id_t arg_id = loom_block_arg_id(entry_block, j);
      loom_value_t* arg_val = &module->values.entries[arg_id];
      loom_test_gen_values_add(&values, arg_id, arg_val->type);
    }

    // Seed with constants for all palette types.
    for (uint16_t j = 0; j < config->body_config.palette.count; ++j) {
      loom_type_t type = loom_type_scalar(config->body_config.palette.types[j]);
      loom_attribute_t const_val =
          loom_test_gen_constant_attr(type, (int64_t)j);
      loom_op_t* const_op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_constant_build(
          &builder, const_val, type, LOOM_LOCATION_UNKNOWN, &const_op));
      loom_test_gen_values_add(&values, loom_op_results(const_op)[0], type);
    }

    // Run the body generator.
    IREE_RETURN_IF_ERROR(loom_test_gen_body_internal(gen, &config->body_config,
                                                     &builder, &values, 0));

    // Insert call-like ops for other functions.
    for (uint16_t c = 0; c < config->calls_per_function; ++c) {
      if (signature_count == 0) {
        break;
      }
      uint16_t target =
          (uint16_t)loom_test_gen_next_range(gen, signature_count);
      loom_test_gen_func_sig_t* target_sig = &signatures[target];

      loom_value_id_t call_operands[LOOM_TEST_GEN_MAX_FUNC_ARGS];
      for (uint16_t j = 0; j < target_sig->arg_count; ++j) {
        loom_scalar_type_t scalar =
            loom_type_element_type(target_sig->arg_types[j]);
        call_operands[j] =
            loom_test_gen_values_pick_typed(gen, &values, scalar);
        if (call_operands[j] == LOOM_VALUE_ID_INVALID) {
          call_operands[j] = loom_test_gen_values_pick_any(gen, &values);
        }
        if (call_operands[j] == LOOM_VALUE_ID_INVALID) {
          loom_attribute_t const_val =
              loom_test_gen_constant_attr(target_sig->arg_types[j], (int64_t)j);
          loom_op_t* const_op = NULL;
          IREE_RETURN_IF_ERROR(loom_test_constant_build(
              &builder, const_val, target_sig->arg_types[j],
              LOOM_LOCATION_UNKNOWN, &const_op));
          call_operands[j] = loom_op_results(const_op)[0];
          loom_test_gen_values_add(&values, call_operands[j],
                                   target_sig->arg_types[j]);
        }
      }

      loom_op_t* call_op = NULL;
      IREE_RETURN_IF_ERROR(loom_test_invoke_build(
          &builder, target_sig->ref, call_operands, target_sig->arg_count,
          target_sig->result_types, target_sig->result_count, NULL, 0,
          LOOM_LOCATION_UNKNOWN, &call_op));

      for (uint16_t j = 0; j < target_sig->result_count; ++j) {
        loom_test_gen_values_add(&values, loom_op_results(call_op)[j],
                                 target_sig->result_types[j]);
      }
    }

    // Build test.yield with values matching the function's result types.
    loom_value_id_t return_vals[LOOM_TEST_GEN_MAX_FUNC_RESULTS];
    for (uint16_t j = 0; j < sig->result_count; ++j) {
      loom_scalar_type_t scalar = loom_type_element_type(sig->result_types[j]);
      return_vals[j] = loom_test_gen_values_pick_typed(gen, &values, scalar);
      if (return_vals[j] == LOOM_VALUE_ID_INVALID) {
        return_vals[j] = loom_test_gen_values_pick_any(gen, &values);
      }
      if (return_vals[j] == LOOM_VALUE_ID_INVALID) {
        loom_attribute_t const_val =
            loom_test_gen_constant_attr(sig->result_types[j], (int64_t)j);
        loom_op_t* const_op = NULL;
        IREE_RETURN_IF_ERROR(
            loom_test_constant_build(&builder, const_val, sig->result_types[j],
                                     LOOM_LOCATION_UNKNOWN, &const_op));
        return_vals[j] = loom_op_results(const_op)[0];
      }
    }
    loom_op_t* return_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_test_yield_build(&builder, return_vals, sig->result_count,
                              LOOM_LOCATION_UNKNOWN, &return_op));

    loom_builder_restore(&builder, saved);
  }

  *out_module = module;
  return iree_ok_status();
}

// Populates the config's inline hooks array with synthetic test-dialect hooks.
// Production dialect hook sets are explicit profiles layered by callers that
// register those dialects.
static void loom_test_gen_config_add_default_hooks(
    loom_test_gen_body_config_t* config) {
  iree_host_size_t test_count = 0;
  const loom_test_gen_op_hook_t* test_hooks =
      loom_test_gen_test_hooks(&test_count);

  IREE_ASSERT(test_count <= LOOM_TEST_GEN_MAX_BUILTIN_HOOKS);

  memcpy(config->hooks, test_hooks,
         test_count * sizeof(loom_test_gen_op_hook_t));
  config->hook_count = test_count;
}

//===----------------------------------------------------------------------===//
// Presets: body configs
//===----------------------------------------------------------------------===//

loom_test_gen_body_config_t loom_test_gen_body_config_representative(
    uint32_t scale) {
  loom_test_gen_body_config_t config = {0};
  config.op_count = (uint16_t)(20 * scale);
  config.max_nesting_depth = 3;
  config.nesting_probability = 20;
  config.dead_op_probability = 10;
  config.duplicate_probability = 10;
  config.write_probability = 10;
  config.cast_probability = 10;
  config.block_arg_count = (uint16_t)(2 + scale);
  config.value_fan_out = 4;
  loom_test_gen_type_palette_default(&config.palette);
  loom_test_gen_config_add_default_hooks(&config);
  return config;
}

loom_test_gen_body_config_t loom_test_gen_body_config_cse_stress(
    uint32_t scale) {
  loom_test_gen_body_config_t config = {0};
  config.op_count = (uint16_t)(30 * scale);
  config.max_nesting_depth = 2;
  config.nesting_probability = 15;
  config.dead_op_probability = 5;
  config.duplicate_probability = 50;
  config.write_probability = 20;
  config.cast_probability = 5;
  config.block_arg_count = (uint16_t)(2 + scale);
  config.value_fan_out = 2;
  loom_test_gen_type_palette_default(&config.palette);
  loom_test_gen_config_add_default_hooks(&config);
  return config;
}

loom_test_gen_body_config_t loom_test_gen_body_config_dce_stress(
    uint32_t scale) {
  loom_test_gen_body_config_t config = {0};
  config.op_count = (uint16_t)(30 * scale);
  config.max_nesting_depth = 2;
  config.nesting_probability = 10;
  config.dead_op_probability = 60;
  config.duplicate_probability = 5;
  config.write_probability = 5;
  config.cast_probability = 10;
  config.block_arg_count = (uint16_t)(2 + scale);
  config.value_fan_out = 1;
  loom_test_gen_type_palette_default(&config.palette);
  loom_test_gen_config_add_default_hooks(&config);
  return config;
}

loom_test_gen_body_config_t loom_test_gen_body_config_nesting_stress(
    uint32_t scale) {
  loom_test_gen_body_config_t config = {0};
  config.op_count = (uint16_t)(15 * scale);
  config.max_nesting_depth = 8;
  config.nesting_probability = 60;
  config.dead_op_probability = 10;
  config.duplicate_probability = 5;
  config.write_probability = 10;
  config.cast_probability = 5;
  config.block_arg_count = (uint16_t)(2 + scale);
  config.value_fan_out = 4;
  loom_test_gen_type_palette_default(&config.palette);
  loom_test_gen_config_add_default_hooks(&config);
  return config;
}

loom_test_gen_body_config_t loom_test_gen_body_config_format_stress(
    uint32_t scale) {
  loom_test_gen_body_config_t config = {0};
  config.op_count = (uint16_t)(40 * scale);
  config.max_nesting_depth = 4;
  config.nesting_probability = 30;
  config.dead_op_probability = 10;
  config.duplicate_probability = 10;
  config.write_probability = 10;
  config.cast_probability = 20;
  config.block_arg_count = (uint16_t)(4 + scale * 2);
  config.value_fan_out = 6;
  loom_test_gen_type_palette_default(&config.palette);
  loom_test_gen_config_add_default_hooks(&config);
  return config;
}

loom_test_gen_module_config_t loom_test_gen_module_config_representative(
    uint32_t scale) {
  loom_test_gen_module_config_t config = {0};
  config.function_count = (uint16_t)(2 + scale);
  config.declaration_count = (uint16_t)(1 + scale / 2);
  config.calls_per_function = (uint16_t)(1 + scale / 2);
  config.body_config = loom_test_gen_body_config_representative(scale);
  // Function-like ops create their own args; don't double-create via body gen.
  config.body_config.block_arg_count = 0;
  return config;
}

loom_test_gen_module_config_t loom_test_gen_module_config_fuzz_preset(
    uint8_t preset_index, uint32_t scale) {
  if (scale == 0) {
    scale = 1;
  }
  if (scale > 10) {
    scale = 10;
  }
  switch (preset_index % 5) {
    case 0:
      return loom_test_gen_module_config_representative(scale);
    case 1: {
      loom_test_gen_module_config_t config = {0};
      config.function_count = 2;
      config.body_config = loom_test_gen_body_config_cse_stress(scale);
      config.body_config.block_arg_count = 0;
      return config;
    }
    case 2: {
      loom_test_gen_module_config_t config = {0};
      config.function_count = 2;
      config.body_config = loom_test_gen_body_config_dce_stress(scale);
      config.body_config.block_arg_count = 0;
      return config;
    }
    case 3: {
      loom_test_gen_module_config_t config = {0};
      config.function_count = 1;
      config.body_config = loom_test_gen_body_config_nesting_stress(scale);
      config.body_config.block_arg_count = 0;
      return config;
    }
    default: {
      loom_test_gen_module_config_t config = {0};
      config.function_count = 2;
      config.body_config = loom_test_gen_body_config_format_stress(scale);
      config.body_config.block_arg_count = 0;
      return config;
    }
  }
}
