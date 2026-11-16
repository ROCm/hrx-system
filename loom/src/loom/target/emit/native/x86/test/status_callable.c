// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"

typedef struct result_t {
  // First narrow result; the next field requires natural C padding.
  uint8_t byte;
  // Computed value transported through nested Loom and native calls.
  uint64_t sum;
  // Second narrow result preceding an independently aligned floating field.
  uint16_t short_value;
  // Floating result written through the same caller-owned record.
  double double_value;
  // Logical i1 occupies one C byte.
  bool predicate;
  // Final floating field verifies record tail alignment.
  float float_value;
} result_t;

extern iree_status_t status_forward(result_t* result, uint64_t seed);
extern iree_status_t status_stack(uint64_t* result, uint64_t a, uint64_t b,
                                  uint64_t c, uint64_t d, uint64_t e,
                                  uint64_t f, uint64_t g, uint64_t h);
extern iree_status_t status_identity(uint64_t* result, uint64_t value);
extern iree_status_t status_void(uint64_t value);

typedef float vector8_t __attribute__((ext_vector_type(8)));
typedef struct vector_result_t {
  // Narrow leading field makes vector padding observable.
  uint8_t first;
  // Native vector field with 32-byte alignment.
  vector8_t sum;
  // Trailing field verifies the complete record layout.
  uint64_t last;
} vector_result_t;
extern iree_status_t status_vector(vector_result_t* result, const float* input);

// One host-selected failing call; zero selects success.
static unsigned failure_call;
// Calls reached before propagation returns to the host.
static unsigned calls;
// One owned status payload, transferred unchanged through generated callers.
static iree_status_t failure;
// Normal C calls remain normal C calls within a status-returning function.
static unsigned ordinary_calls;

iree_status_t native_read(uint64_t* result, uint32_t position) {
  ++calls;
  if (calls == failure_call) {
    return failure;
  }
  *result = (uint64_t)position * 13 + 5;
  return iree_ok_status();
}

iree_status_t native_accept(uint64_t value) {
  (void)value;
  ++calls;
  return calls == failure_call ? failure : iree_ok_status();
}

uint64_t native_mix(uint64_t value) {
  ++ordinary_calls;
  return value * 7;
}

iree_status_t native_vector(vector8_t* result, vector8_t a, vector8_t b,
                            vector8_t c, vector8_t d, vector8_t e, vector8_t f,
                            vector8_t g, vector8_t h, vector8_t i) {
  ++calls;
  if (calls == failure_call) {
    return failure;
  }
  *result = a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h + 9 * i;
  return iree_ok_status();
}

iree_status_t native_sum(uint64_t* result, uint64_t a, uint64_t b, uint64_t c,
                         uint64_t d, uint64_t e, uint64_t f, uint64_t g,
                         uint64_t h) {
  ++calls;
  if (calls == failure_call) {
    return failure;
  }
  *result = a + 2 * b + 3 * c + 4 * d + 5 * e + 6 * f + 7 * g + 8 * h;
  return iree_ok_status();
}

#define CHECK(condition)                                                     \
  do {                                                                       \
    if (!(condition)) {                                                      \
      fprintf(stderr, "line %d: %s (failure_call=%u, calls=%u)\n", __LINE__, \
              #condition, failure_call, calls);                              \
      passed = false;                                                        \
    }                                                                        \
  } while (0)

int main(void) {
  bool passed = true;
  failure = iree_make_status(IREE_STATUS_DATA_LOSS, "native failure payload");
  for (unsigned fail_at = 0; fail_at <= 4; ++fail_at) {
    calls = ordinary_calls = 0;
    failure_call = fail_at;
    result_t result;
    memset(&result, 0xCD, sizeof(result));
    uint8_t sentinel[sizeof(result)];
    memset(sentinel, 0xCD, sizeof(sentinel));
    iree_status_t status = status_forward(&result, 11);
    CHECK(status == (fail_at ? failure : iree_ok_status()));
    CHECK(calls == (fail_at ? fail_at : 4));
    CHECK(ordinary_calls == (fail_at != 0 && fail_at < 4 ? 0u : 1u));
    if (fail_at) {
      CHECK(memcmp(&result, sentinel, sizeof(result)) == 0);
    } else {
      CHECK(result.byte == 167 && result.sum == 131 &&
            result.short_value == 43981 && result.double_value == 3.5 &&
            result.predicate && result.float_value == -1.25f);
    }
  }
  for (unsigned fail_at = 0; fail_at <= 1; ++fail_at) {
    calls = 0;
    failure_call = fail_at;
    uint64_t result = UINT64_MAX;
    iree_status_t status = status_stack(&result, 1, 2, 3, 4, 5, 6, 7, 8);
    CHECK(status == (fail_at ? failure : iree_ok_status()));
    CHECK(calls == 1);
    CHECK(result == (fail_at ? UINT64_MAX : 204));
  }
  for (unsigned fail_at = 0; fail_at <= 2; ++fail_at) {
    calls = 0;
    failure_call = fail_at;
    iree_status_t status = status_void(19);
    CHECK(status == (fail_at ? failure : iree_ok_status()));
    CHECK(calls == (fail_at ? fail_at : 2));
  }
  uint64_t identity = 0;
  iree_status_t status =
      status_identity(&identity, UINT64_C(0xFEDCBA9876543210));
  CHECK(iree_status_is_ok(status) && identity == UINT64_C(0xFEDCBA9876543210));
  float input[72];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(input); ++i) {
    input[i] = i;
  }
  for (unsigned fail_at = 0; fail_at <= 1; ++fail_at) {
    calls = 0;
    failure_call = fail_at;
    vector_result_t result;
    memset(&result, 0xCD, sizeof(result));
    uint8_t sentinel[sizeof(result)];
    memset(sentinel, 0xCD, sizeof(sentinel));
    status = status_vector(&result, input);
    CHECK(status == (fail_at ? failure : iree_ok_status()));
    CHECK(calls == 1);
    if (fail_at) {
      CHECK(memcmp(&result, sentinel, sizeof(result)) == 0);
    } else {
      CHECK(result.first == 19 && result.last == 771);
      for (unsigned lane = 0; lane < 8; ++lane) {
        float expected = 0;
        for (unsigned row = 0; row < 9; ++row) {
          expected += (row + 1) * input[8 * row + lane];
        }
        CHECK(result.sum[lane] == expected);
      }
    }
  }
  iree_status_ignore(failure);
  if (!passed) {
    fprintf(stderr, "native status ABI contract failed\n");
    return 1;
  }
  return 0;
}
