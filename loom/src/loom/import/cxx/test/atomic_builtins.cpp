// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Every update checks its returned value, independently of the final storage.
// The unsigned sequence crosses zero and alternates old/new fetch conventions.
unsigned builtin_fetch_values(volatile unsigned* storage) {
  auto* word = storage + 1;
  unsigned failures = 0;
  failures += __atomic_exchange_n(word, 0xfffffffcu, __ATOMIC_SEQ_CST) != 37u;
  failures += __atomic_fetch_add(word, 3u, __ATOMIC_RELAXED) != 0xfffffffcu;
  failures += __atomic_add_fetch(word, 1u, __ATOMIC_ACQUIRE) != 0u;
  failures += __atomic_fetch_sub(word, 1u, __ATOMIC_RELEASE) != 0u;
  failures += __atomic_sub_fetch(word, 2u, __ATOMIC_ACQ_REL) != 0xfffffffdu;
  failures += __atomic_fetch_and(word, 15u, __ATOMIC_RELAXED) != 0xfffffffdu;
  failures += __atomic_and_fetch(word, 7u, __ATOMIC_RELAXED) != 5u;
  failures += __atomic_fetch_or(word, 16u, __ATOMIC_RELAXED) != 5u;
  failures += __atomic_or_fetch(word, 32u, __ATOMIC_RELAXED) != 53u;
  failures += __atomic_fetch_xor(word, 3u, __ATOMIC_RELAXED) != 53u;
  failures += __atomic_xor_fetch(word, 7u, __ATOMIC_RELAXED) != 49u;
  return failures;
}

// Atomic signed arithmetic has modular semantics, including returned-new
// values.
unsigned builtin_signed_wrap(int* storage) {
  auto* word = storage + 1;
  unsigned failures = 0;
  __atomic_exchange_n(word, 2147483647, __ATOMIC_RELAXED);
  failures +=
      __atomic_add_fetch(word, 1, __ATOMIC_RELAXED) != (-2147483647 - 1);
  failures += __atomic_sub_fetch(word, 1, __ATOMIC_RELAXED) != 2147483647;
  return failures;
}

template <class T>
static T once(T value, unsigned* evaluations) {
  ++*evaluations;
  return value;
}

// Separate counters witness once-only evaluation of every non-order argument.
// CAS reads expected after argument evaluation and only replaces it on failure.
unsigned builtin_compare_values(unsigned* storage, unsigned* evaluations) {
  auto* word = storage + 1;
  auto* expected = storage + 2;
  unsigned failures = 0;
  failures += !__atomic_compare_exchange_n(
      once(word, evaluations + 1), once(expected, evaluations + 2),
      once(7u, evaluations + 3), once(true, evaluations + 4), __ATOMIC_ACQ_REL,
      __ATOMIC_ACQUIRE);
  failures += *expected != 37u;
  failures += __atomic_compare_exchange_n(
      once(word, evaluations + 1), once(expected, evaluations + 2),
      once(11u, evaluations + 3), once(false, evaluations + 4),
      __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  failures += *expected != 7u;
  // Aliasing makes an erroneous store on success observable: it would undo 13.
  failures += !__atomic_compare_exchange_n(word, word, 13u, false,
                                           __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return failures;
}

// High bits, carry, wrap, and both CAS outcomes exercise the complete i64
// value.
unsigned builtin_wide_values(unsigned long long* storage) {
  auto* word = storage + 1;
  auto* expected = storage + 2;
  unsigned failures = 0;
  failures += __atomic_exchange_n(word, 0xffffffffffffffffULL,
                                  __ATOMIC_RELAXED) != 37ULL;
  failures += __atomic_add_fetch(word, 1ULL, __ATOMIC_RELAXED) != 0ULL;
  failures += __atomic_fetch_add(word, 37ULL, __ATOMIC_RELAXED) != 0ULL;
  failures +=
      !__atomic_compare_exchange_n(word, expected, 0x100000003ULL, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  failures += *expected != 37ULL;
  failures += __atomic_compare_exchange_n(word, expected, 19ULL, true,
                                          __ATOMIC_ACQUIRE, __ATOMIC_CONSUME);
  failures += *expected != 0x100000003ULL;
  failures += !__atomic_compare_exchange_n(word, word, 0x100000004ULL, false,
                                           __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return failures;
}
