// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Typed adaptive in-place sorting for compiler hot paths.
//
// Many Loom compiler arrays are already close to program order. A direct
// insertion sort keeps that common path tiny and cache-local, while an in-place
// heap sort bounds heavily disordered generated-kernel inputs without libc
// qsort callback dispatch or temporary arena scratch.

#ifndef LOOM_UTIL_ADAPTIVE_SORT_H_
#define LOOM_UTIL_ADAPTIVE_SORT_H_

#include "iree/base/api.h"

#define LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD 64u
#define LOOM_ADAPTIVE_SORT_NEARLY_SORTED_INVERSION_DIVISOR 16u

// Defines |function_name| plus static local helpers for sorting |element_type|
// arrays. |less_fn| must have this shape:
//
//   bool less_fn(const element_type* lhs, const element_type* rhs)
#define LOOM_DEFINE_ADAPTIVE_SORT(function_name, element_type, less_fn)      \
  static void function_name##_swap(element_type* lhs, element_type* rhs) {   \
    element_type temporary = *lhs;                                           \
    *lhs = *rhs;                                                             \
    *rhs = temporary;                                                        \
  }                                                                          \
                                                                             \
  static void function_name##_insertion_sort(element_type* values,           \
                                             iree_host_size_t count) {       \
    for (iree_host_size_t i = 1; i < count; ++i) {                           \
      element_type value = values[i];                                        \
      iree_host_size_t j = i;                                                \
      while (j > 0 && less_fn(&value, &values[j - 1])) {                     \
        values[j] = values[j - 1];                                           \
        --j;                                                                 \
      }                                                                      \
      values[j] = value;                                                     \
    }                                                                        \
  }                                                                          \
                                                                             \
  static void function_name##_heap_sift_down(                                \
      element_type* values, iree_host_size_t root, iree_host_size_t count) { \
    while (true) {                                                           \
      const iree_host_size_t left_child = root * 2u + 1u;                    \
      if (left_child >= count) return;                                       \
                                                                             \
      iree_host_size_t child = left_child;                                   \
      const iree_host_size_t right_child = left_child + 1u;                  \
      if (right_child < count &&                                             \
          less_fn(&values[child], &values[right_child])) {                   \
        child = right_child;                                                 \
      }                                                                      \
      if (!less_fn(&values[root], &values[child])) {                         \
        return;                                                              \
      }                                                                      \
      function_name##_swap(&values[root], &values[child]);                   \
      root = child;                                                          \
    }                                                                        \
  }                                                                          \
                                                                             \
  static void function_name##_heap_sort(element_type* values,                \
                                        iree_host_size_t count) {            \
    iree_host_size_t root = count / 2u;                                      \
    while (root > 0) {                                                       \
      --root;                                                                \
      function_name##_heap_sift_down(values, root, count);                   \
    }                                                                        \
                                                                             \
    iree_host_size_t end = count;                                            \
    while (end > 1) {                                                        \
      --end;                                                                 \
      function_name##_swap(&values[0], &values[end]);                        \
      function_name##_heap_sift_down(values, 0, end);                        \
    }                                                                        \
  }                                                                          \
                                                                             \
  static void function_name(element_type* values, iree_host_size_t count) {  \
    if (count < 2) return;                                                   \
                                                                             \
    iree_host_size_t adjacent_inversion_count = 0;                           \
    for (iree_host_size_t i = 1; i < count; ++i) {                           \
      if (less_fn(&values[i], &values[i - 1])) {                             \
        ++adjacent_inversion_count;                                          \
      }                                                                      \
    }                                                                        \
    if (adjacent_inversion_count == 0) return;                               \
                                                                             \
    if (count <= LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD ||             \
        adjacent_inversion_count <=                                          \
            count / LOOM_ADAPTIVE_SORT_NEARLY_SORTED_INVERSION_DIVISOR) {    \
      function_name##_insertion_sort(values, count);                         \
      return;                                                                \
    }                                                                        \
    function_name##_heap_sort(values, count);                                \
  }

#endif  // LOOM_UTIL_ADAPTIVE_SORT_H_
