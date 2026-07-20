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
// qsort callback dispatch or temporary arena scratch. Large arrays attempt at
// most one insertion movement per element before switching to heap sort, so
// adversarial order adds only linear work to the O(n log n) fallback.

#ifndef LOOM_UTIL_ADAPTIVE_SORT_H_
#define LOOM_UTIL_ADAPTIVE_SORT_H_

#include "iree/base/api.h"

#define LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD 64u

// Defines |function_name| plus static local helpers for sorting |element_type|
// arrays. |less_fn| must have this shape:
//
//   bool less_fn(const element_type* lhs, const element_type* rhs)
#define LOOM_DEFINE_ADAPTIVE_SORT(function_name, element_type, less_fn)        \
  static void function_name##_swap(element_type* lhs, element_type* rhs) {     \
    element_type temporary = *lhs;                                             \
    *lhs = *rhs;                                                               \
    *rhs = temporary;                                                          \
  }                                                                            \
                                                                               \
  static bool function_name##_try_insertion_sort(                              \
      element_type* values, iree_host_size_t count,                            \
      iree_host_size_t start_index, iree_host_size_t move_budget) {            \
    for (iree_host_size_t i = start_index; i < count; ++i) {                   \
      if (i != start_index && !less_fn(&values[i], &values[i - 1])) continue;  \
      element_type value = values[i];                                          \
      iree_host_size_t j = i;                                                  \
      while (true) {                                                           \
        if (move_budget == 0) {                                                \
          values[j] = value;                                                   \
          return false;                                                        \
        }                                                                      \
        --move_budget;                                                         \
        values[j] = values[j - 1];                                             \
        --j;                                                                   \
        if (j == 0 || !less_fn(&value, &values[j - 1])) break;                 \
      }                                                                        \
      values[j] = value;                                                       \
    }                                                                          \
    return true;                                                               \
  }                                                                            \
                                                                               \
  static void function_name##_heap_sift_down(                                  \
      element_type* values, iree_host_size_t root, iree_host_size_t count) {   \
    while (true) {                                                             \
      if (root >= count / 2u) return;                                          \
      const iree_host_size_t left_child = root * 2u + 1u;                      \
      iree_host_size_t child = left_child;                                     \
      const iree_host_size_t right_child = left_child + 1u;                    \
      if (right_child < count &&                                               \
          less_fn(&values[child], &values[right_child])) {                     \
        child = right_child;                                                   \
      }                                                                        \
      if (!less_fn(&values[root], &values[child])) {                           \
        return;                                                                \
      }                                                                        \
      function_name##_swap(&values[root], &values[child]);                     \
      root = child;                                                            \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void function_name##_heap_sort(element_type* values,                  \
                                        iree_host_size_t count) {              \
    iree_host_size_t root = count / 2u;                                        \
    while (root > 0) {                                                         \
      --root;                                                                  \
      function_name##_heap_sift_down(values, root, count);                     \
    }                                                                          \
                                                                               \
    iree_host_size_t end = count;                                              \
    while (end > 1) {                                                          \
      --end;                                                                   \
      function_name##_swap(&values[0], &values[end]);                          \
      function_name##_heap_sift_down(values, 0, end);                          \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void function_name(element_type* values, iree_host_size_t count) {    \
    if (count < 2) return;                                                     \
                                                                               \
    iree_host_size_t first_inversion = 1;                                      \
    while (first_inversion < count &&                                          \
           !less_fn(&values[first_inversion], &values[first_inversion - 1])) { \
      ++first_inversion;                                                       \
    }                                                                          \
    if (first_inversion == count) return;                                      \
                                                                               \
    const iree_host_size_t move_budget =                                       \
        count <= LOOM_ADAPTIVE_SORT_INSERTION_COUNT_THRESHOLD                  \
            ? IREE_HOST_SIZE_MAX                                               \
            : count;                                                           \
    if (function_name##_try_insertion_sort(values, count, first_inversion,     \
                                           move_budget)) {                     \
      return;                                                                  \
    }                                                                          \
    function_name##_heap_sort(values, count);                                  \
  }

#endif  // LOOM_UTIL_ADAPTIVE_SORT_H_
