// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/call_aliases.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace loom::cxx_import {
namespace {

template <typename T>
iree_status_t allocate_array(iree_arena_allocator_t* arena, size_t count,
                             T** out_values) {
  *out_values = nullptr;
  if (!count) {
    return iree_ok_status();
  }
  return iree_arena_allocate_array_aligned(
      arena, count, sizeof(T), alignof(T),
      reinterpret_cast<void**>(out_values));
}

struct IntervalOrder {
  // Shared immutable interval table indexed by each compared ordinal.
  std::span<const CallAliasInterval> intervals;
  // Position field selecting begin order or end order.
  size_t CallAliasInterval::* position;

  bool operator()(size_t left, size_t right) const {
    const auto& lhs = intervals[left];
    const auto& rhs = intervals[right];
    if (lhs.binding != rhs.binding) {
      return std::less<const void*>{}(lhs.binding, rhs.binding);
    }
    return lhs.*position < rhs.*position;
  }
};

}  // namespace

iree_status_t CallAliases::select_storage(
    iree_arena_allocator_t* scratch_arena,
    std::span<const uint8_t>* out_selected) const {
  *out_selected = {};
  if (parameters_.size() < 2) {
    return iree_ok_status();
  }
  const size_t written_parameter_count = std::ranges::count_if(
      parameters_, [](const auto& parameter) { return parameter.written; });
  if (!written_parameter_count) {
    return iree_ok_status();
  }

  uint8_t* selected = nullptr;
  IREE_RETURN_IF_ERROR(
      allocate_array(scratch_arena, parameters_.size(), &selected));
  std::fill_n(selected, parameters_.size(), uint8_t{0});
  *out_selected = {selected, parameters_.size()};

  size_t written_interval_count = 0;
  for (const auto& interval : intervals_) {
    written_interval_count += parameters_[interval.parameter].written;
  }
  if (written_interval_count) {
    size_t* all_by_begin = nullptr;
    size_t* all_by_end = nullptr;
    size_t* written_by_begin = nullptr;
    size_t* written_by_end = nullptr;
    IREE_RETURN_IF_ERROR(
        allocate_array(scratch_arena, intervals_.size(), &all_by_begin));
    IREE_RETURN_IF_ERROR(
        allocate_array(scratch_arena, intervals_.size(), &all_by_end));
    IREE_RETURN_IF_ERROR(allocate_array(scratch_arena, written_interval_count,
                                        &written_by_begin));
    IREE_RETURN_IF_ERROR(
        allocate_array(scratch_arena, written_interval_count, &written_by_end));
    std::iota(all_by_begin, all_by_begin + intervals_.size(), 0);
    std::iota(all_by_end, all_by_end + intervals_.size(), 0);
    size_t written_index = 0;
    for (size_t index = 0; index < intervals_.size(); ++index) {
      if (parameters_[intervals_[index].parameter].written) {
        written_by_begin[written_index] = index;
        written_by_end[written_index++] = index;
      }
    }

    // Marks query parameters whose intervals overlap a candidate interval
    // owned by a different parameter. Each source binding is swept once in
    // increasing query-end order. The two furthest-reaching distinct owners
    // make exclusion of the query's own intervals constant time.
    auto mark_overlaps = [&](std::span<size_t> queries,
                             std::span<size_t> candidates) {
      auto binding_less = std::less<const void*>{};
      auto same_binding = [&](size_t left, size_t right) {
        return intervals_[left].binding == intervals_[right].binding;
      };
      std::ranges::sort(queries,
                        IntervalOrder{intervals_, &CallAliasInterval::end});
      std::ranges::sort(candidates,
                        IntervalOrder{intervals_, &CallAliasInterval::begin});
      struct Reach {
        // Parameter owning this maximum end, or SIZE_MAX when absent.
        size_t parameter = SIZE_MAX;
        // Furthest exclusive interval end reached by the parameter.
        size_t end = 0;
      };
      size_t query_index = 0;
      size_t candidate_index = 0;
      while (query_index < queries.size() &&
             candidate_index < candidates.size()) {
        const auto* query_binding = intervals_[queries[query_index]].binding;
        const auto* candidate_binding =
            intervals_[candidates[candidate_index]].binding;
        if (query_binding != candidate_binding) {
          if (binding_less(query_binding, candidate_binding)) {
            do {
              ++query_index;
            } while (query_index < queries.size() &&
                     intervals_[queries[query_index]].binding == query_binding);
          } else {
            do {
              ++candidate_index;
            } while (candidate_index < candidates.size() &&
                     intervals_[candidates[candidate_index]].binding ==
                         candidate_binding);
          }
          continue;
        }
        size_t query_end = query_index + 1;
        while (query_end < queries.size() &&
               same_binding(queries[query_index], queries[query_end])) {
          ++query_end;
        }
        size_t candidate_end = candidate_index + 1;
        while (candidate_end < candidates.size() &&
               same_binding(candidates[candidate_index],
                            candidates[candidate_end])) {
          ++candidate_end;
        }
        size_t active_end = candidate_index;
        Reach first;
        Reach second;
        auto append_reach = [&](const CallAliasInterval& interval) {
          if (first.parameter == interval.parameter) {
            first.end = std::max(first.end, interval.end);
          } else if (second.parameter == interval.parameter) {
            second.end = std::max(second.end, interval.end);
          } else if (first.parameter == SIZE_MAX || interval.end > first.end) {
            second = first;
            first = {interval.parameter, interval.end};
          } else if (second.parameter == SIZE_MAX ||
                     interval.end > second.end) {
            second = {interval.parameter, interval.end};
          }
          if (second.parameter != SIZE_MAX && second.end > first.end) {
            std::swap(first, second);
          }
        };
        for (; query_index < query_end; ++query_index) {
          const auto& query = intervals_[queries[query_index]];
          while (active_end < candidate_end &&
                 intervals_[candidates[active_end]].begin < query.end) {
            append_reach(intervals_[candidates[active_end++]]);
          }
          const auto& reach =
              first.parameter != query.parameter ? first : second;
          if (reach.parameter != SIZE_MAX && reach.end > query.begin) {
            selected[query.parameter] = true;
          }
        }
        candidate_index = candidate_end;
      }
    };

    mark_overlaps({all_by_end, intervals_.size()},
                  {written_by_begin, written_interval_count});
    mark_overlaps({written_by_end, written_interval_count},
                  {all_by_begin, intervals_.size()});
  }

  size_t identity_count = 0;
  size_t unresolved_count = 0;
  size_t written_identity_count = 0;
  size_t written_unresolved_count = 0;
  for (const auto& parameter : parameters_) {
    identity_count += parameter.has_identity;
    unresolved_count += parameter.unresolved_identity;
    written_identity_count += parameter.written && parameter.has_identity;
    written_unresolved_count +=
        parameter.written && parameter.unresolved_identity;
  }
  auto has_other = [](size_t count, bool current) {
    return count > static_cast<size_t>(current);
  };
  for (size_t index = 0; index < parameters_.size(); ++index) {
    const auto& parameter = parameters_[index];
    if (parameter.unresolved_identity &&
        ((parameter.written &&
          has_other(identity_count, parameter.has_identity)) ||
         has_other(written_identity_count,
                   parameter.written && parameter.has_identity))) {
      selected[index] = true;
    }
    if (parameter.has_identity &&
        ((parameter.written &&
          has_other(unresolved_count, parameter.unresolved_identity)) ||
         has_other(written_unresolved_count,
                   parameter.written && parameter.unresolved_identity))) {
      selected[index] = true;
    }
  }
  return iree_ok_status();
}

}  // namespace loom::cxx_import
