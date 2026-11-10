// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/control/components.h"

#include <algorithm>
#include <utility>

namespace loom::cxx_import {

bool Components::contains(ComponentRange range) const {
  if (range.begin == range.end) {
    return true;
  }
  auto found = std::lower_bound(ranges_.begin(), ranges_.end(), range.begin,
                                [](ComponentRange candidate, size_t begin) {
                                  return candidate.end <= begin;
                                });
  return found != ranges_.end() && found->begin <= range.begin &&
         found->end >= range.end;
}

bool Components::intersects(ComponentRange range) const {
  auto found = std::lower_bound(ranges_.begin(), ranges_.end(), range.begin,
                                [](ComponentRange candidate, size_t begin) {
                                  return candidate.end <= begin;
                                });
  return found != ranges_.end() && found->begin < range.end &&
         range.begin < range.end;
}

bool Components::add(ComponentRange range) {
  if (range.begin == range.end) {
    return false;
  }
  auto first = std::lower_bound(ranges_.begin(), ranges_.end(), range.begin,
                                [](ComponentRange candidate, size_t begin) {
                                  return candidate.end < begin;
                                });
  if (first != ranges_.end() && first->begin <= range.begin &&
      first->end >= range.end) {
    return false;
  }
  auto last = first;
  while (last != ranges_.end() && last->begin <= range.end) {
    range.begin = std::min(range.begin, last->begin);
    range.end = std::max(range.end, last->end);
    ++last;
  }
  if (first == last) {
    ranges_.insert(first, range);
  } else {
    *first = range;
    ranges_.erase(first + 1, last);
  }
  return true;
}

bool Components::add(const Components& other) {
  bool changed = false;
  for (auto range : other.ranges_) {
    changed = add(range) || changed;
  }
  return changed;
}

void Components::intersect(const Components& other) {
  std::vector<ComponentRange> result;
  auto right = other.ranges_.begin();
  for (auto left : ranges_) {
    while (right != other.ranges_.end() && right->end <= left.begin) {
      ++right;
    }
    for (auto current = right;
         current != other.ranges_.end() && current->begin < left.end;
         ++current) {
      result.push_back({std::max(left.begin, current->begin),
                        std::min(left.end, current->end)});
    }
  }
  ranges_ = std::move(result);
}

void Components::subtract(const Components& other) {
  std::vector<ComponentRange> result;
  auto right = other.ranges_.begin();
  for (auto left : ranges_) {
    while (right != other.ranges_.end() && right->end <= left.begin) {
      ++right;
    }
    for (auto current = right;
         current != other.ranges_.end() && current->begin < left.end;
         ++current) {
      if (left.begin < current->begin) {
        result.push_back({left.begin, current->begin});
      }
      left.begin = std::min(left.end, current->end);
    }
    if (left.begin < left.end) {
      result.push_back(left);
    }
  }
  ranges_ = std::move(result);
}

Components Components::project(ComponentRange source,
                               size_t destination) const {
  Components result;
  if (source.begin == source.end) {
    return result;
  }
  auto first = std::lower_bound(ranges_.begin(), ranges_.end(), source.begin,
                                [](ComponentRange candidate, size_t begin) {
                                  return candidate.end <= begin;
                                });
  for (; first != ranges_.end() && first->begin < source.end; ++first) {
    result.ranges_.push_back(
        {std::max(first->begin, source.begin) - source.begin + destination,
         std::min(first->end, source.end) - source.begin + destination});
  }
  return result;
}

}  // namespace loom::cxx_import
