// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_CONTROL_COMPONENTS_H_
#define LOOM_IMPORT_CXX_CONTROL_COMPONENTS_H_

#include <cstddef>
#include <span>
#include <vector>

namespace loom::cxx_import {

// A half-open interval in a source object's flattened value partition.
struct ComponentRange {
  // Inclusive first component.
  size_t begin;
  // Exclusive final component.
  size_t end;

  bool operator==(const ComponentRange&) const = default;
};

// Sparse source-component facts. Sorted disjoint intervals coalesce adjacent
// fields, so an untouched wide object does not require a component-sized mask.
class Components {
 public:
  bool empty() const { return ranges_.empty(); }
  std::span<const ComponentRange> ranges() const { return ranges_; }
  bool contains(ComponentRange range) const;
  bool intersects(ComponentRange range) const;
  // Returns whether the set grew.
  bool add(ComponentRange range);
  bool add(const Components& other);
  void intersect(const Components& other);
  void subtract(const Components& other);
  // Projects a source interval into another object's component coordinates.
  Components project(ComponentRange source, size_t destination) const;

  bool operator==(const Components&) const = default;

 private:
  // Canonical nonempty intervals in increasing component order.
  std::vector<ComponentRange> ranges_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_CONTROL_COMPONENTS_H_
