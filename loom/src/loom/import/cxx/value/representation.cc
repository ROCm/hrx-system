// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/value/representation.h"

#include <algorithm>

namespace loom::cxx_import {

Value::Value(const Partition& partition,
             std::span<const loom_value_id_t> components)
    : partition_(&partition) {
  IREE_ASSERT_EQ(components.size(), partition.component_count);
  if (partition.component_count <= 2) {
    std::copy(components.begin(), components.end(), storage_.inline_values);
  } else {
    storage_.indirect_values = components.data();
  }
}

Value ValueArena::capture(const Partition& partition,
                          std::span<const loom_value_id_t> components) {
  if (partition.component_count <= 2) {
    return Value(partition, components);
  }
  auto* owned = static_cast<loom_value_id_t*>(
      storage_.allocate(components.size_bytes(), alignof(loom_value_id_t)));
  std::copy(components.begin(), components.end(), owned);
  return Value(partition, {owned, components.size()});
}

Value ValueArena::replace(Value original, size_t component_offset,
                          Value replacement) {
  if (!replacement.partition().component_count) {
    return original;
  }
  auto source = original.components();
  loom_value_id_t local[2];
  auto* owned = source.size() <= 2
                    ? local
                    : static_cast<loom_value_id_t*>(storage_.allocate(
                          source.size_bytes(), alignof(loom_value_id_t)));
  std::copy(source.begin(), source.end(), owned);
  auto member = replacement.components();
  std::copy(member.begin(), member.end(), owned + component_offset);
  return Value(original.partition(), {owned, source.size()});
}

}  // namespace loom::cxx_import
