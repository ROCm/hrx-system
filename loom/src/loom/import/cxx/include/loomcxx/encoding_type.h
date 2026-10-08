// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_ENCODING_TYPE_H_
#define LOOMCXX_ENCODING_TYPE_H_

namespace loom::encoding {

// Semantic role carried by an encoding value. Names match the encoding dialect.
enum class role { layout, schema, storage, transform };

}  // namespace loom::encoding

namespace loom::type {

using size_type = __SIZE_TYPE__;

// A value-owned encoding. Layouts and physical storage describe Rank logical
// axes; schemas and numeric transforms have rank zero. Private storage
// preserves ordinary C++ copy, lifetime, and sizeof semantics. Import projects
// the object to one first-class Loom encoding value, retaining its semantic
// role.
template <loom::encoding::role Role,
          size_type Rank = (Role == loom::encoding::role::layout ||
                            Role == loom::encoding::role::storage)
                               ? 2
                               : 0>
class [[loom::type("encoding")]] encoding {
  // Source storage for layout parameters or a schema identity.
  size_type parameters_[Rank ? Rank : 1];
};

}  // namespace loom::type

#endif  // LOOMCXX_ENCODING_TYPE_H_
