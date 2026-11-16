// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_OBJECT_H_
#define LOOMCXX_OBJECT_H_

namespace loom {

// A borrowed handle to an incomplete payload declared with loom::import.
// Copying the handle preserves the borrow; it cannot extend payload lifetime.
// The handle exposes neither the payload's layout nor a native address.
template <class T>
class [[loom::type("ptr")]] ptr {
  // Source representation retaining ordinary trivial-copy semantics.
  T* data_;
};

}  // namespace loom

#endif  // LOOMCXX_OBJECT_H_
