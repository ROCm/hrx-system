// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C++ RAII helpers for inspecting immutable byte sequences in tests.

#ifndef LOOM_TESTING_BYTE_SEQUENCE_H_
#define LOOM_TESTING_BYTE_SEQUENCE_H_

#include "iree/base/byte_sequence.h"

namespace loom::testing {

// Owns one contiguous test-only clone of an immutable byte sequence.
class ByteSequenceClone {
 public:
  explicit ByteSequenceClone(iree_allocator_t allocator)
      : allocator_(allocator) {}

  ~ByteSequenceClone() { iree_allocator_free(allocator_, contents_.data); }

  ByteSequenceClone(const ByteSequenceClone&) = delete;
  ByteSequenceClone& operator=(const ByteSequenceClone&) = delete;

  // Replaces the current contents with a clone of |sequence|.
  iree_status_t Clone(const iree_byte_sequence_t* sequence) {
    iree_allocator_free(allocator_, contents_.data);
    contents_ = iree_byte_span_empty();
    return iree_byte_sequence_clone(sequence, allocator_, &contents_);
  }

  // Returns a borrowed immutable view over the clone.
  iree_const_byte_span_t contents() const {
    return iree_const_cast_byte_span(contents_);
  }

 private:
  // Allocator owning |contents_|.
  iree_allocator_t allocator_;
  // Independently owned contiguous sequence contents.
  iree_byte_span_t contents_ = iree_byte_span_empty();
};

}  // namespace loom::testing

#endif  // LOOM_TESTING_BYTE_SEQUENCE_H_
