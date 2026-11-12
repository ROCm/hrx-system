// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/api.h"

#if (IREE_STATUS_FEATURES & IREE_STATUS_FEATURE_STACK_TRACE) != 0 && \
    defined(IREE_HAVE_LIBBACKTRACE)

// A shallow call chain with deeply nested debug scopes. Traversing this
// compilation unit must not require one native stack frame per namespace.
namespace n00::n01::n02::n03::n04::n05::n06::n07::n08::n09::n10::n11::n12::n13::
    n14::n15::n16::n17::n18::n19::n20::n21::n22::n23::n24::n25::n26::n27::n28::
        n29::n30::n31 {

IREE_ATTRIBUTE_ALWAYS_INLINE iree_status_t InlineInner() {
  return iree_make_status(IREE_STATUS_DATA_LOSS, "nested scope diagnostic");
}

IREE_ATTRIBUTE_ALWAYS_INLINE iree_status_t InlineOuter() {
  return iree_status_annotate_f(InlineInner(), "preserved annotation");
}

IREE_ATTRIBUTE_NOINLINE iree_status_t CaptureDeepScope() {
  return InlineOuter();
}

IREE_ATTRIBUTE_NOINLINE iree_status_t CaptureSibling() { return InlineOuter(); }

}  // namespace
   // n00::n01::n02::n03::n04::n05::n06::n07::n08::n09::n10::n11::n12::n13::n14::n15::n16::n17::n18::n19::n20::n21::n22::n23::n24::n25::n26::n27::n28::n29::n30::n31

extern "C" IREE_ATTRIBUTE_NOINLINE iree_status_t
iree_status_stack_trace_test_capture_scoped(bool sibling) {
  namespace scopes = n00::n01::n02::n03::n04::n05::n06::n07::n08::n09::n10::
      n11::n12::n13::n14::n15::n16::n17::n18::n19::n20::n21::n22::n23::n24::
          n25::n26::n27::n28::n29::n30::n31;
  return sibling ? scopes::CaptureSibling() : scopes::CaptureDeepScope();
}

#endif  // stack traces with libbacktrace
