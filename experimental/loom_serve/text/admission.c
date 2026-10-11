// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/text/admission.h"

bool loom_serve_text_admission_blocked(loom_serve_text_admission_t* admission,
                                       iree_host_size_t count) {
  if (admission->overtakes < admission->budget &&
      admission->cursor + 1 < count) {
    ++admission->cursor;
    return true;
  }
  admission->cursor = 0;
  return false;
}

void loom_serve_text_admission_acquired(
    loom_serve_text_admission_t* admission) {
  admission->overtakes += admission->cursor != 0;
}

void loom_serve_text_admission_remove(loom_serve_text_admission_t* admission,
                                      iree_host_size_t index) {
  if (!index) {
    admission->overtakes = 0;
  }
  if (index <= admission->cursor) {
    admission->cursor = 0;
  }
}
