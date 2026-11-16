// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IR_ARGUMENT_ACCESS_H_
#define LOOM_IR_ARGUMENT_ACCESS_H_

// Shallow access bounds on a buffer/view parameter and its derived aliases.
// Explicit bounds restrict storage access to the duration of the call: the
// parameter cannot escape or be used asynchronously. They do not imply noalias,
// full initialization, extent confinement, or constness of referenced objects.
typedef enum loom_argument_access_e {
  LOOM_ARGUMENT_ACCESS_UNSPECIFIED = 0,
  LOOM_ARGUMENT_ACCESS_READ = 1,
  LOOM_ARGUMENT_ACCESS_WRITE = 2,
  LOOM_ARGUMENT_ACCESS_READ_WRITE = 3,
  LOOM_ARGUMENT_ACCESS_NONE = 4,
} loom_argument_access_t;

#endif  // LOOM_IR_ARGUMENT_ACCESS_H_
