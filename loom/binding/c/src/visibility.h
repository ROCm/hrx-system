// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_VISIBILITY_H_
#define LOOMC_VISIBILITY_H_

#include "loomc/base.h"

// Internal interfaces used by separately linked LoomC target bindings. These
// symbols are exported but require bindings built from the same source
// revision; they are not part of the supported public Loom C API.
#define LOOMC_API_PRIVATE LOOMC_API_EXPORT

#endif  // LOOMC_VISIBILITY_H_
