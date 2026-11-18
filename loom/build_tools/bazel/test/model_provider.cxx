// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "model_provider.h"

#include "model_private.h"

extern "C" int model_add_bias(int value) { return value + kPrivateBias; }
