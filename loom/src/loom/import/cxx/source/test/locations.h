// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#define HEADER_LINE __LINE__
#line 600 "presumed_header.h"
static int header_line() { return __LINE__; }

#define HEADER_JOIN(left, right) left##right
static int header_pasted() { return HEADER_JOIN(4, 2); }
