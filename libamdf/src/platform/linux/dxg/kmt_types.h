// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_

// Microsoft's public WSL headers use MSVC warning pragmas around portable C
// types. The definitions themselves are shared by the Linux /dev/dxg ABI.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#endif
#include <basetsd.h>
#if !defined(__cplusplus)
#undef C_ASSERT
#define C_ASSERT(expression) _Static_assert((expression), #expression)
#endif
#define D3DKMDT_SPECIAL_MULTIPLATFORM_TOOL
#include <dxg/d3dkmthk.h>
#undef D3DKMDT_SPECIAL_MULTIPLATFORM_TOOL
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_
