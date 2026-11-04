// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_

// Treat the pinned public ABI declarations as a system-header boundary. This
// keeps their MSVC pragmas and deliberately narrow enum bitfields from
// acquiring libamdf's first-party warning policy without modifying the source.
#if defined(__GNUC__)
#pragma GCC system_header
#endif

// Microsoft's public WSL headers use MSVC warning pragmas around portable C
// types. The definitions themselves are shared by the Linux /dev/dxg ABI.
#include <basetsd.h>
#if !defined(__cplusplus)
#undef C_ASSERT
#define C_ASSERT(expression) _Static_assert((expression), #expression)
#endif
#define D3DKMDT_SPECIAL_MULTIPLATFORM_TOOL
#include <dxg/d3dkmthk.h>
#undef D3DKMDT_SPECIAL_MULTIPLATFORM_TOOL

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_KMT_TYPES_H_
