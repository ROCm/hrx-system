// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_VISIBILITY_H_
#define LOOMC_VISIBILITY_H_

// Marks implementation symbols shared between source files but not exported as
// part of the public Loom C API dynamic library surface.
#if defined(_WIN32)
#define LOOMC_API_PRIVATE
#elif defined(__GNUC__) || defined(__clang__)
#define LOOMC_API_PRIVATE __attribute__((visibility("hidden")))
#else
#define LOOMC_API_PRIVATE
#endif

#endif  // LOOMC_VISIBILITY_H_
