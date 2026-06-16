# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This header licenses this build file only. The shim sources in this directory
# tree are vendored under their own per-file licenses (Apache-2.0; GPL-2.0 WITH
# Linux-syscall-note; dual Apache-2.0/GPL-2.0) -- see shim/README.md for
# provenance and the per-file license map.

if(UNIX)
  add_subdirectory(linux)
endif()

if(WIN32)
  add_subdirectory(windows)
endif()
