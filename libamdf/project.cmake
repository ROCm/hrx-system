# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

option(AMDF_BUILD
  "Build the portable AMD GPU and XDNA native device library."
  OFF)

option(AMDF_FAMILY_RDNA "Admit RDNA implementation packages." ON)
option(AMDF_FAMILY_CDNA "Admit CDNA implementation packages." ON)
option(AMDF_FAMILY_XDNA "Admit XDNA implementation packages." ON)

set(AMDF_VERSION "0.1.0")
set(AMDF_ABI_VERSION "1")
