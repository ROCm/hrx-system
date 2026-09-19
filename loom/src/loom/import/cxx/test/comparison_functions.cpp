// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

unsigned comparison_chain(unsigned first, unsigned second, unsigned third,
                          unsigned fourth, unsigned fifth, unsigned sixth,
                          unsigned seventh) {
  return first < 256u && second < 256u && third < 256u && fourth < 256u &&
         fifth < 256u && sixth < 256u && seventh < 256u;
}
