// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

hal.executable.source public @executable_global {
  hal.executable.export public @export0 ordinal(0) layout(#hal.pipeline.layout<constants = 0, bindings = []>) count(%arg0: !hal.device, %dim: index) -> (index, index, index) {
    hal.return %dim, %dim, %dim : index, index, index
  }
  builtin.module {
    llvm.mlir.global @cts_lookup_global(0 : i32) {addr_space = 1 : i32, alignment = 4 : i64, sym_visibility = "public"} : i32

    func.func @export0() {
      return
    }
  }
}
