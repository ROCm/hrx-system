// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @abs ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @abs() {
      %c0 = arith.constant 0 : index

      %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(4) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
      %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(32) alignment(4) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>

      %input_value = iree_tensor_ext.dispatch.tensor.load %input, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
      %empty = tensor.empty() : tensor<2xf32>
      %result = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%input_value : tensor<2xf32>) outs(%empty : tensor<2xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %abs = math.absf %arg0 : f32
        linalg.yield %abs : f32
      } -> tensor<2xf32>
      iree_tensor_ext.dispatch.tensor.store %result, %output, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>

      return
    }
  }
}
