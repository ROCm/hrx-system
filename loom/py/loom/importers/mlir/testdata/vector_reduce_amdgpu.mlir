#map = affine_map<(d0) -> (d0 + 1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [1, 1, 1] subgroup_size = 1>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb">

hal.executable private @smoke {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
    hal.executable.export public @vector_reduce_amdgpu layout(#pipeline_layout)
    builtin.module {
      func.func @vector_reduce_amdgpu() attributes {translation_info = #translation} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %zero = arith.constant 0.000000e+00 : f32
        %input = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4xf32>
        %output = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<4xf32>
        %raw_input = amdgpu.fat_raw_buffer_cast %input resetOffset : memref<4xf32> to memref<4xf32, #amdgpu.address_space<fat_raw_buffer>>
        %raw_output = amdgpu.fat_raw_buffer_cast %output resetOffset : memref<4xf32> to memref<4xf32, #amdgpu.address_space<fat_raw_buffer>>
        %vec = vector.transfer_read %raw_input[%c0], %zero {in_bounds = [true]} : memref<4xf32, #amdgpu.address_space<fat_raw_buffer>>, vector<4xf32>
        %sum = vector.reduction <add>, %vec : vector<4xf32> into f32
        %loop_sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %sum) -> (f32) {
          %masked = arith.andi %i, %c1 : index
          %idx = affine.apply #map(%masked)
          %next = arith.addf %acc, %sum : f32
          scf.yield %next : f32
        }
        %broadcast = vector.broadcast %loop_sum : f32 to vector<4xf32>
        vector.transfer_write %broadcast, %raw_output[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #amdgpu.address_space<fat_raw_buffer>>
        return
      }
    }
  }
}
// ----
target.generic<reference> @rocm_hsaco_fb

kernel.def target(@rocm_hsaco_fb) export("vector_reduce_amdgpu") @vector_reduce_amdgpu(%binding0: buffer, %binding1: buffer) {
  %wg_count_x = index.constant 1 : index
  %wg_count_y = index.constant 1 : index
  %wg_count_z = index.constant 1 : index
  %wg_size_x = index.constant 1 : index
  %wg_size_y = index.constant 1 : index
  %wg_size_z = index.constant 1 : index
  kernel.launch.config workgroups(%wg_count_x, %wg_count_y, %wg_count_z) workgroup_size(%wg_size_x, %wg_size_y, %wg_size_z) : index
} launch {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %zero = scalar.constant 0.0 : f32
  %c0_bytes = index.cast %c0 : index to offset
  %input = buffer.view %binding0[%c0_bytes] : buffer -> view<4xf32, #dense>
  %c0_bytes_2 = index.cast %c0 : index to offset
  %output = buffer.view %binding1[%c0_bytes_2] : buffer -> view<4xf32, #dense>
  %vec = vector.load %input[%c0] : view<4xf32, #dense> -> vector<4xf32>
  %sum = vector.reduce<addf> %vec, %zero : vector<4xf32>, f32
  %loop_sum = scf.for %i = [%c0 to %c4 step %c1](%acc = %sum : f32) -> (f32) {
    %masked = index.andi %i, %c1 : index
    %idx = index.add %masked, %c1 : index
    %next = scalar.addf %acc, %sum : f32
    scf.yield %next : f32
  }
  %broadcast = vector.splat %loop_sum : vector<4xf32>
  vector.store %broadcast, %output[%c0] : vector<4xf32>, view<4xf32, #dense>
  kernel.return
}

// ====

#pipeline_layout_barrier = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#translation_barrier = #iree_codegen.translation_info<pipeline = None workgroup_size = [1, 1, 1] subgroup_size = 1>
#executable_target_barrier = #hal.executable.target<"rocm", "rocm-hsaco-fb">

hal.executable private @barrier_smoke {
  hal.executable.variant public @rocm_hsaco_fb target(#executable_target_barrier) {
    hal.executable.export public @barrier_smoke layout(#pipeline_layout_barrier)
    builtin.module {
      func.func @barrier_smoke() attributes {translation_info = #translation_barrier} {
        %c0 = arith.constant 0 : index
        %input = hal.interface.binding.subspan layout(#pipeline_layout_barrier) binding(0) alignment(64) offset(%c0) : memref<4xf32>
        %raw_input = amdgpu.fat_raw_buffer_cast %input resetOffset : memref<4xf32> to memref<4xf32, #amdgpu.address_space<fat_raw_buffer>>
        amdgpu.lds_barrier
        return
      }
    }
  }
}
// ----
target.generic<reference> @rocm_hsaco_fb

kernel.def target(@rocm_hsaco_fb) export("barrier_smoke") @barrier_smoke(%binding0: buffer) {
  %wg_count_x = index.constant 1 : index
  %wg_count_y = index.constant 1 : index
  %wg_count_z = index.constant 1 : index
  %wg_size_x = index.constant 1 : index
  %wg_size_y = index.constant 1 : index
  %wg_size_z = index.constant 1 : index
  kernel.launch.config workgroups(%wg_count_x, %wg_count_y, %wg_count_z) workgroup_size(%wg_size_x, %wg_size_y, %wg_size_z) : index
} launch {
  %c0 = index.constant 0 : index
  %c0_bytes = index.cast %c0 : index to offset
  %input = buffer.view %binding0[%c0_bytes] : buffer -> view<4xf32, #dense>
  kernel.barrier<workgroup> {ordering = acq_rel, scope = workgroup}
  kernel.return
}
