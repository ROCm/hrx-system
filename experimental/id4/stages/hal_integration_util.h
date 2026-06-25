// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_
#define EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_

#include <string>
#include <vector>

#include "experimental/id4/pipeline/diagnostics.h"
#include "experimental/id4/pipeline/kernel_cache.h"
#include "experimental/id4/pipeline/kernel_library.h"
#include "experimental/id4/pipeline/plan.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

namespace id4::test {

template <typename T, void (*Release)(T*)>
class OwningRef {
 public:
  OwningRef() = default;
  OwningRef(const OwningRef&) = delete;
  OwningRef& operator=(const OwningRef&) = delete;

  ~OwningRef() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  // Owned reference released by this wrapper.
  T* value_ = nullptr;
};

using FrontierTrackerRef = OwningRef<iree_async_frontier_tracker_t,
                                     iree_async_frontier_tracker_release>;
using HalDeviceRef = OwningRef<iree_hal_device_t, iree_hal_device_release>;
using HalDeviceGroupRef =
    OwningRef<iree_hal_device_group_t, iree_hal_device_group_release>;
using HalExecutableCacheRef =
    OwningRef<iree_hal_executable_cache_t, iree_hal_executable_cache_release>;
using KernelCacheRef =
    OwningRef<id4_pipeline_kernel_cache_t, id4_pipeline_kernel_cache_release>;
using KernelLibraryRef = OwningRef<id4_pipeline_kernel_library_t,
                                   id4_pipeline_kernel_library_release>;
using ProactorPoolRef =
    OwningRef<iree_async_proactor_pool_t, iree_async_proactor_pool_release>;

struct SemaphoreListStorage {
  // Semaphore carried by this single-entry list.
  iree_hal_semaphore_t* semaphore = nullptr;
  // Payload value paired with the semaphore.
  uint64_t payload_value = 0;

  // Returns a HAL semaphore list backed by this storage.
  iree_hal_semaphore_list_t list();
};

struct FixedSemaphoreListStorage {
  // Number of live semaphore edges in this storage.
  iree_host_size_t count = 0;
  // Stack-owned semaphore handles.
  iree_hal_semaphore_t* semaphores[4] = {};
  // Stack-owned payload values.
  uint64_t payload_values[4] = {};

  // Appends a semaphore edge to this stack-backed list.
  iree_status_t push(iree_hal_semaphore_t* semaphore, uint64_t payload_value);

  // Returns a HAL semaphore list backed by this storage.
  iree_hal_semaphore_list_t list();
};

class BufferBindingSet {
 public:
  BufferBindingSet() = default;
  BufferBindingSet(const BufferBindingSet&) = delete;
  BufferBindingSet& operator=(const BufferBindingSet&) = delete;

  ~BufferBindingSet();

  // Releases all owned buffers and binding arrays.
  void reset();

  // Number of bindings allocated from the plan.
  iree_host_size_t count = 0;
  // Owned HAL buffers backing each binding.
  iree_hal_buffer_t** buffers = nullptr;
  // Binding table entries in plan order.
  iree_hal_buffer_binding_t* bindings = nullptr;
};

typedef struct FixtureTensor {
  // Stable tensor name from the fixture manifest.
  std::string name;
  // Stable stage name from the fixture manifest.
  std::string stage;
  // Fixture tensor role such as input or expected.
  std::string role;
  // Relative tensor payload path from the fixture manifest.
  std::string file;
  // Tensor dtype declared by the fixture manifest and validated against
  // payload.
  id4_pipeline_tensor_dtype_t dtype = ID4_PIPELINE_TENSOR_DTYPE_INVALID;
  // Tensor shape declared by the fixture manifest and validated against
  // payload.
  id4_pipeline_tensor_shape_t shape = {};
  // Full source tensor shape before fixture slicing.
  id4_pipeline_tensor_shape_t source_shape = {};
  // Per-dimension source offsets for this payload slice.
  id4_pipeline_tensor_shape_t slice_offsets = {};
  // Absolute tolerance used when this tensor is an expected value.
  double absolute_tolerance = 0.0;
  // Relative tolerance used when this tensor is an expected value.
  double relative_tolerance = 0.0;
  // True when tolerance metadata was present in the fixture manifest.
  bool has_tolerance = false;
  // Raw dense tensor bytes parsed from the payload.
  std::vector<uint8_t> payload;
} FixtureTensor;

typedef struct FixtureTensorSet {
  // Fixture directory containing manifest.json and payload files.
  std::string directory;
  // Tensor payloads loaded from the fixture manifest.
  std::vector<FixtureTensor> tensors;

  // Returns the loaded tensor with |name| or NULL when absent.
  const FixtureTensor* FindTensor(iree_string_view_t name) const;
  // Returns the loaded tensor with |role| and |name| or NULL when absent.
  const FixtureTensor* FindTensor(iree_string_view_t role,
                                  iree_string_view_t name) const;
  // Returns the loaded tensor with |role|, |stage|, and |name| or NULL when
  // absent.
  const FixtureTensor* FindTensor(iree_string_view_t role,
                                  iree_string_view_t stage,
                                  iree_string_view_t name) const;
} FixtureTensorSet;

typedef struct LiveStageContext {
  // Proactor pool used by the live HAL device.
  ProactorPoolRef proactor_pool;
  // Frontier tracker retained by the HAL device group.
  FrontierTrackerRef frontier_tracker;
  // Device group passed through the ID4 stage API.
  HalDeviceGroupRef device_group;
  // HAL device selected by the parsed --device= flag.
  HalDeviceRef device;
  // HAL executable cache used by prepared stage kernels.
  HalExecutableCacheRef executable_cache;
  // Loom kernel cache configured for the selected live device.
  KernelCacheRef kernel_cache;
  // Command-buffer mode selected from standard device/profiling flags.
  iree_hal_command_buffer_mode_t command_buffer_mode =
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT;
} LiveStageContext;

typedef struct StageDiagnostics {
  // Number of diagnostic events observed.
  iree_host_size_t event_count;
  // Number of kernel diagnostic events observed.
  iree_host_size_t kernel_event_count;
} StageDiagnostics;

// Returns a diagnostics sink that counts lifecycle and kernel events.
id4_pipeline_diagnostics_sink_t DiagnosticsSink(StageDiagnostics* diagnostics);

// Creates a live HAL context from the standard --device= flag.
iree_status_t CreateLiveStageContextFromFlags(LiveStageContext* out_context);

// Creates a kernel library from embedded ID4 Loom source files.
iree_status_t CreateEmbeddedKernelLibrary(
    id4_pipeline_kernel_library_t** out_library);

// Creates a parameter provider for |scope| from parsed --parameters flags.
iree_status_t CreateParameterProviderFromFlags(
    iree_string_view_t scope, iree_io_parameter_provider_t** out_provider);

// Allocates device-local buffers for every boundary tensor in |plan|.
iree_status_t AllocateBoundaryBindings(iree_hal_device_t* device,
                                       iree_hal_queue_affinity_t queue_affinity,
                                       const id4_pipeline_plan_t* plan,
                                       BufferBindingSet* out_binding_set);

// Allocates device-local buffers for every diagnostic tap tensor in |plan|.
iree_status_t AllocateDiagnosticTapBindings(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, BufferBindingSet* out_binding_set);

// Finds a boundary binding by planned boundary tensor name.
iree_status_t FindBoundaryBinding(const id4_pipeline_plan_t* plan,
                                  const BufferBindingSet& binding_set,
                                  iree_string_view_t name,
                                  iree_hal_buffer_binding_t* out_binding);

// Finds a diagnostic tap binding by planned tap name.
iree_status_t FindDiagnosticTapBinding(const id4_pipeline_plan_t* plan,
                                       const BufferBindingSet& binding_set,
                                       iree_string_view_t name,
                                       iree_hal_buffer_binding_t* out_binding);

// Queues a direct update into |binding| and advances |inout_payload_value|.
iree_status_t QueueUpdateBinding(iree_hal_device_t* device,
                                 iree_hal_queue_affinity_t queue_affinity,
                                 const iree_hal_buffer_binding_t* binding,
                                 const void* source_data,
                                 iree_host_size_t source_length,
                                 iree_hal_semaphore_t* semaphore,
                                 uint64_t* inout_payload_value);

// Queues a file read from host allocation bytes into |binding|.
iree_status_t QueueReadBindingFromHostAllocation(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding, const void* source_data,
    iree_host_size_t source_length, iree_hal_semaphore_t* semaphore,
    uint64_t* inout_payload_value);

// Queues a fill of |binding| and advances |inout_payload_value|.
iree_status_t QueueFillBinding(iree_hal_device_t* device,
                               iree_hal_queue_affinity_t queue_affinity,
                               const iree_hal_buffer_binding_t* binding,
                               const void* pattern,
                               iree_host_size_t pattern_length,
                               iree_hal_semaphore_t* semaphore,
                               uint64_t* inout_payload_value);

// Queues fills for boundary tensors whose flags contain |required_flags|.
iree_status_t QueueFillBoundaryTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    id4_pipeline_boundary_tensor_flags_t required_flags, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value);

// Queues fills for all diagnostic tap tensors.
iree_status_t QueueFillDiagnosticTapTensors(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const BufferBindingSet& binding_set, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_semaphore_t* fill_semaphore,
    uint64_t* out_fill_value);

// Queues one fixture input tensor into a named boundary tensor.
iree_status_t QueueUpdateBoundaryTensorFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    iree_string_view_t boundary_name, const FixtureTensorSet& fixture_tensors,
    iree_string_view_t fixture_name, iree_hal_semaphore_t* update_semaphore,
    uint64_t* inout_update_value);

// Reads |binding| into a host byte vector after |wait_list| is satisfied.
iree_status_t ReadBindingToHost(iree_hal_device_t* device,
                                iree_hal_queue_affinity_t queue_affinity,
                                const iree_hal_buffer_binding_t* binding,
                                iree_hal_semaphore_list_t wait_list,
                                std::vector<uint8_t>* out_bytes);

// Reads |binding| and verifies it matches an expected F32 fixture tensor.
iree_status_t CompareF32BindingWithFixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list, const FixtureTensor& expected_tensor);

// Reads a BF16 |binding| and verifies it exactly matches the expected F32
// fixture tensor after F32-to-BF16 rounding.
iree_status_t CompareBf16BindingWithRoundedF32FixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const FixtureTensor& expected_tensor);

// Reads |binding| and verifies it matches an expected F32 fixture tensor.
//
// |actual_layout| describes the binding's planned dtype and shape. F32, F16,
// and BF16 actual storage are decoded to F32 before comparison.
iree_status_t CompareBindingWithFixtureTensor(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_buffer_binding_t* binding,
    iree_hal_semaphore_list_t wait_list,
    const id4_pipeline_tensor_layout_t* actual_layout,
    const FixtureTensor& expected_tensor);

// Loads fixture tensors from a fixture manifest directory.
iree_status_t LoadFixtureTensors(iree_string_view_t fixture_directory,
                                 FixtureTensorSet* out_fixture_tensors);

// Reads a uint32-compatible length from a rank-1 fixture tensor.
iree_status_t InferRank1TensorLengthFromFixture(
    const FixtureTensorSet& fixture_tensors, iree_string_view_t tensor_name,
    id4_pipeline_tensor_dtype_t dtype, uint32_t* out_length);

// Queues updates for all initialized boundary tensors from fixture inputs.
iree_status_t QueueUpdateInitializedBoundaryTensorsFromFixture(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const id4_pipeline_plan_t* plan, const BufferBindingSet& binding_set,
    const FixtureTensorSet& fixture_tensors,
    iree_hal_semaphore_t* fill_semaphore, uint64_t* out_fill_value);

// Verifies exported boundary capture payloads differ from |sentinel|.
iree_status_t VerifyCapturedExportedBoundaryTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory,
    uint8_t sentinel);

// Verifies diagnostic tap capture payloads differ from |sentinel|.
iree_status_t VerifyCapturedDiagnosticTapTensorsWereWritten(
    const id4_pipeline_plan_t* plan, iree_string_view_t capture_directory,
    uint8_t sentinel);

}  // namespace id4::test

#endif  // EXPERIMENTAL_ID4_STAGES_HAL_INTEGRATION_UTIL_H_
