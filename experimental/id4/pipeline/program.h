// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PROGRAM_H_
#define EXPERIMENTAL_ID4_PIPELINE_PROGRAM_H_

#include <stdint.h>

#include "experimental/id4/pipeline/kernel_library.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/command_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Current inline tensor rank storage limit for semantic pipeline programs.
#define ID4_PIPELINE_PROGRAM_TENSOR_MAX_RANK 5

// Invalid semantic program tensor ordinal.
#define ID4_PIPELINE_PROGRAM_TENSOR_ORDINAL_INVALID UINT32_MAX

// Maximum provider source tensors used to populate one execution parameter.
#define ID4_PIPELINE_PROGRAM_PARAMETER_MAX_SOURCE_COUNT 2

// Immutable semantic pipeline program authored by a stage.
typedef struct id4_pipeline_program_t id4_pipeline_program_t;

// Mutable builder used while authoring a semantic pipeline program.
typedef struct id4_pipeline_program_builder_t id4_pipeline_program_builder_t;

// Scalar element type for semantic tensor values.
typedef enum id4_pipeline_program_dtype_e {
  // Invalid element type.
  ID4_PIPELINE_PROGRAM_DTYPE_INVALID = 0,
  // IEEE 754 single-precision floating point.
  ID4_PIPELINE_PROGRAM_DTYPE_F32 = 1,
  // IEEE 754 half-precision floating point.
  ID4_PIPELINE_PROGRAM_DTYPE_F16 = 2,
  // Brain floating point 16-bit value.
  ID4_PIPELINE_PROGRAM_DTYPE_BF16 = 3,
  // Signed 32-bit integer.
  ID4_PIPELINE_PROGRAM_DTYPE_I32 = 4,
  // Unsigned 32-bit integer.
  ID4_PIPELINE_PROGRAM_DTYPE_U32 = 5,
  // E4M3 8-bit floating point value.
  ID4_PIPELINE_PROGRAM_DTYPE_F8_E4M3 = 6,
} id4_pipeline_program_dtype_t;

// Kind of operation authored into a semantic pipeline program.
typedef enum id4_pipeline_program_op_kind_e {
  // Invalid operation kind.
  ID4_PIPELINE_PROGRAM_OP_KIND_INVALID = 0,
  // External tensor imported by the stage caller.
  ID4_PIPELINE_PROGRAM_OP_KIND_IMPORT = 1,
  // Initialized tensor loaded from the model parameter provider.
  ID4_PIPELINE_PROGRAM_OP_KIND_PARAMETER = 2,
  // Initialized tensor whose contents are embedded in the program.
  ID4_PIPELINE_PROGRAM_OP_KIND_CONSTANT = 3,
  // Uninitialized transient tensor acquired by the program.
  ID4_PIPELINE_PROGRAM_OP_KIND_ACQUIRE = 4,
  // Loom kernel dispatch over explicit tensor bindings.
  ID4_PIPELINE_PROGRAM_OP_KIND_DISPATCH_LOOM = 5,
  // Execution epoch boundary for planning and lifetime analysis.
  ID4_PIPELINE_PROGRAM_OP_KIND_BARRIER = 6,
  // Diagnostic capture point for a tensor value.
  ID4_PIPELINE_PROGRAM_OP_KIND_TAP = 7,
  // External initialized tensor exported by the stage after producer writes.
  ID4_PIPELINE_PROGRAM_OP_KIND_EXPORT = 8,
  // Stage-region scheduling boundary between executable regions.
  ID4_PIPELINE_PROGRAM_OP_KIND_REGION_CUT = 9,
} id4_pipeline_program_op_kind_t;

// Tensor access flags observed by a semantic dispatch.
typedef uint32_t id4_pipeline_program_tensor_access_flags_t;

// Tensor access flag bits.
typedef enum id4_pipeline_program_tensor_access_flag_bits_e {
  // Dispatch reads the tensor contents.
  ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ = 1u << 0,
  // Dispatch writes the tensor contents.
  ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE = 1u << 1,
} id4_pipeline_program_tensor_access_flag_bits_t;

// Semantic dispatch binding behavior flags.
typedef uint32_t id4_pipeline_program_dispatch_binding_flags_t;

// Semantic dispatch binding behavior flag bits.
typedef enum id4_pipeline_program_dispatch_binding_flag_bits_e {
  // write_range describes the byte interval written by this binding.
  ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE = 1u << 0,
  // read_range describes the byte interval read by this binding.
  ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_READ_RANGE = 1u << 1,
} id4_pipeline_program_dispatch_binding_flag_bits_t;

// Tensor byte interval relative to the bound logical tensor.
typedef struct id4_pipeline_program_tensor_byte_range_t {
  // Byte offset from the start of the logical tensor.
  iree_device_size_t offset;
  // Byte length of the interval.
  iree_device_size_t length;
} id4_pipeline_program_tensor_byte_range_t;

// External tensor import flags.
typedef uint32_t id4_pipeline_program_import_tensor_flags_t;

// External tensor import flag bits.
typedef enum id4_pipeline_program_import_tensor_flag_bits_e {
  // Imported tensor contents are initialized before stage execution begins.
  ID4_PIPELINE_PROGRAM_IMPORT_TENSOR_FLAG_INITIALIZED = 1u << 0,
} id4_pipeline_program_import_tensor_flag_bits_t;

// Inline tensor shape used by semantic program values.
typedef struct id4_pipeline_program_shape_t {
  // Number of used dimensions in dims.
  uint32_t rank;
  // Dimension extents stored inline up to the current rank limit.
  uint64_t dims[ID4_PIPELINE_PROGRAM_TENSOR_MAX_RANK];
} id4_pipeline_program_shape_t;

// Value handle for a tensor known to a semantic program or builder.
typedef struct id4_pipeline_program_tensor_t {
  // Program-local tensor ordinal.
  uint32_t ordinal;
} id4_pipeline_program_tensor_t;

// Prepare-time transformation from provider sources to execution storage.
typedef enum id4_pipeline_program_parameter_encoding_e {
  // Invalid parameter encoding.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_INVALID = 0,
  // Copies one provider source tensor directly into execution storage.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_DIRECT = 1,
  // Converts FP8 e4m3 source weights and F32 row scales to BF16 storage.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16 = 2,
  // Packs BF16 matrix weights into compact 16x16 RHS tiles for WMMA.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_BF16_LINEAR_RHS_TILE = 3,
  // Converts FP8 e4m3 scaled weights into compact BF16 RHS tiles.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_SCALED_TO_BF16_LINEAR_RHS_TILE =
      4,
  // Packs FP8 e4m3 matrix weights into compact 16x16 RHS tiles.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_LINEAR_RHS_TILE = 5,
  // Converts FP8 e4m3 block-scaled weights into compact BF16 RHS tiles.
  ID4_PIPELINE_PROGRAM_PARAMETER_ENCODING_FP8_E4M3_BLOCK_SCALED_TO_BF16_LINEAR_RHS_TILE =
      6,
} id4_pipeline_program_parameter_encoding_t;

// Provider tensor used while preparing one execution parameter.
typedef struct id4_pipeline_program_parameter_source_t {
  // Provider scope containing this source tensor.
  iree_string_view_t source_scope;
  // Provider key for this source tensor.
  iree_string_view_t key;
  // Scalar element type stored by the provider source.
  id4_pipeline_program_dtype_t dtype;
  // Logical shape stored by the provider source.
  id4_pipeline_program_shape_t shape;
} id4_pipeline_program_parameter_source_t;

// Provider-to-execution byte span used to assemble a direct parameter tensor.
typedef struct id4_pipeline_program_parameter_source_span_t {
  // Byte offset in the provider source tensor.
  uint64_t source_offset;
  // Byte offset in the execution parameter tensor.
  iree_device_size_t target_offset;
  // Byte length copied from the provider source into the execution tensor.
  iree_device_size_t length;
} id4_pipeline_program_parameter_source_span_t;

// Tensor metadata copied into a semantic program.
typedef struct id4_pipeline_program_tensor_record_t {
  // Stable tensor name used for diagnostics, taps, and plan dumps.
  iree_string_view_t name;
  // Scalar element type.
  id4_pipeline_program_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_program_shape_t shape;
  // Dense tensor byte length derived from dtype and shape.
  iree_device_size_t byte_length;
  // Operation ordinal that introduced the tensor.
  iree_host_size_t producer_operation_ordinal;
} id4_pipeline_program_tensor_record_t;

// Tensor binding passed to one semantic Loom dispatch.
typedef struct id4_pipeline_program_dispatch_binding_t {
  // Tensor bound to the dispatch argument.
  id4_pipeline_program_tensor_t tensor;
  // Access performed by the dispatch.
  id4_pipeline_program_tensor_access_flags_t access;
  // Dispatch binding behavior flags.
  id4_pipeline_program_dispatch_binding_flags_t flags;
  // Tensor-relative byte range covered by reads when READ_RANGE is set.
  id4_pipeline_program_tensor_byte_range_t read_range;
  // Tensor-relative byte range covered by writes when WRITE_RANGE is set.
  id4_pipeline_program_tensor_byte_range_t write_range;
} id4_pipeline_program_dispatch_binding_t;

// Imported tensor operation payload.
typedef struct id4_pipeline_program_import_op_t {
  // Import flags describing the external tensor boundary contract.
  id4_pipeline_program_import_tensor_flags_t flags;
  // Imported external tensor.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_import_op_t;

// Parameter tensor operation payload.
typedef struct id4_pipeline_program_parameter_op_t {
  // Prepare-time source-to-execution transformation.
  id4_pipeline_program_parameter_encoding_t encoding;
  // Number of provider source descriptors.
  iree_host_size_t source_count;
  // Provider source descriptors copied into the program.
  const id4_pipeline_program_parameter_source_t* sources;
  // Number of explicit direct source spans.
  iree_host_size_t source_span_count;
  // Explicit direct source spans copied into the program.
  const id4_pipeline_program_parameter_source_span_t* source_spans;
  // Initialized tensor whose record name is the provider key.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_parameter_op_t;

// Constant tensor operation payload.
typedef struct id4_pipeline_program_constant_op_t {
  // Initialized tensor whose record name is the constant diagnostic name.
  id4_pipeline_program_tensor_t tensor;
  // Constant data byte length.
  iree_host_size_t data_length;
  // Constant data bytes owned by the containing program operation.
  const uint8_t* data;
} id4_pipeline_program_constant_op_t;

// Transient tensor acquire operation payload.
typedef struct id4_pipeline_program_acquire_op_t {
  // Acquired uninitialized transient tensor.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_acquire_op_t;

// Loom dispatch operation payload.
typedef struct id4_pipeline_program_dispatch_loom_op_t {
  // Stable operation name used for diagnostics and specialization keys.
  iree_string_view_t name;
  // Exported Loom kernel selected by the program.
  id4_pipeline_kernel_ref_t kernel;
  // Number of copied Loom config bindings.
  iree_host_size_t config_binding_count;
  // Copied Loom config bindings.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
  // Number of tensor bindings passed to the kernel.
  iree_host_size_t binding_count;
  // Tensor bindings in kernel ABI order.
  const id4_pipeline_program_dispatch_binding_t* bindings;
} id4_pipeline_program_dispatch_loom_op_t;

// Execution barrier operation payload.
typedef struct id4_pipeline_program_barrier_op_t {
  // Stable barrier name used for diagnostics.
  iree_string_view_t name;
} id4_pipeline_program_barrier_op_t;

// Stage-region cut operation payload.
typedef struct id4_pipeline_program_region_cut_op_t {
  // Stable cut name used for region diagnostics and scheduling.
  iree_string_view_t name;
} id4_pipeline_program_region_cut_op_t;

// Diagnostic tap operation payload.
typedef struct id4_pipeline_program_tap_op_t {
  // Stable tap name used for diagnostics and trace matching.
  iree_string_view_t name;
  // Initialized tensor captured by the tap.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_tap_op_t;

// Export operation payload.
typedef struct id4_pipeline_program_export_op_t {
  // Stable exported value name.
  iree_string_view_t name;
  // Initialized tensor exported by the stage.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_export_op_t;

// Operation record copied into a semantic program.
typedef struct id4_pipeline_program_op_t {
  // Operation kind selecting the payload.
  id4_pipeline_program_op_kind_t kind;
  // Program-local operation ordinal.
  iree_host_size_t ordinal;
  // Kind-selected operation payload.
  union {
    // Import operation payload.
    id4_pipeline_program_import_op_t import_value;
    // Parameter operation payload.
    id4_pipeline_program_parameter_op_t parameter;
    // Constant operation payload.
    id4_pipeline_program_constant_op_t constant;
    // Acquire operation payload.
    id4_pipeline_program_acquire_op_t acquire;
    // Loom dispatch operation payload.
    id4_pipeline_program_dispatch_loom_op_t dispatch_loom;
    // Barrier operation payload.
    id4_pipeline_program_barrier_op_t barrier;
    // Stage-region cut operation payload.
    id4_pipeline_program_region_cut_op_t region_cut;
    // Diagnostic tap operation payload.
    id4_pipeline_program_tap_op_t tap;
    // Export operation payload.
    id4_pipeline_program_export_op_t export_value;
  } payload;
} id4_pipeline_program_op_t;

// Options for creating a semantic program builder.
typedef struct id4_pipeline_program_builder_create_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Program name borrowed for the create call and copied by the builder.
  iree_string_view_t program_name;
  // Arena block pool used for append-only builder transient allocations.
  iree_arena_block_pool_t* block_pool;
} id4_pipeline_program_builder_create_options_t;

// Options for importing an external stage-boundary tensor.
typedef struct id4_pipeline_program_import_tensor_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Import flags describing the external tensor boundary contract.
  id4_pipeline_program_import_tensor_flags_t flags;
  // Stable tensor name used by the stage boundary.
  iree_string_view_t name;
  // Scalar element type.
  id4_pipeline_program_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_program_shape_t shape;
} id4_pipeline_program_import_tensor_options_t;

// Options for adding a model parameter tensor.
typedef struct id4_pipeline_program_parameter_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Prepare-time source-to-execution transformation.
  id4_pipeline_program_parameter_encoding_t encoding;
  // Number of provider source descriptors.
  iree_host_size_t source_count;
  // Provider source descriptors borrowed for the call and copied by builder.
  const id4_pipeline_program_parameter_source_t* sources;
  // Execution tensor key and diagnostic name.
  iree_string_view_t key;
  // Scalar element type.
  id4_pipeline_program_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_program_shape_t shape;
  // Number of explicit direct source spans.
  iree_host_size_t source_span_count;
  // Explicit direct source spans borrowed for the call and copied by builder.
  const id4_pipeline_program_parameter_source_span_t* source_spans;
} id4_pipeline_program_parameter_options_t;

// Options for adding a program-owned constant tensor.
typedef struct id4_pipeline_program_constant_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable tensor name used for diagnostics.
  iree_string_view_t name;
  // Scalar element type.
  id4_pipeline_program_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_program_shape_t shape;
  // Constant data bytes copied by the builder.
  iree_const_byte_span_t data;
} id4_pipeline_program_constant_options_t;

// Options for acquiring an uninitialized transient tensor.
typedef struct id4_pipeline_program_acquire_tensor_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable tensor name used for diagnostics.
  iree_string_view_t name;
  // Scalar element type.
  id4_pipeline_program_dtype_t dtype;
  // Tensor shape.
  id4_pipeline_program_shape_t shape;
} id4_pipeline_program_acquire_tensor_options_t;

// Options for authoring a Loom dispatch.
typedef struct id4_pipeline_program_dispatch_loom_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable operation name used for diagnostics and specialization keys.
  iree_string_view_t name;
  // Exported Loom kernel selected by the program.
  id4_pipeline_kernel_ref_t kernel;
  // Number of Loom config bindings.
  iree_host_size_t config_binding_count;
  // Loom config bindings borrowed for the dispatch call.
  const id4_pipeline_kernel_config_binding_t* config_bindings;
  // Number of tensor bindings passed to the kernel.
  iree_host_size_t binding_count;
  // Tensor bindings in kernel ABI order.
  const id4_pipeline_program_dispatch_binding_t* bindings;
} id4_pipeline_program_dispatch_loom_options_t;

// Options for authoring an execution barrier.
typedef struct id4_pipeline_program_barrier_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable barrier name used for diagnostics.
  iree_string_view_t name;
} id4_pipeline_program_barrier_options_t;

// Options for authoring a stage-region cut.
typedef struct id4_pipeline_program_region_cut_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable cut name used for region diagnostics and scheduling.
  iree_string_view_t name;
} id4_pipeline_program_region_cut_options_t;

// Options for authoring a diagnostic tap.
typedef struct id4_pipeline_program_tap_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable tap name used for diagnostics and trace matching.
  iree_string_view_t name;
  // Initialized tensor captured by the tap.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_tap_options_t;

// Options for authoring a stage export.
typedef struct id4_pipeline_program_export_options_t {
  // Size of this structure for versioning.
  iree_host_size_t structure_size;
  // Extension structure chain; must be NULL for now.
  const void* next;
  // Stable exported value name.
  iree_string_view_t name;
  // Initialized tensor exported by the stage.
  id4_pipeline_program_tensor_t tensor;
} id4_pipeline_program_export_options_t;

// Returns an invalid semantic program tensor handle.
static inline id4_pipeline_program_tensor_t id4_pipeline_program_tensor_invalid(
    void) {
  id4_pipeline_program_tensor_t tensor;
  tensor.ordinal = ID4_PIPELINE_PROGRAM_TENSOR_ORDINAL_INVALID;
  return tensor;
}

// Returns true when |tensor| is a valid semantic program tensor handle.
static inline bool id4_pipeline_program_tensor_is_valid(
    id4_pipeline_program_tensor_t tensor) {
  return tensor.ordinal != ID4_PIPELINE_PROGRAM_TENSOR_ORDINAL_INVALID;
}

// Returns a scalar tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank0(void) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 0;
  return shape;
}

// Returns a rank-1 tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank1(uint64_t dim0) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 1;
  shape.dims[0] = dim0;
  return shape;
}

// Returns a rank-2 tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank2(uint64_t dim0, uint64_t dim1) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 2;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  return shape;
}

// Returns a rank-3 tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank3(uint64_t dim0, uint64_t dim1,
                                      uint64_t dim2) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 3;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  shape.dims[2] = dim2;
  return shape;
}

// Returns a rank-4 tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank4(uint64_t dim0, uint64_t dim1,
                                      uint64_t dim2, uint64_t dim3) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 4;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  shape.dims[2] = dim2;
  shape.dims[3] = dim3;
  return shape;
}

// Returns a rank-5 tensor shape.
static inline id4_pipeline_program_shape_t
id4_pipeline_program_make_shape_rank5(uint64_t dim0, uint64_t dim1,
                                      uint64_t dim2, uint64_t dim3,
                                      uint64_t dim4) {
  id4_pipeline_program_shape_t shape = {0};
  shape.rank = 5;
  shape.dims[0] = dim0;
  shape.dims[1] = dim1;
  shape.dims[2] = dim2;
  shape.dims[3] = dim3;
  shape.dims[4] = dim4;
  return shape;
}

// Returns a dispatch binding with explicit tensor access flags.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_dispatch_binding(
    id4_pipeline_program_tensor_t tensor,
    id4_pipeline_program_tensor_access_flags_t access) {
  return (id4_pipeline_program_dispatch_binding_t){
      // Tensor bound to the dispatch argument.
      /*.tensor=*/tensor,
      // Access performed by the dispatch.
      /*.access=*/access,
      // No optional binding behavior flags.
      /*.flags=*/0,
      // No explicit read coverage for whole-tensor reads.
      /*.read_range=*/{0, 0},
      // No explicit write coverage for whole-tensor writes.
      /*.write_range=*/{0, 0},
  };
}

// Returns a dispatch binding with explicit tensor read coverage.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_dispatch_binding_read_range(
    id4_pipeline_program_tensor_t tensor,
    id4_pipeline_program_tensor_access_flags_t access,
    iree_device_size_t offset, iree_device_size_t length) {
  id4_pipeline_program_dispatch_binding_t binding =
      id4_pipeline_program_dispatch_binding(tensor, access);
  binding.flags = ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_READ_RANGE;
  binding.read_range = (id4_pipeline_program_tensor_byte_range_t){
      // Byte offset from the start of the logical tensor.
      /*.offset=*/offset,
      // Byte length of the interval.
      /*.length=*/length,
  };
  return binding;
}

// Returns a dispatch binding with explicit tensor write coverage.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_dispatch_binding_write_range(
    id4_pipeline_program_tensor_t tensor,
    id4_pipeline_program_tensor_access_flags_t access,
    iree_device_size_t offset, iree_device_size_t length) {
  id4_pipeline_program_dispatch_binding_t binding =
      id4_pipeline_program_dispatch_binding(tensor, access);
  binding.flags = ID4_PIPELINE_PROGRAM_DISPATCH_BINDING_FLAG_WRITE_RANGE;
  binding.write_range = (id4_pipeline_program_tensor_byte_range_t){
      // Byte offset from the start of the logical tensor.
      /*.offset=*/offset,
      // Byte length of the interval.
      /*.length=*/length,
  };
  return binding;
}

// Returns a read-only dispatch binding.
static inline id4_pipeline_program_dispatch_binding_t id4_pipeline_program_read(
    id4_pipeline_program_tensor_t tensor) {
  return id4_pipeline_program_dispatch_binding(
      tensor, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ);
}

// Returns a read-only dispatch binding with explicit byte coverage.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_read_range(id4_pipeline_program_tensor_t tensor,
                                iree_device_size_t offset,
                                iree_device_size_t length) {
  return id4_pipeline_program_dispatch_binding_read_range(
      tensor, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ, offset, length);
}

// Returns a write-only dispatch binding.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_write(id4_pipeline_program_tensor_t tensor) {
  return id4_pipeline_program_dispatch_binding(
      tensor, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE);
}

// Returns a write-only dispatch binding with explicit byte coverage.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_write_range(id4_pipeline_program_tensor_t tensor,
                                 iree_device_size_t offset,
                                 iree_device_size_t length) {
  return id4_pipeline_program_dispatch_binding_write_range(
      tensor, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE, offset, length);
}

// Returns a read-write dispatch binding.
static inline id4_pipeline_program_dispatch_binding_t
id4_pipeline_program_read_write(id4_pipeline_program_tensor_t tensor) {
  return id4_pipeline_program_dispatch_binding(
      tensor, ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_READ |
                  ID4_PIPELINE_PROGRAM_TENSOR_ACCESS_WRITE);
}

// Returns the dense byte width of one scalar element or zero for invalid
// dtypes.
iree_device_size_t id4_pipeline_program_dtype_byte_length(
    id4_pipeline_program_dtype_t dtype);

// Returns the dense element count for |shape|.
iree_status_t id4_pipeline_program_shape_element_count(
    id4_pipeline_program_shape_t shape, uint64_t* out_element_count);

// Returns the dense tensor byte length for |dtype| and |shape|.
iree_status_t id4_pipeline_program_tensor_byte_length(
    id4_pipeline_program_dtype_t dtype, id4_pipeline_program_shape_t shape,
    iree_device_size_t* out_byte_length);

// Creates a mutable semantic program builder.
iree_status_t id4_pipeline_program_builder_create(
    const id4_pipeline_program_builder_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_program_builder_t** out_builder);

// Destroys a mutable semantic program builder.
void id4_pipeline_program_builder_destroy(
    id4_pipeline_program_builder_t* builder);

// Imports an external tensor from the stage boundary.
iree_status_t id4_pipeline_program_import_tensor(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_import_tensor_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor);

// Adds an initialized model parameter tensor.
iree_status_t id4_pipeline_program_parameter(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_parameter_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor);

// Adds an initialized constant tensor owned by the program.
iree_status_t id4_pipeline_program_constant(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_constant_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor);

// Acquires an uninitialized transient tensor.
iree_status_t id4_pipeline_program_acquire_tensor(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_acquire_tensor_options_t* options,
    id4_pipeline_program_tensor_t* out_tensor);

// Authors a Loom dispatch over explicit tensor bindings.
iree_status_t id4_pipeline_program_dispatch_loom(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_dispatch_loom_options_t* options);

// Authors an execution barrier and advances the planning epoch.
iree_status_t id4_pipeline_program_barrier(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_barrier_options_t* options);

// Authors a stage-region scheduling cut.
iree_status_t id4_pipeline_program_region_cut(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_region_cut_options_t* options);

// Authors a diagnostic tap for an initialized tensor.
iree_status_t id4_pipeline_program_tap(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_tap_options_t* options);

// Authors a stage export for an initialized tensor.
iree_status_t id4_pipeline_program_export(
    id4_pipeline_program_builder_t* builder,
    const id4_pipeline_program_export_options_t* options);

// Seals a builder into an immutable semantic program.
iree_status_t id4_pipeline_program_builder_seal(
    const id4_pipeline_program_builder_t* builder,
    iree_allocator_t host_allocator, id4_pipeline_program_t** out_program);

// Retains |program| for the caller.
void id4_pipeline_program_retain(id4_pipeline_program_t* program);

// Releases |program| from the caller.
void id4_pipeline_program_release(id4_pipeline_program_t* program);

// Returns the program name copied into |program|.
iree_string_view_t id4_pipeline_program_name(
    const id4_pipeline_program_t* program);

// Returns the number of tensors in |program|.
iree_host_size_t id4_pipeline_program_tensor_count(
    const id4_pipeline_program_t* program);

// Returns tensor record |index| or NULL when out of range.
const id4_pipeline_program_tensor_record_t* id4_pipeline_program_tensor_at(
    const id4_pipeline_program_t* program, iree_host_size_t index);

// Returns the number of operations in |program|.
iree_host_size_t id4_pipeline_program_operation_count(
    const id4_pipeline_program_t* program);

// Returns operation |index| or NULL when out of range.
const id4_pipeline_program_op_t* id4_pipeline_program_operation_at(
    const id4_pipeline_program_t* program, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PROGRAM_H_
