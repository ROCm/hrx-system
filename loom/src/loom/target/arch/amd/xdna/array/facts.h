// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable physical-array facts shared by AMD XDNA device profiles.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_FACTS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_FACTS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independent AIE instruction-set identity selected by an array family.
typedef enum loom_xdna_architecture_e {
  LOOM_XDNA_ARCHITECTURE_AIE2P = 3,
} loom_xdna_architecture_t;

// Physical tile role.
typedef enum loom_xdna_tile_kind_e {
  LOOM_XDNA_TILE_KIND_SHIM_NOC = 1,
  LOOM_XDNA_TILE_KIND_MEMORY = 2,
  LOOM_XDNA_TILE_KIND_COMPUTE = 3,
} loom_xdna_tile_kind_t;

// Independently addressed configuration-register module.
typedef enum loom_xdna_register_module_e {
  LOOM_XDNA_REGISTER_MODULE_CORE = 1,
  LOOM_XDNA_REGISTER_MODULE_COMPUTE_MEMORY = 2,
  LOOM_XDNA_REGISTER_MODULE_MEMORY_TILE = 3,
  LOOM_XDNA_REGISTER_MODULE_SHIM_NOC = 4,
  LOOM_XDNA_REGISTER_MODULE_SHIM_PL = 5,
} loom_xdna_register_module_t;

#define LOOM_XDNA_REGISTER_MODULE_BIT(module) (1u << ((module) - 1u))

// Independent source classes supporting generated physical facts.
typedef enum loom_xdna_provenance_bit_e {
  LOOM_XDNA_PROVENANCE_AIE_RT = 1u << 0,
  LOOM_XDNA_PROVENANCE_REGISTER_DATABASE = 1u << 1,
  LOOM_XDNA_PROVENANCE_MLIR_AIE = 1u << 2,
  LOOM_XDNA_PROVENANCE_XDNA_DRIVER = 1u << 3,
  LOOM_XDNA_PROVENANCE_HARDWARE = 1u << 4,
} loom_xdna_provenance_bit_t;
typedef uint32_t loom_xdna_provenance_bits_t;

// Direction relative to the programmable stream switch.
typedef enum loom_xdna_stream_direction_e {
  LOOM_XDNA_STREAM_DIRECTION_MASTER = 1,
  LOOM_XDNA_STREAM_DIRECTION_SLAVE = 2,
} loom_xdna_stream_direction_t;

// Architectural stream-switch port class.
typedef enum loom_xdna_stream_port_e {
  LOOM_XDNA_STREAM_PORT_CORE = 1,
  LOOM_XDNA_STREAM_PORT_DMA = 2,
  LOOM_XDNA_STREAM_PORT_TILE_CONTROL = 3,
  LOOM_XDNA_STREAM_PORT_FIFO = 4,
  LOOM_XDNA_STREAM_PORT_SOUTH = 5,
  LOOM_XDNA_STREAM_PORT_WEST = 6,
  LOOM_XDNA_STREAM_PORT_NORTH = 7,
  LOOM_XDNA_STREAM_PORT_EAST = 8,
  LOOM_XDNA_STREAM_PORT_TRACE = 9,
} loom_xdna_stream_port_t;

// Tile-local memory interpretation used by placement resolution.
typedef enum loom_xdna_memory_space_e {
  LOOM_XDNA_MEMORY_SPACE_PROGRAM = 1,
  LOOM_XDNA_MEMORY_SPACE_DATA = 2,
} loom_xdna_memory_space_t;

// Physical array coordinate.
typedef struct loom_xdna_tile_coordinate_t {
  // Physical column ordinal.
  uint16_t column;
  // Physical row ordinal.
  uint16_t row;
} loom_xdna_tile_coordinate_t;

// One tile-relative load aperture mapped to canonical owner storage.
typedef struct loom_xdna_address_window_t {
  // Stable diagnostic window name.
  const char* name;
  // First address in the tile-relative load aperture.
  uint32_t base;
  // Addressable bytes in the aperture.
  uint32_t capacity;
  // First accessor-relative lock selector for the owner memory module.
  uint16_t lock_selector_base;
  // Signed column displacement from accessor to owner.
  int8_t owner_column_delta;
  // Signed row displacement from accessor to owner.
  int8_t owner_row_delta;
  // Required physical kind of the owner tile.
  loom_xdna_tile_kind_t owner_kind;
} loom_xdna_address_window_t;

// Allocation geometry and load apertures for one tile kind.
typedef struct loom_xdna_tile_memory_facts_t {
  // First address in the tile's local allocation space.
  uint32_t local_base;
  // Addressable bytes in the tile's local allocation space.
  uint32_t local_capacity;
  // Core startup address in program memory.
  uint32_t program_base;
  // Addressable program-memory bytes.
  uint32_t program_capacity;
  // First load-window row in the owning array family.
  uint16_t window_start;
  // Number of load-window rows.
  uint8_t window_count;
  // Number of independently addressable local-memory banks.
  uint8_t bank_count;
} loom_xdna_tile_memory_facts_t;

// DMA feature bits shared by each tile engine.
typedef enum loom_xdna_dma_feature_bit_e {
  LOOM_XDNA_DMA_FEATURE_COMPRESSION = 1u << 0,
  LOOM_XDNA_DMA_FEATURE_PADDING = 1u << 1,
  LOOM_XDNA_DMA_FEATURE_OUT_OF_ORDER = 1u << 2,
  LOOM_XDNA_DMA_FEATURE_TOKENS = 1u << 3,
  LOOM_XDNA_DMA_FEATURE_REPEAT = 1u << 4,
  LOOM_XDNA_DMA_FEATURE_TLAST_SUPPRESSION = 1u << 5,
} loom_xdna_dma_feature_bit_t;
typedef uint8_t loom_xdna_dma_feature_bits_t;

// Resource and field-width limits for one tile DMA engine.
typedef struct loom_xdna_dma_facts_t {
  // Exclusive maximum byte address accepted by the DMA engine.
  uint64_t address_maximum;
  // Number of buffer descriptors.
  uint16_t buffer_descriptor_count;
  // Number of channels in each transfer direction.
  uint8_t channel_count_per_direction;
  // Number of address dimensions.
  uint8_t address_dimension_count;
  // Required byte-address alignment.
  uint8_t address_alignment;
  // Right shift applied to byte addresses before register-field encoding.
  uint8_t address_encoding_shift;
  // Number of bytes represented by one encoded transfer-length unit.
  uint8_t transfer_length_granularity;
  // Value subtracted after scaling a transfer length to encoded units.
  uint8_t transfer_length_offset;
  // Stream-switch port channel selected by memory-to-stream DMA channel zero.
  uint8_t memory_to_stream_port_base;
  // Stream-switch port-channel stride between memory-to-stream DMA channels.
  uint8_t memory_to_stream_port_stride;
  // Stream-switch port channel selected by stream-to-memory DMA channel zero.
  uint8_t stream_to_memory_port_base;
  // Stream-switch port-channel stride between stream-to-memory DMA channels.
  uint8_t stream_to_memory_port_stride;
  // Encoded step-size field width.
  uint8_t step_size_bits;
  // Encoded wrap field width.
  uint8_t wrap_bits;
  // Encoded iteration field width.
  uint8_t iteration_bits;
  // Maximum queued start tasks per channel.
  uint8_t task_queue_depth;
  // Supported DMA behavior bits.
  loom_xdna_dma_feature_bits_t feature_bits;
} loom_xdna_dma_facts_t;

// All physical resources shared by tiles of one kind.
typedef struct loom_xdna_tile_facts_t {
  // Physical tile role.
  loom_xdna_tile_kind_t kind;
  // First physical row with this role.
  uint8_t first_row;
  // Number of consecutive physical rows with this role.
  uint8_t row_count;
  // Number of hardware locks.
  uint8_t lock_count;
  // Minimum signed lock value.
  int8_t lock_value_minimum;
  // Maximum signed lock value.
  int8_t lock_value_maximum;
  // Bit set of configuration-register modules present on the tile.
  uint8_t register_module_bits;
  // Local and program memory geometry.
  loom_xdna_tile_memory_facts_t memory;
  // DMA engine resources and limits.
  loom_xdna_dma_facts_t dma;
} loom_xdna_tile_facts_t;

// Event identifier domain implemented by one register module.
typedef struct loom_xdna_event_module_facts_t {
  // Register module producing or consuming the events.
  loom_xdna_register_module_t module;
  // Complete event-ID cardinality.
  uint16_t event_count;
  // Number of event IDs with a stable semantic name.
  uint16_t named_event_count;
} loom_xdna_event_module_facts_t;

// Contiguous channel range in one switch direction's ordinal space.
typedef struct loom_xdna_stream_port_range_t {
  // Tile role implementing the range.
  loom_xdna_tile_kind_t tile_kind;
  // Stream-switch direction.
  loom_xdna_stream_direction_t direction;
  // Architectural port class.
  loom_xdna_stream_port_t port;
  // First programmable switch ordinal.
  uint8_t ordinal;
  // Number of port channels.
  uint8_t count;
} loom_xdna_stream_port_range_t;

// Complete materialized physical-array facts shared by device profiles.
typedef struct loom_xdna_array_family_t {
  // Stable semantic family key.
  const char* key;
  // Incompatible family-table revision.
  uint32_t revision;
  // Selected instruction-set identity.
  loom_xdna_architecture_t architecture;
  // Physical column count.
  uint16_t column_count;
  // Physical row count.
  uint16_t row_count;
  // Column shift in absolute register addresses.
  uint8_t column_shift;
  // Row shift in absolute register addresses.
  uint8_t row_shift;
  // Native address-generation granularity in bits.
  uint8_t address_generation_granularity_bits;
  // Independent sources supporting the family.
  loom_xdna_provenance_bits_t provenance_bits;
  // Column-wise tile controller packet identities indexed by physical row.
  const uint8_t* controller_ids;
  // Flattened load-aperture rows.
  const loom_xdna_address_window_t* address_windows;
  // Number of load-aperture rows.
  uint16_t address_window_count;
  // Tile-role rows.
  const loom_xdna_tile_facts_t* tiles;
  // Number of tile-role rows.
  uint8_t tile_count;
  // Event-module rows.
  const loom_xdna_event_module_facts_t* events;
  // Number of event-module rows.
  uint8_t event_count;
  // Stream-port range rows.
  const loom_xdna_stream_port_range_t* stream_ports;
  // Number of stream-port range rows.
  uint8_t stream_port_count;
  // Pinned aie-rt source revision.
  const char* aie_rt_source_commit;
  // Pinned MLIR-AIE target-model source revision.
  const char* mlir_aie_source_commit;
  // Pinned independent register-database revision.
  const char* register_database_version;
} loom_xdna_array_family_t;

// Canonical physical storage selected by a local or load-aperture range.
typedef struct loom_xdna_memory_placement_t {
  // Physical tile owning the storage.
  loom_xdna_tile_coordinate_t owner;
  // Byte offset in the owner's local storage.
  uint32_t owner_offset;
  // Number of bytes in the resolved placement.
  uint32_t byte_length;
  // Bytes available from |owner_offset| through the resolved aperture.
  uint32_t available_capacity;
} loom_xdna_memory_placement_t;

// Returns the complete immutable NPU2 physical-array family.
const loom_xdna_array_family_t* loom_xdna_npu2_array_family(void);

// Resolves the controller packet identity used by one physical tile.
iree_status_t loom_xdna_array_controller_id(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate, uint8_t* out_controller_id);

// Resolves the tile facts at one physical coordinate.
iree_status_t loom_xdna_array_tile_facts(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate,
    const loom_xdna_tile_facts_t** out_facts);

// Resolves one architectural stream port to its programmable ordinal range.
iree_status_t loom_xdna_array_stream_port_range(
    const loom_xdna_array_family_t* family, loom_xdna_tile_kind_t tile_kind,
    loom_xdna_stream_direction_t direction, loom_xdna_stream_port_t port,
    const loom_xdna_stream_port_range_t** out_range);

// Forms one absolute tile register address after validating the module.
iree_status_t loom_xdna_array_register_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate, loom_xdna_register_module_t module,
    uint32_t register_offset, uint64_t* out_address);

// Resolves one direct tile-local allocation range to canonical storage.
iree_status_t loom_xdna_array_resolve_local_memory(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t coordinate, uint32_t address,
    uint32_t byte_length, loom_xdna_memory_placement_t* out_placement);

// Resolves one program or data load range to canonical physical storage.
//
// Data addresses are interpreted through the accessor tile's load apertures.
// Neighbor windows resolve to the same owner and owner-relative offset as a
// direct placement through the owner's self window.
iree_status_t loom_xdna_array_resolve_load_memory(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t accessor, loom_xdna_memory_space_t memory_space,
    uint32_t address, uint32_t byte_length,
    loom_xdna_memory_placement_t* out_placement);

// Forms a tile-relative load address for canonical physical storage.
//
// |owner_offset| is relative to the owner's program or local-data allocation,
// matching loom_xdna_memory_placement_t. The owner must be visible through one
// of the accessor's architectural load windows.
iree_status_t loom_xdna_array_form_load_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t accessor, loom_xdna_memory_space_t memory_space,
    loom_xdna_tile_coordinate_t owner, uint32_t owner_offset,
    uint32_t byte_length, uint32_t* out_address);

// Forms an accessor-relative lock selector for a physical owner lock.
//
// The owner lock ordinal is relative to its physical tile. The returned
// selector is relative to the accessor's architectural lock namespace and can
// be encoded directly in an accessor core instruction.
iree_status_t loom_xdna_array_form_lock_selector(
    const loom_xdna_array_family_t* family,
    loom_xdna_tile_coordinate_t accessor, loom_xdna_tile_coordinate_t owner,
    uint16_t owner_lock_ordinal, uint16_t* out_selector);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_FACTS_H_
