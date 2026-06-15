// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_LLVMIR_TYPES_H_
#define LOOM_TARGET_LLVMIR_TYPES_H_

#include "iree/base/internal/arena.h"
#include "loom/target/emit/llvmir/builder.h"
#include "loom/target/emit/llvmir/module.h"

typedef struct loom_llvmir_attr_list_t {
  // Attribute storage owned by the containing module arena.
  const loom_llvmir_attr_t* attrs;
  // Number of entries in |attrs|.
  iree_host_size_t attr_count;
} loom_llvmir_attr_list_t;

typedef struct loom_llvmir_type_t {
  // Type kind selecting the active payload fields.
  loom_llvmir_type_kind_t kind;
  // Integer bit width for INTEGER types.
  uint32_t bit_width;
  // Pointer address space for POINTER types.
  uint32_t address_space;
  // Element count for VECTOR and STRUCT types.
  uint32_t element_count;
  // Vector element type for VECTOR types.
  loom_llvmir_type_id_t element_type;
  // Struct element types for STRUCT types.
  loom_llvmir_type_id_t* element_types;
  // Floating-point scalar kind for FLOAT types.
  loom_llvmir_float_kind_t float_kind;
} loom_llvmir_type_t;

typedef struct loom_llvmir_value_t {
  // Value kind selecting the active payload fields.
  loom_llvmir_value_kind_t kind;
  // SSA/result type for the value.
  loom_llvmir_type_id_t type_id;
  // Optional printed/debug name without the leading percent sign.
  iree_string_view_t name;
  // Kind-specific payload.
  union {
    // Integer constant payload.
    uint64_t integer_value;
    // Floating-point constant bit pattern payload.
    uint64_t float_bits;
    // Integer vector constant payload.
    struct {
      // Element values in lane order.
      uint64_t* values;
      // Number of entries in |values|.
      iree_host_size_t value_count;
    } integer_vector;
    // Parameter value location.
    struct {
      // Owning function.
      loom_llvmir_function_id_t function_id;
      // Ordinal within the owning function's parameter list.
      uint32_t parameter_ordinal;
    } parameter;
    // Global value location.
    struct {
      // Owning global.
      loom_llvmir_global_id_t global_id;
    } global;
    // Instruction result value location.
    struct {
      // Owning function.
      loom_llvmir_function_id_t function_id;
      // Owning block.
      loom_llvmir_block_id_t block_id;
      // Ordinal within the owning block's instruction list.
      uint32_t instruction_ordinal;
    } instruction;
  };
} loom_llvmir_value_t;

typedef struct loom_llvmir_parameter_t {
  // Value table entry representing this parameter.
  loom_llvmir_value_id_t value_id;
  // Parameter type.
  loom_llvmir_type_id_t type_id;
  // Optional printed/debug name without the leading percent sign.
  iree_string_view_t name;
  // Structured parameter attributes.
  loom_llvmir_attr_list_t attrs;
} loom_llvmir_parameter_t;

typedef enum loom_llvmir_inst_kind_e {
  LOOM_LLVMIR_INST_PHI = 0,
  LOOM_LLVMIR_INST_BINOP = 1,
  LOOM_LLVMIR_INST_ICMP = 2,
  LOOM_LLVMIR_INST_FCMP = 3,
  LOOM_LLVMIR_INST_SELECT = 4,
  LOOM_LLVMIR_INST_CAST = 5,
  LOOM_LLVMIR_INST_GEP = 6,
  LOOM_LLVMIR_INST_ALLOCA = 7,
  LOOM_LLVMIR_INST_LOAD = 8,
  LOOM_LLVMIR_INST_STORE = 9,
  LOOM_LLVMIR_INST_EXTRACT_ELEMENT = 10,
  LOOM_LLVMIR_INST_INSERT_ELEMENT = 11,
  LOOM_LLVMIR_INST_SHUFFLE_VECTOR = 12,
  LOOM_LLVMIR_INST_CALL = 13,
  LOOM_LLVMIR_INST_INLINE_ASM = 14,
  LOOM_LLVMIR_INST_RET = 15,
  LOOM_LLVMIR_INST_BR = 16,
  LOOM_LLVMIR_INST_COND_BR = 17,
  LOOM_LLVMIR_INST_UNREACHABLE = 18,
  LOOM_LLVMIR_INST_UNOP = 19,
  LOOM_LLVMIR_INST_ATOMIC_RMW = 20,
  LOOM_LLVMIR_INST_CMPXCHG = 21,
  LOOM_LLVMIR_INST_EXTRACT_VALUE = 22,
} loom_llvmir_inst_kind_t;

typedef struct loom_llvmir_metadata_attachment_storage_t {
  // Attachment name without the leading exclamation mark.
  iree_string_view_t name;
  // Referenced metadata node id.
  loom_llvmir_metadata_id_t metadata_id;
} loom_llvmir_metadata_attachment_storage_t;

typedef struct loom_llvmir_instruction_t {
  // Instruction kind selecting the active payload.
  loom_llvmir_inst_kind_t kind;
  // Result value id, or INVALID for void/terminator instructions.
  loom_llvmir_value_id_t result_value_id;
  // Kind-specific instruction payload.
  union {
    // Phi node incoming edges.
    struct {
      // Incoming block/value pairs owned by the module arena.
      loom_llvmir_phi_incoming_t* incoming;
      // Number of entries in |incoming|.
      iree_host_size_t incoming_count;
    } phi;
    // Binary operation operands.
    struct {
      // Binary operation opcode.
      loom_llvmir_binop_t op;
      // Left operand.
      loom_llvmir_value_id_t lhs;
      // Right operand.
      loom_llvmir_value_id_t rhs;
      // Integer arithmetic flags such as no-wrap and exact.
      loom_llvmir_integer_arithmetic_flags_t integer_flags;
      // Fast-math flags for floating-point binary operations.
      loom_llvmir_fast_math_flags_t fast_math_flags;
    } binop;
    // Unary operation operand.
    struct {
      // Unary operation opcode.
      loom_llvmir_unop_t op;
      // Operand.
      loom_llvmir_value_id_t value;
      // Fast-math flags for floating-point unary operations.
      loom_llvmir_fast_math_flags_t fast_math_flags;
    } unop;
    // Integer comparison operands.
    struct {
      // Integer comparison predicate.
      loom_llvmir_icmp_predicate_t predicate;
      // Left operand.
      loom_llvmir_value_id_t lhs;
      // Right operand.
      loom_llvmir_value_id_t rhs;
    } icmp;
    // Floating-point comparison operands.
    struct {
      // Floating-point comparison predicate.
      loom_llvmir_fcmp_predicate_t predicate;
      // Left operand.
      loom_llvmir_value_id_t lhs;
      // Right operand.
      loom_llvmir_value_id_t rhs;
    } fcmp;
    // Select operands.
    struct {
      // Scalar or vector mask condition.
      loom_llvmir_value_id_t condition;
      // Value selected when |condition| is true.
      loom_llvmir_value_id_t true_value;
      // Value selected when |condition| is false.
      loom_llvmir_value_id_t false_value;
    } select;
    // Cast operands.
    struct {
      // Cast opcode.
      loom_llvmir_cast_op_t op;
      // Source value.
      loom_llvmir_value_id_t value;
    } cast;
    // GetElementPtr operands.
    struct {
      // Pointee element type used by LLVM's opaque-pointer GEP syntax.
      loom_llvmir_type_id_t element_type;
      // Base pointer value.
      loom_llvmir_value_id_t base;
      // Index values owned by the module arena.
      loom_llvmir_value_id_t* indices;
      // Number of entries in |indices|.
      iree_host_size_t index_count;
    } gep;
    // Stack allocation operands and memory attributes.
    struct {
      // Allocated element type.
      loom_llvmir_type_id_t element_type;
      // Element count value.
      loom_llvmir_value_id_t count;
      // True when the source IR supplied an explicit count operand.
      bool has_explicit_count;
      // Optional byte alignment.
      uint32_t alignment;
    } alloca;
    // Load operands and memory attributes.
    struct {
      // Loaded result type.
      loom_llvmir_type_id_t result_type;
      // Pointer operand.
      loom_llvmir_value_id_t pointer;
      // Optional byte alignment.
      uint32_t alignment;
      // Memory operation flags.
      uint32_t flags;
      // Metadata attachments owned by the module arena.
      loom_llvmir_metadata_attachment_storage_t* metadata_attachments;
      // Number of entries in |metadata_attachments|.
      iree_host_size_t metadata_attachment_count;
    } load;
    // Store operands and memory attributes.
    struct {
      // Stored value.
      loom_llvmir_value_id_t value;
      // Pointer operand.
      loom_llvmir_value_id_t pointer;
      // Optional byte alignment.
      uint32_t alignment;
      // Memory operation flags.
      uint32_t flags;
      // Metadata attachments owned by the module arena.
      loom_llvmir_metadata_attachment_storage_t* metadata_attachments;
      // Number of entries in |metadata_attachments|.
      iree_host_size_t metadata_attachment_count;
    } store;
    // Atomic read-modify-write operands and memory attributes.
    struct {
      // Atomic operation result type.
      loom_llvmir_type_id_t result_type;
      // Read-modify-write opcode.
      loom_llvmir_atomic_rmw_op_t op;
      // Pointer operand.
      loom_llvmir_value_id_t pointer;
      // Update value operand.
      loom_llvmir_value_id_t value;
      // Atomic memory ordering.
      loom_llvmir_atomic_ordering_t ordering;
      // Optional LLVM syncscope name.
      iree_string_view_t sync_scope;
      // Optional byte alignment.
      uint32_t alignment;
      // Memory operation flags.
      uint32_t flags;
    } atomic_rmw;
    // Atomic compare-exchange operands and memory attributes.
    struct {
      // Aggregate result type containing the old value and success flag.
      loom_llvmir_type_id_t result_type;
      // Scalar compared and replaced by the operation.
      loom_llvmir_type_id_t value_type;
      // Pointer operand.
      loom_llvmir_value_id_t pointer;
      // Expected value operand.
      loom_llvmir_value_id_t expected;
      // Replacement value operand.
      loom_llvmir_value_id_t replacement;
      // Success atomic memory ordering.
      loom_llvmir_atomic_ordering_t success_ordering;
      // Failure atomic memory ordering.
      loom_llvmir_atomic_ordering_t failure_ordering;
      // Optional LLVM syncscope name.
      iree_string_view_t sync_scope;
      // Optional byte alignment.
      uint32_t alignment;
      // Memory operation flags.
      uint32_t flags;
      // True when the compare-exchange may fail spuriously.
      bool is_weak;
    } cmpxchg;
    // Aggregate value extraction operands.
    struct {
      // Aggregate value to extract from.
      loom_llvmir_value_id_t aggregate;
      // Zero-based aggregate element index.
      uint32_t index;
    } extract_value;
    // Vector lane extraction operands.
    struct {
      // Vector value to extract from.
      loom_llvmir_value_id_t vector;
      // Integer lane index.
      loom_llvmir_value_id_t index;
    } extract_element;
    // Vector lane insertion operands.
    struct {
      // Original vector value.
      loom_llvmir_value_id_t vector;
      // Scalar element value to insert.
      loom_llvmir_value_id_t element;
      // Integer lane index.
      loom_llvmir_value_id_t index;
    } insert_element;
    // Vector shuffle operands.
    struct {
      // Left vector input.
      loom_llvmir_value_id_t lhs;
      // Right vector input.
      loom_llvmir_value_id_t rhs;
      // Constant vector mask.
      loom_llvmir_value_id_t mask;
    } shuffle_vector;
    // Function call operands.
    struct {
      // Callee function id.
      loom_llvmir_function_id_t callee;
      // Argument values owned by the module arena.
      loom_llvmir_value_id_t* args;
      // Number of entries in |args|.
      iree_host_size_t arg_count;
      // Attributes attached to the call result.
      loom_llvmir_attr_list_t result_attrs;
    } call;
    // Inline asm call operands.
    struct {
      // Inline asm result type.
      loom_llvmir_type_id_t result_type;
      // Inline asm flags.
      loom_llvmir_inline_asm_flags_t flags;
      // Asm template string owned by the module arena.
      iree_string_view_t asm_template;
      // Constraint string owned by the module arena.
      iree_string_view_t constraints;
      // Argument values owned by the module arena.
      loom_llvmir_value_id_t* args;
      // Number of entries in |args|.
      iree_host_size_t arg_count;
    } inline_asm;
    // Return terminator payload.
    struct {
      // True when returning a value instead of void.
      bool has_value;
      // Returned value when |has_value| is true.
      loom_llvmir_value_id_t value;
    } ret;
    // Unconditional branch terminator payload.
    struct {
      // Target block id.
      loom_llvmir_block_id_t target;
    } br;
    // Conditional branch terminator payload.
    struct {
      // Branch condition value.
      loom_llvmir_value_id_t condition;
      // Target block id when |condition| is true.
      loom_llvmir_block_id_t true_block;
      // Target block id when |condition| is false.
      loom_llvmir_block_id_t false_block;
    } cond_br;
  };
} loom_llvmir_instruction_t;

struct loom_llvmir_block_t {
  // Owning function.
  loom_llvmir_function_t* function;
  // Function-local block id.
  loom_llvmir_block_id_t id;
  // Optional label name without the trailing colon.
  iree_string_view_t name;
  // Instruction storage owned by the module arena.
  loom_llvmir_instruction_t* instructions;
  // Number of entries in |instructions|.
  iree_host_size_t instruction_count;
  // Capacity of |instructions|.
  iree_host_size_t instruction_capacity;
};

typedef struct loom_llvmir_metadata_node_t {
  // Integer tuple storage owned by the module arena.
  int32_t* i32_values;
  // Module constants corresponding to |i32_values| in tuple order.
  loom_llvmir_value_id_t* i32_value_ids;
  // Number of entries in |i32_values|.
  iree_host_size_t i32_value_count;
} loom_llvmir_metadata_node_t;

struct loom_llvmir_global_t {
  // Owning module.
  loom_llvmir_module_t* module;
  // Module-local global id.
  loom_llvmir_global_id_t id;
  // Value table entry representing the global pointer.
  loom_llvmir_value_id_t value_id;
  // Global symbol name without the leading at sign.
  iree_string_view_t name;
  // Linkage/preemption option.
  loom_llvmir_linkage_t linkage;
  // Stored value type.
  loom_llvmir_type_id_t value_type;
  // Pointer address space for references to this global.
  uint32_t address_space;
  // True when the global is immutable LLVM constant storage.
  bool is_constant;
  // Constant initializer value.
  loom_llvmir_value_id_t initializer;
  // Optional byte alignment.
  uint32_t alignment;
};

struct loom_llvmir_function_t {
  // Owning module.
  loom_llvmir_module_t* module;
  // Module-local function id.
  loom_llvmir_function_id_t id;
  // Declaration/definition kind.
  loom_llvmir_function_kind_t kind;
  // Function symbol name without the leading at sign.
  iree_string_view_t name;
  // Return type id.
  loom_llvmir_type_id_t return_type;
  // Linkage/preemption option.
  loom_llvmir_linkage_t linkage;
  // Calling convention option.
  loom_llvmir_calling_convention_t calling_convention;
  // Numbered module attribute group, or INVALID for no group.
  loom_llvmir_attr_group_id_t attr_group_id;
  // Parameter storage owned by the module arena.
  loom_llvmir_parameter_t* parameters;
  // Number of entries in |parameters|.
  iree_host_size_t parameter_count;
  // Capacity of |parameters|.
  iree_host_size_t parameter_capacity;
  // Function metadata attachments owned by the module arena.
  loom_llvmir_metadata_attachment_storage_t* metadata_attachments;
  // Number of entries in |metadata_attachments|.
  iree_host_size_t metadata_attachment_count;
  // Capacity of |metadata_attachments|.
  iree_host_size_t metadata_attachment_capacity;
  // Block pointers owned by the module arena.
  loom_llvmir_block_t** blocks;
  // Number of entries in |blocks|.
  iree_host_size_t block_count;
  // Capacity of |blocks|.
  iree_host_size_t block_capacity;
};

typedef struct loom_llvmir_attr_group_t {
  // Attributes in this numbered group.
  loom_llvmir_attr_list_t attrs;
} loom_llvmir_attr_group_t;

typedef struct loom_llvmir_target_config_storage_t {
  // Optional LLVM source_filename module field.
  iree_string_view_t source_name;
  // Optional LLVM target triple.
  iree_string_view_t target_triple;
  // Optional LLVM data layout.
  iree_string_view_t data_layout;
  // Optional producer string for future metadata/debug output.
  iree_string_view_t producer;
  // Default pointer bit width for target-independent layout decisions.
  uint32_t default_pointer_bitwidth;
  // Index bit width chosen for lowered index values.
  uint32_t index_bitwidth;
  // Offset bit width chosen for lowered byte offsets.
  uint32_t offset_bitwidth;
} loom_llvmir_target_config_storage_t;

struct loom_llvmir_module_t {
  // Host allocator used for the module object and arena blocks.
  iree_allocator_t allocator;
  // Arena block pool backing all module-owned records.
  iree_arena_block_pool_t block_pool;
  // Arena allocator for module-owned child records and strings.
  iree_arena_allocator_t arena;
  // Copied target configuration.
  loom_llvmir_target_config_storage_t target_config;
  // Interned type storage.
  loom_llvmir_type_t* types;
  // Number of entries in |types|.
  iree_host_size_t type_count;
  // Capacity of |types|.
  iree_host_size_t type_capacity;
  // Module-global value storage.
  loom_llvmir_value_t* values;
  // Number of entries in |values|.
  iree_host_size_t value_count;
  // Capacity of |values|.
  iree_host_size_t value_capacity;
  // Numbered attribute groups.
  loom_llvmir_attr_group_t* attr_groups;
  // Number of entries in |attr_groups|.
  iree_host_size_t attr_group_count;
  // Capacity of |attr_groups|.
  iree_host_size_t attr_group_capacity;
  // Numbered metadata nodes.
  loom_llvmir_metadata_node_t* metadata_nodes;
  // Number of entries in |metadata_nodes|.
  iree_host_size_t metadata_node_count;
  // Capacity of |metadata_nodes|.
  iree_host_size_t metadata_node_capacity;
  // Global pointer storage.
  loom_llvmir_global_t** globals;
  // Number of entries in |globals|.
  iree_host_size_t global_count;
  // Capacity of |globals|.
  iree_host_size_t global_capacity;
  // Function pointer storage.
  loom_llvmir_function_t** functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Capacity of |functions|.
  iree_host_size_t function_capacity;
};

#endif  // LOOM_TARGET_LLVMIR_TYPES_H_
