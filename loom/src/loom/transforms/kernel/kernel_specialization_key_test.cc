// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/kernel_specialization_key.h"

#include <array>
#include <bit>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"

namespace loom {
namespace {

class KeyArena {
 public:
  KeyArena() {
    iree_arena_block_pool_initialize(4096, {this, Allocate}, &pool_);
    iree_arena_initialize(&pool_, &arena);
  }
  ~KeyArena() {
    iree_arena_deinitialize(&arena);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  // Query or retained-key storage, independent of the caller module.
  iree_arena_allocator_t arena = {};
  // Number of allocation attempts, excluding trim/free calls.
  size_t allocations = 0;
  // Exact attempt to fail, or SIZE_MAX to allow every allocation.
  size_t failure = SIZE_MAX;

 private:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* params, void** pointer) {
    auto* owner = static_cast<KeyArena*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        command != IREE_ALLOCATOR_COMMAND_TRIM &&
        owner->allocations++ == owner->failure) {
      return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, params, pointer);
  }

  // Owns all storage until both queries and keys have retired.
  iree_arena_block_pool_t pool_ = {};
};

class LiteralModule {
 public:
  LiteralModule() {
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_CHECK_OK(loom_context_finalize(&context_));
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    IREE_CHECK_OK(loom_module_allocate(&context_, IREE_SV("literal"), &pool_,
                                       nullptr, iree_allocator_system(),
                                       &module));
  }
  ~LiteralModule() {
    loom_module_free(module);
    iree_arena_block_pool_deinitialize(&pool_);
    loom_context_deinitialize(&context_);
  }

  loom_string_id_t String(const char* text) {
    loom_string_id_t id;
    IREE_CHECK_OK(
        loom_module_intern_string(module, iree_make_cstring_view(text), &id));
    return id;
  }

  uint16_t Encoding(const char* name, const loom_named_attr_t* attributes,
                    uint8_t count, const char* alias = nullptr) {
    const loom_encoding_t encoding = {
        .name_id = String(name),
        .alias_id = alias ? String(alias) : LOOM_STRING_ID_INVALID,
        .attribute_count = count,
        .attributes = attributes,
    };
    uint16_t id;
    IREE_CHECK_OK(loom_module_add_encoding(module, &encoding, &id));
    return id;
  }

  // A repeated encoding DAG has exponentially many paths but linear nodes.
  uint16_t EncodingDag(unsigned depth, int64_t leaf) {
    const loom_named_attr_t payload = {
        .name_id = String("value"),
        .value = loom_attr_i64(leaf),
    };
    uint16_t id = Encoding("test.leaf", &payload, 1);
    for (unsigned i = 0; i < depth; ++i) {
      const loom_named_attr_t children[] = {
          {.name_id = String("left"), .value = loom_attr_encoding(id)},
          {.name_id = String("right"), .value = loom_attr_encoding(id)},
      };
      id = Encoding("test.pair", children, 2);
    }
    return id;
  }

  // Canonical module-owned strings, types, and encoding payloads.
  loom_module_t* module = nullptr;

 private:
  // A permissive context admits ordinary structured encoding payloads.
  loom_context_t context_ = {};
  // Backing is destroyed with the module to detect borrowed retained keys.
  iree_arena_block_pool_t pool_ = {};
};

loom_kernel_specialization_query_t* Query(
    const LiteralModule& module,
    const loom_kernel_specialization_value_t* values, uint16_t count,
    KeyArena& scratch) {
  loom_kernel_specialization_query_t* query = nullptr;
  IREE_CHECK_OK(loom_kernel_specialization_query_allocate(
      module.module, values, count, &scratch.arena, &query));
  return query;
}

loom_kernel_specialization_key_t Capture(
    const loom_kernel_specialization_query_t* query, KeyArena& owner) {
  loom_kernel_specialization_key_t key = {};
  IREE_CHECK_OK(loom_kernel_specialization_key_capture(query, SIZE_MAX,
                                                       &owner.arena, &key));
  return key;
}

loom_kernel_specialization_value_t Index(int64_t value) {
  return {
      .type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
      .facts = loom_value_facts_exact_i64(value),
  };
}

loom_kernel_specialization_value_t Literal(loom_type_t type,
                                           loom_op_kind_t kind,
                                           const loom_attribute_t* attributes,
                                           uint8_t count) {
  return {
      .type = type,
      .kind = kind,
      .attribute_count = count,
      .attributes = attributes,
  };
}

TEST(KernelSpecializationKeyTest, HashCollisionsRequireExactTupleComparison) {
  LiteralModule module;
  KeyArena scratch, owner;
  const loom_kernel_specialization_value_t values[] = {
      Index(1724915559227727183),
      Index(3844235512610326073),
  };
  auto* first = Query(module, &values[0], 1, scratch);
  auto* second = Query(module, &values[1], 1, scratch);
  EXPECT_EQ(loom_kernel_specialization_query_hash(first),
            loom_kernel_specialization_query_hash(second));
  const auto key = Capture(first, owner);
  EXPECT_TRUE(loom_kernel_specialization_key_matches(first, key));
  EXPECT_FALSE(loom_kernel_specialization_key_matches(second, key));
  auto changed_type = values[0];
  changed_type.type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &changed_type, 1, scratch), key));
  const auto tuple_key = Capture(Query(module, values, 2, scratch), owner);
  const loom_kernel_specialization_value_t reversed[] = {values[1], values[0]};
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, reversed, 2, scratch), tuple_key));
}

TEST(KernelSpecializationKeyTest, LiteralBitsShapesAndOperationKindsAreExact) {
  LiteralModule module;
  KeyArena scratch, owner;
  const uint64_t bits[] = {UINT64_C(0), UINT64_C(0x8000000000000000),
                           UINT64_C(0x7ff8000000000001),
                           UINT64_C(0x7ff8000000000002)};
  std::array<loom_kernel_specialization_key_t, 4> keys;
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto attr = loom_attr_f64(std::bit_cast<double>(bits[i]));
    const auto value = Literal(loom_type_scalar(LOOM_SCALAR_TYPE_F64),
                               LOOM_OP_SCALAR_CONSTANT, &attr, 1);
    // Capture while payload is live; subsequent comparisons use new storage.
    keys[i] = Capture(Query(module, &value, 1, scratch), owner);
  }
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto attr = loom_attr_f64(std::bit_cast<double>(bits[i]));
    const auto value = Literal(loom_type_scalar(LOOM_SCALAR_TYPE_F64),
                               LOOM_OP_SCALAR_CONSTANT, &attr, 1);
    auto* query = Query(module, &value, 1, scratch);
    for (size_t j = 0; j < keys.size(); ++j) {
      EXPECT_EQ(loom_kernel_specialization_key_matches(query, keys[j]), i == j);
    }
  }
  int64_t lanes[] = {1, 2, 3, 4};
  const auto attr = loom_attr_i64_array(lanes, 4);
  auto value =
      Literal(loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                  loom_dim_pack_static(4), 0),
              LOOM_OP_VECTOR_CONSTANTI, &attr, 1);
  const auto key = Capture(Query(module, &value, 1, scratch), owner);
  value.type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(2), loom_dim_pack_static(2), 0);
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
  const auto zero = loom_attr_i64(0);
  value = Literal(loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32,
                                      loom_dim_pack_static(4), 0),
                  LOOM_OP_VECTOR_SPLATI, &zero, 1);
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
}

TEST(KernelSpecializationKeyTest, OrderedFloatingLanesPreserveCarrierBits) {
  LiteralModule module;
  KeyArena scratch, owner;
  double lanes[] = {1.0, -0.0,
                    std::bit_cast<double>(UINT64_C(0x7ff8000000000001)), 4.0};
  const auto attr = loom_attr_f64_array(lanes, 4);
  const auto value =
      Literal(loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F64,
                                  loom_dim_pack_static(4), 0),
              LOOM_OP_VECTOR_CONSTANTF, &attr, 1);
  const auto key = Capture(Query(module, &value, 1, scratch), owner);
  EXPECT_TRUE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
  lanes[1] = 0.0;
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
  lanes[1] = -0.0;
  lanes[2] = std::bit_cast<double>(UINT64_C(0x7ff8000000000002));
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
}

TEST(KernelSpecializationKeyTest,
     NullSymbolAndStringPayloadsAreModuleIndependent) {
  KeyArena owner;
  loom_kernel_specialization_key_t key = {};
  for (unsigned round = 0; round < 2; ++round) {
    LiteralModule module;
    if (round) {
      module.String("local name shifts every subsequent identity");
    }
    const loom_named_attr_t metadata[] = {
        {
            .name_id = module.String("name"),
            .value = loom_attr_string(module.String("payload")),
        },
        {
            .name_id = module.String("optional"),
            .value = loom_attr_symbol(loom_symbol_ref_null()),
        },
    };
    const auto id =
        module.Encoding("test.metadata", metadata, IREE_ARRAYSIZE(metadata),
                        round ? "different_alias" : "original_alias");
    const loom_attribute_t attrs[] = {loom_attr_encoding(id),
                                      loom_attr_absent()};
    const auto value =
        Literal(loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA),
                LOOM_OP_ENCODING_DEFINE, attrs, 2);
    KeyArena scratch;
    auto* query = Query(module, &value, 1, scratch);
    if (!round) {
      key = Capture(query, owner);
    } else {
      EXPECT_TRUE(loom_kernel_specialization_key_matches(query, key));
    }
  }
}

TEST(KernelSpecializationKeyTest, BytePayloadIsBorrowedUntilBoundedCapture) {
  LiteralModule module;
  KeyArena scratch, owner;
  loom_kernel_specialization_key_t key = {};
  {
    std::string bytes(1024 * 1024, 'x');
    const loom_attribute_t attrs[] = {
        loom_attr_bytes(reinterpret_cast<const uint8_t*>(bytes.data()),
                        bytes.size()),
        loom_attr_i64(16),
    };
    const auto value =
        Literal(loom_type_buffer(), LOOM_OP_BUFFER_CONSTANT, attrs, 2);
    auto* query = Query(module, &value, 1, scratch);
    EXPECT_LT(scratch.arena.used_allocation_size, 4096u);
    IREE_ASSERT_OK(loom_kernel_specialization_key_capture(
        query, loom_kernel_specialization_query_size(query) - 1, &owner.arena,
        &key));
    EXPECT_EQ(key.data, nullptr);
    EXPECT_EQ(owner.allocations, 0u);
    key = Capture(query, owner);
  }
  std::string bytes(1024 * 1024, 'x');
  loom_attribute_t attrs[] = {
      loom_attr_bytes(reinterpret_cast<const uint8_t*>(bytes.data()),
                      bytes.size()),
      loom_attr_i64(16),
  };
  const auto value =
      Literal(loom_type_buffer(), LOOM_OP_BUFFER_CONSTANT, attrs, 2);
  auto* query = Query(module, &value, 1, scratch);
  const auto allocations = scratch.allocations + owner.allocations;
  EXPECT_TRUE(loom_kernel_specialization_key_matches(query, key));
  bytes.back() = 'y';
  EXPECT_FALSE(loom_kernel_specialization_key_matches(query, key));
  bytes.back() = 'x';
  attrs[1] = loom_attr_i64(32);
  EXPECT_FALSE(loom_kernel_specialization_key_matches(
      Query(module, &value, 1, scratch), key));
  EXPECT_EQ(scratch.allocations + owner.allocations, allocations);
}

TEST(KernelSpecializationKeyTest, EncodingIdentityOutlivesModulesAndLocalIds) {
  KeyArena owner;
  loom_kernel_specialization_key_t key = {};
  {
    LiteralModule source;
    const auto id = source.EncodingDag(128, 4);
    const loom_attribute_t attrs[] = {loom_attr_encoding(id),
                                      loom_attr_absent()};
    const auto value =
        Literal(loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA),
                LOOM_OP_ENCODING_DEFINE, attrs, 2);
    KeyArena scratch;
    key = Capture(Query(source, &value, 1, scratch), owner);
  }
  for (int64_t leaf : {4, 8}) {
    LiteralModule source;
    source.String("unrelated string shifting local identities");
    source.Encoding("test.unrelated", nullptr, 0);
    const auto id = source.EncodingDag(128, leaf);
    const loom_attribute_t attrs[] = {loom_attr_encoding(id),
                                      loom_attr_absent()};
    const auto value =
        Literal(loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA),
                LOOM_OP_ENCODING_DEFINE, attrs, 2);
    KeyArena scratch;
    auto* query = Query(source, &value, 1, scratch);
    EXPECT_EQ(loom_kernel_specialization_key_matches(query, key), leaf == 4);
    EXPECT_LT(loom_kernel_specialization_query_size(query), 128u * 256u);
    EXPECT_LT(scratch.arena.used_allocation_size, 128u * 2048u);
  }
}

TEST(KernelSpecializationKeyTest,
     AllocationFailuresLeaveQueriesAndKeysRetryable) {
  LiteralModule source;
  const auto id = source.EncodingDag(128, 4);
  const loom_attribute_t attrs[] = {loom_attr_encoding(id), loom_attr_absent()};
  const auto value =
      Literal(loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA),
              LOOM_OP_ENCODING_DEFINE, attrs, 2);
  for (bool capture : {false, true}) {
    for (size_t failure = 0;; ++failure) {
      KeyArena scratch, owner;
      (capture ? owner : scratch).failure = failure;
      loom_kernel_specialization_query_t* query = nullptr;
      auto status = loom_kernel_specialization_query_allocate(
          source.module, &value, 1, &scratch.arena, &query);
      loom_kernel_specialization_key_t key = {};
      if (iree_status_is_ok(status)) {
        status = loom_kernel_specialization_key_capture(query, SIZE_MAX,
                                                        &owner.arena, &key);
      } else {
        EXPECT_EQ(query, nullptr);
      }
      const bool success = iree_status_is_ok(status);
      if (!success) {
        EXPECT_TRUE(iree_status_is_resource_exhausted(status));
        EXPECT_EQ(key.data, nullptr);
      }
      iree_status_ignore(status);
      scratch.failure = owner.failure = SIZE_MAX;
      query = Query(source, &value, 1, scratch);
      key = Capture(query, owner);
      EXPECT_TRUE(loom_kernel_specialization_key_matches(query, key));
      if (success) {
        break;
      }
    }
  }
}

}  // namespace
}  // namespace loom
