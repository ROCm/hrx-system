// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/bytecode/interpreter_atomic.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>

#include "iree/base/config.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::vm::bytecode::testing {
namespace {

template <typename T>
void StoreBits(uint8_t* address, T bits) {
  std::memcpy(address, &bits, sizeof(bits));
}

template <typename T>
T LoadBits(const uint8_t* address) {
  T bits = 0;
  std::memcpy(&bits, address, sizeof(bits));
  return bits;
}

template <typename T, typename SignedT>
T SignedBits(SignedT value) {
  static_assert(sizeof(T) == sizeof(SignedT));
  T bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

template <typename T>
struct FloatBits;

template <>
struct FloatBits<uint32_t> {
  static uint32_t FromValue(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static constexpr uint32_t kNegativeZero = UINT32_C(0x80000000);
  static constexpr uint32_t kQuietNaN = UINT32_C(0x7FC00000);
  static constexpr uint32_t kSignalingNaN = UINT32_C(0x7F800123);
};

template <>
struct FloatBits<uint64_t> {
  static uint64_t FromValue(double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  }

  static constexpr uint64_t kNegativeZero = UINT64_C(0x8000000000000000);
  static constexpr uint64_t kQuietNaN = UINT64_C(0x7FF8000000000000);
  static constexpr uint64_t kSignalingNaN = UINT64_C(0x7FF0000000000123);
};

template <typename T>
void ExpectApply(uint8_t* address,
                 iree_vm_bytecode_buffer_atomic_carrier_t carrier,
                 iree_vm_bytecode_buffer_atomic_kind_t kind, T initial_bits,
                 T operand_bits, T expected_bits) {
  StoreBits(address, initial_bits);
  EXPECT_EQ(iree_vm_bytecode_atomic_apply(
                address, operand_bits, kind, carrier,
                IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED),
            initial_bits);
  EXPECT_EQ(LoadBits<T>(address), expected_bits);
}

template <typename T, typename SignedT, typename FloatT>
void ExpectAllApplyKinds(uint8_t* address,
                         iree_vm_bytecode_buffer_atomic_carrier_t carrier) {
  using Limits = std::numeric_limits<T>;
  using Bits = FloatBits<T>;
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER, T{9}, T{4},
              T{4});
  ExpectApply(
      address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_FLOAT,
      Bits::FromValue(FloatT{1}), Bits::kSignalingNaN, Bits::kSignalingNaN);
  ExpectApply(address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER,
              Limits::max(), T{2}, T{1});
  ExpectApply(address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_FLOAT,
              Bits::FromValue(FloatT{1.5}), Bits::FromValue(FloatT{2.25}),
              Bits::FromValue(FloatT{3.75}));
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_SUBTRACT_INTEGER, T{1}, T{2},
              Limits::max());
  ExpectApply(address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_AND_INTEGER,
              T{0xA5}, T{0x3C}, T{0x24});
  ExpectApply(address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_OR_INTEGER,
              T{0xA0}, T{0x0F}, T{0xAF});
  ExpectApply(address, carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_XOR_INTEGER,
              T{0xAA}, T{0xFF}, T{0x55});
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_SIGNED,
              SignedBits<T>(std::numeric_limits<SignedT>::max()),
              SignedBits<T>(std::numeric_limits<SignedT>::min()),
              SignedBits<T>(std::numeric_limits<SignedT>::min()));
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_SIGNED,
              SignedBits<T>(std::numeric_limits<SignedT>::min()),
              SignedBits<T>(std::numeric_limits<SignedT>::max()),
              SignedBits<T>(std::numeric_limits<SignedT>::max()));
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_UNSIGNED,
              Limits::max(), T{0}, T{0});
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_UNSIGNED, T{0},
              Limits::max(), Limits::max());
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT, T{0},
              Bits::kNegativeZero, Bits::kNegativeZero);
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT,
              Bits::kNegativeZero, T{0}, T{0});
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINIMUM_FLOAT,
              Bits::kSignalingNaN, Bits::FromValue(FloatT{3}), Bits::kQuietNaN);
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXIMUM_FLOAT,
              Bits::FromValue(FloatT{3}), Bits::kSignalingNaN, Bits::kQuietNaN);
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MINNUM_FLOAT, Bits::kQuietNaN,
              Bits::FromValue(FloatT{3}), Bits::FromValue(FloatT{3}));
  ExpectApply(address, carrier,
              IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_MAXNUM_FLOAT,
              Bits::FromValue(FloatT{3}), Bits::kSignalingNaN,
              Bits::FromValue(FloatT{3}));
}

template <typename T>
void ExpectEveryApplyOrdering(
    uint8_t* address, iree_vm_bytecode_buffer_atomic_carrier_t carrier) {
  for (uint8_t ordering = IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED;
       ordering <= IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST;
       ++ordering) {
    SCOPED_TRACE(static_cast<int>(ordering));
    StoreBits<T>(address, T{7});
    EXPECT_EQ(
        iree_vm_bytecode_atomic_apply(
            address, T{5}, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER,
            carrier, ordering),
        T{7});
    EXPECT_EQ(LoadBits<T>(address), T{12});
  }
}

struct AtomicOrderingPair {
  // Ordering applied when the comparison succeeds.
  iree_vm_bytecode_buffer_atomic_ordering_t success;
  // Ordering applied when the comparison fails.
  iree_vm_bytecode_buffer_atomic_ordering_t failure;
};

constexpr AtomicOrderingPair kAtomicOrderingPairs[] = {
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELEASE,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQ_REL,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQ_REL,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE},
    {IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST,
     IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_SEQ_CST},
};

template <typename T>
void ExpectEveryCompareExchangeOrderingPair(
    uint8_t* address, iree_vm_bytecode_buffer_atomic_carrier_t carrier) {
  for (const AtomicOrderingPair& pair : kAtomicOrderingPairs) {
    SCOPED_TRACE(static_cast<int>(pair.success));
    SCOPED_TRACE(static_cast<int>(pair.failure));
    StoreBits<T>(address, T{7});
    EXPECT_EQ(iree_vm_bytecode_atomic_compare_exchange(
                  address, T{7}, T{9}, carrier, pair.success, pair.failure),
              T{7});
    EXPECT_EQ(LoadBits<T>(address), T{9});
    EXPECT_EQ(iree_vm_bytecode_atomic_compare_exchange(
                  address, T{7}, T{11}, carrier, pair.success, pair.failure),
              T{9});
    EXPECT_EQ(LoadBits<T>(address), T{9});
  }
}

TEST(VMBytecodeInterpreterAtomicTest, AppliesEveryI32KindToRawStorage) {
  alignas(4) std::array<uint8_t, 4> storage = {};
  ExpectAllApplyKinds<uint32_t, int32_t, float>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32);
  ExpectEveryApplyOrdering<uint32_t>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32);
}

TEST(VMBytecodeInterpreterAtomicTest, AppliesEveryI64KindToRawStorage) {
  alignas(8) std::array<uint8_t, 8> storage = {};
  ExpectAllApplyKinds<uint64_t, int64_t, double>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64);
  ExpectEveryApplyOrdering<uint64_t>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64);
}

TEST(VMBytecodeInterpreterAtomicTest,
     CompareExchangeSupportsEveryLegalOrderingPair) {
  alignas(8) std::array<uint8_t, 8> storage = {};
  ExpectEveryCompareExchangeOrderingPair<uint32_t>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32);
  ExpectEveryCompareExchangeOrderingPair<uint64_t>(
      storage.data(), IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64);
}

TEST(VMBytecodeInterpreterAtomicTest, CompareExchangeUsesExactCarrierBits) {
  alignas(8) std::array<uint8_t, 8> storage = {};
  StoreBits<uint64_t>(storage.data(), FloatBits<uint64_t>::kSignalingNaN);
  EXPECT_EQ(iree_vm_bytecode_atomic_compare_exchange(
                storage.data(), FloatBits<uint64_t>::kSignalingNaN,
                FloatBits<uint64_t>::kNegativeZero,
                IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64,
                IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED,
                IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED),
            FloatBits<uint64_t>::kSignalingNaN);
  EXPECT_EQ(LoadBits<uint64_t>(storage.data()),
            FloatBits<uint64_t>::kNegativeZero);
}

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
TEST(VMBytecodeInterpreterAtomicTest, PublishesMemoryWithReleaseAcquire) {
  alignas(4) std::array<uint8_t, 4> flag_storage = {};
  uint32_t payload = 0;
  uint32_t observed_payload = 0;
  std::thread reader([&]() {
    uint64_t observed_flag = 0;
    do {
      observed_flag = iree_vm_bytecode_atomic_apply(
          flag_storage.data(), 0,
          IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER,
          IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32,
          IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_ACQUIRE);
    } while (observed_flag == 0);
    observed_payload = payload;
  });
  std::thread writer([&]() {
    payload = 42;
    (void)iree_vm_bytecode_atomic_apply(
        flag_storage.data(), 1,
        IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_EXCHANGE_INTEGER,
        IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32,
        IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELEASE);
  });
  writer.join();
  reader.join();
  EXPECT_EQ(observed_payload, 42u);
}

template <typename T>
void ExpectUpdatesPreservedUnderContention(
    iree_vm_bytecode_buffer_atomic_carrier_t carrier) {
  alignas(8) std::array<uint8_t, sizeof(T)> storage = {};
  constexpr uint32_t kThreadCount = 4;
  constexpr uint32_t kIterationCount = 10000;
  std::array<std::thread, kThreadCount> threads;
  for (std::thread& thread : threads) {
    thread = std::thread([&storage, carrier]() {
      for (uint32_t i = 0; i < kIterationCount; ++i) {
        (void)iree_vm_bytecode_atomic_apply(
            storage.data(), 1, IREE_VM_BYTECODE_BUFFER_ATOMIC_KIND_ADD_INTEGER,
            carrier, IREE_VM_BYTECODE_BUFFER_ATOMIC_ORDERING_RELAXED);
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_EQ(LoadBits<T>(storage.data()),
            static_cast<T>(kThreadCount * kIterationCount));
}

TEST(VMBytecodeInterpreterAtomicTest, PreservesUpdatesUnderContention) {
  ExpectUpdatesPreservedUnderContention<uint32_t>(
      IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I32);
  ExpectUpdatesPreservedUnderContention<uint64_t>(
      IREE_VM_BYTECODE_BUFFER_ATOMIC_CARRIER_I64);
}
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

}  // namespace
}  // namespace iree::vm::bytecode::testing
