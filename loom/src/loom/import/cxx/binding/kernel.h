// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_KERNEL_H_
#define LOOM_IMPORT_CXX_BINDING_KERNEL_H_

#include <cxx/attributes.h>
#include <cxx/symbols_fwd.h>

#include <optional>
#include <span>
#include <string_view>

#include "loom/import/cxx/value/types.h"
#include "loom/ir/facts.h"
#include "loom/ops/atomic.h"
#include "loom/ops/combining.h"

namespace loom::cxx_import {

// Lane exchange retains both High results in an ordinary source record. The
// admitted record partition supplies member identity at every source use;
// inactive-source handling remains explicit in the caller.
class SubgroupShuffleIntrinsic {
 public:
  static bool supports(std::string_view name) {
    return name == "kernel.subgroup.shuffle";
  }

  static std::optional<SubgroupShuffleIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  Value call(std::span<const Value> arguments, ValueArena& arena,
             loom_builder_t* builder, loom_location_id_t location) const;

  bool equivalent(const SubgroupShuffleIntrinsic& other) const {
    return mode_ == other.mode_ && result_partition_ == other.result_partition_;
  }

 private:
  SubgroupShuffleIntrinsic(uint8_t mode,
                           const RecordPartition* result_partition)
      : mode_(mode), result_partition_(result_partition) {}

  // Lane addressing mode in the canonical High enum representation.
  uint8_t mode_;
  // Admitted value/bool record, owned by the source Types instance.
  const RecordPartition* result_partition_;
};

// Typed source projection of kernel topology, votes, broadcasts, reductions,
// and scans. Calls require a kernel or required-inline helper; the translation
// driver owns that placement check. Collective participation and convergence
// remain High semantics, independent of the source spelling or target wave
// width.
class KernelCollectiveIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Admits one concrete scalar/vector signature. Unknown operations return
  // nullopt; invalid source signatures diagnose at owner.
  static std::optional<KernelCollectiveIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  // Emits the admitted collective from already evaluated arguments. Topology
  // queries project High's index result to the source's unsigned i32 value.
  loom_value_id_t call(std::span<const Value> arguments,
                       loom_builder_t* builder,
                       loom_location_id_t location) const;

  bool equivalent(const KernelCollectiveIntrinsic& other) const;

 private:
  enum class Operation {
    Id,
    Count,
    Size,
    LaneId,
    Any,
    All,
    Ballot,
    ActiveMask,
    Broadcast,
    BroadcastFirst,
    SubgroupReduce,
    WorkgroupReduce,
    SubgroupScan,
    WorkgroupScan,
  };

  struct Scan {
    // Inclusive or exclusive prefix, in the High enum representation.
    uint8_t mode;
    // Forward or reverse prefix, in the High enum representation.
    uint8_t direction;

    bool operator==(const Scan& other) const = default;
  };

  struct Combining {
    // Combining operation applied across the selected lanes.
    loom_combining_kind_t combining_kind;
    // Static subgroup cluster width, or zero when absent.
    uint32_t cluster_size;
    // Static subgroup cluster lane stride, or zero when absent.
    uint32_t cluster_stride;
    // Explicit collective arithmetic requirements in the High flag encoding.
    uint8_t math_flags;
    // Prefix semantics, present exactly for scan operations.
    std::optional<Scan> scan;
  };

  static std::optional<Operation> find(std::string_view name);

  KernelCollectiveIntrinsic(Operation operation, loom_type_t result_type)
      : operation_(operation), result_type_(result_type) {}
  KernelCollectiveIntrinsic(Operation operation, loom_type_t result_type,
                            Combining combining)
      : operation_(operation),
        result_type_(result_type),
        combining_(combining) {}

  // Operation selected once at declaration or template-instance admission.
  Operation operation_;
  // Source-selected scalar or rank-one vector representation.
  loom_type_t result_type_;
  // Combining semantics; present exactly for reduction and scan operations.
  std::optional<Combining> combining_;
};

// An execution rendezvous with explicit memory effects. Unlike subgroup
// queries, High barriers may also appear in ordinary callable helpers.
class BarrierIntrinsic {
 public:
  static bool supports(std::string_view name);

  // Admits void() with leading memory-space, scope, and ordering template
  // arguments. High owns the allowed combinations of memory effects.
  static std::optional<BarrierIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  void call(loom_builder_t* builder, loom_location_id_t location) const;

  bool equivalent(const BarrierIntrinsic& other) const;

 private:
  BarrierIntrinsic(loom_value_fact_memory_space_t memory_space,
                   loom_atomic_scope_t scope, loom_atomic_ordering_t ordering)
      : memory_space_(memory_space), scope_(scope), ordering_(ordering) {}

  // Storage space whose accesses participate in the barrier's ordering.
  loom_value_fact_memory_space_t memory_space_;
  // Invocations participating in the execution rendezvous.
  loom_atomic_scope_t scope_;
  // Memory ordering within the named space and scope.
  loom_atomic_ordering_t ordering_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_KERNEL_H_
