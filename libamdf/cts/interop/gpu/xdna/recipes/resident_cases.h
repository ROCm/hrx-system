// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_RESIDENT_CASES_H_
#define AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_RESIDENT_CASES_H_

#include "libamdf/cts/interop/gpu/xdna/recipes/resident_fixture.h"

namespace {

struct ImportedExchangeCase {
  // Exact external representation, selected before allocation.
  amdf_external_memory_type_t transport;
  // Export at the logical source base or an aligned interior range.
  ResidentSourceOffset source_offset;
  // First accepted participant when both run.
  LaunchOrder order;
  // Complete dependent exchanges before final acknowledgement.
  uint32_t round_count;
  // A sole participant exercises the ordinary prestart ABORT protocol.
  Participants participants;
};

class ResidentImportedExchangeTest
    : public ResidentGpuXdnaTest,
      public ::testing::WithParamInterface<ImportedExchangeCase> {};

TEST_P(ResidentImportedExchangeTest, CausalRoundTrip) {
  const auto& parameters = GetParam();
  ResidentImportPlan import_plan;
  auto gpu_access = accesses_[1];
  gpu_access.requirements.flags |= AMDF_MEMORY_FLAG_QUEUE_STORAGE;
  ASSERT_NO_FATAL_FAILURE(import_plan.Find(api_, system_scope_, gpu_access,
                                           accesses_[0], parameters.transport,
                                           parameters.source_offset));
  if (IsSkipped()) {
    return;
  }
  Run(AMDF_MEMORY_PROFILE_ROLE_CREATE, parameters.round_count, 0xFFFFFFFEu,
      parameters.order, parameters.participants, {1024, 1, 2},
      ServiceSchedule::kWindow, &import_plan);
}

std::vector<ImportedExchangeCase> ImportedExchangeCases() {
  std::vector<ImportedExchangeCase> cases;
  for (amdf_external_memory_type_t transport :
       {AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD}) {
    for (auto offset :
         {ResidentSourceOffset::kZero, ResidentSourceOffset::kAligned}) {
      for (auto order : {LaunchOrder::kGpuFirst, LaunchOrder::kNpuFirst}) {
        for (uint32_t rounds : {0u, 1u, 17u}) {
          cases.push_back(
              {transport, offset, order, rounds, Participants::kBoth});
        }
        cases.push_back({transport, offset, order, 17,
                         order == LaunchOrder::kGpuFirst ? Participants::kGpu
                                                         : Participants::kNpu});
      }
    }
  }
  return cases;
}

std::string ImportedExchangeCaseName(
    const ::testing::TestParamInfo<ImportedExchangeCase>& info) {
  std::string name =
      info.param.transport == AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD ? "DmaBuf"
                                                                   : "Opaque";
  name += info.param.source_offset == ResidentSourceOffset::kZero ? "Base"
                                                                  : "Subrange";
  name += info.param.order == LaunchOrder::kGpuFirst ? "GpuFirst" : "NpuFirst";
  return name + (info.param.participants == Participants::kBoth
                     ? "Rounds" + std::to_string(info.param.round_count)
                     : "Abort");
}

struct NpuInitiatedCase {
  // GPU return count; a positive count adds one closing NPU payload.
  uint32_t round_count;
  // Complete payload length in 32-bit words.
  uint32_t word_count;
  // Accepted participants; a sole participant exercises prestart ABORT.
  Participants participants;
  // Advertised construction role for all joint backing owners.
  amdf_memory_profile_roles_t role = AMDF_MEMORY_PROFILE_ROLE_REGISTER;
  // First participant submitted when both will run.
  LaunchOrder order = LaunchOrder::kNpuFirst;
  // Word offset from slot generation to payload, sharing or separating a line.
  uint32_t word_offset = 16;
};

class ResidentNpuInitiatedTest
    : public ResidentGpuXdnaTest,
      public ::testing::WithParamInterface<NpuInitiatedCase> {};

TEST_P(ResidentNpuInitiatedTest, ReturnsEveryWord) {
  const auto& parameters = GetParam();
  const auto order =
      parameters.participants == Participants::kGpu   ? LaunchOrder::kGpuFirst
      : parameters.participants == Participants::kNpu ? LaunchOrder::kNpuFirst
                                                      : parameters.order;
  Run(parameters.role, parameters.round_count, 0xFFFFFFFEu, order,
      parameters.participants, {parameters.word_count, parameters.word_offset},
      ServiceSchedule::kNpuInitiated);
}

std::string NpuInitiatedCaseName(
    const ::testing::TestParamInfo<NpuInitiatedCase>& info) {
  if (info.param.participants != Participants::kBoth) {
    return info.param.participants == Participants::kGpu ? "GpuOnlyAbort"
                                                         : "NpuOnlyAbort";
  }
  return "Rounds" + std::to_string(info.param.round_count) + "Words" +
         std::to_string(info.param.word_count);
}

std::vector<NpuInitiatedCase> NpuInitiatedBackingCases() {
  std::vector<NpuInitiatedCase> cases;
  for (amdf_memory_profile_roles_t role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_REGISTER}) {
    if (role == AMDF_MEMORY_PROFILE_ROLE_CREATE) {
      for (auto participant : {Participants::kGpu, Participants::kNpu}) {
        cases.push_back({17, 16, participant, role});
      }
    }
    for (auto order : {LaunchOrder::kGpuFirst, LaunchOrder::kNpuFirst}) {
      for (uint32_t round_count : {0u, 1u, 17u, 257u}) {
        // The registered anchors cover these three NPU-first counts.
        if (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER &&
            order == LaunchOrder::kNpuFirst && round_count != 257) {
          continue;
        }
        cases.push_back({round_count, 16, Participants::kBoth, role, order});
      }
    }
    for (uint32_t word_count : {1u, 4u, 15u, 16u, 17u, 64u, 1024u}) {
      for (uint32_t word_offset : {1u, 16u}) {
        // Startup covers W16/P16; the registered anchor also covers W1/P16.
        if (word_offset == 16 &&
            (word_count == 16 ||
             (role == AMDF_MEMORY_PROFILE_ROLE_REGISTER && word_count == 1))) {
          continue;
        }
        cases.push_back({17, word_count, Participants::kBoth, role,
                         LaunchOrder::kNpuFirst, word_offset});
      }
    }
  }
  return cases;
}

std::string NpuInitiatedBackingCaseName(
    const ::testing::TestParamInfo<NpuInitiatedCase>& info) {
  std::string name = info.param.role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                         ? "Registered"
                         : "Allocated";
  if (info.param.participants != Participants::kBoth) {
    return name + NpuInitiatedCaseName(info);
  }
  name += info.param.order == LaunchOrder::kGpuFirst ? "GpuFirst" : "NpuFirst";
  return name + NpuInitiatedCaseName(info) +
         (info.param.word_offset == 1 ? "SharedFirstLine"
                                      : "SeparateFirstLine");
}

class ResidentNpuSdmaTest
    : public ResidentGpuXdnaTest,
      public ::testing::WithParamInterface<NpuInitiatedCase> {
 protected:
  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    return MatchSdmaEndpoint(endpoint, out_matches);
  }
};

TEST_P(ResidentNpuSdmaTest, CopiesFeedTheNextNpuRequest) {
  const auto& parameters = GetParam();
  // The page-sized shape starts with a full copy; the others start with one
  // word. Subsequent page/length choices depend on the actual returned data.
  const uint32_t seed =
      parameters.word_count == 1024 ? 0x0003FE06u : 0xFFFFFFFEu;
  const auto order =
      parameters.participants == Participants::kGpu   ? LaunchOrder::kGpuFirst
      : parameters.participants == Participants::kNpu ? LaunchOrder::kNpuFirst
                                                      : parameters.order;
  Run(parameters.role, parameters.round_count, seed, order,
      parameters.participants, {parameters.word_count, parameters.word_offset},
      ServiceSchedule::kNpuSdma);
}

std::vector<NpuInitiatedCase> NpuSdmaCases() {
  auto cases = NpuInitiatedBackingCases();
  for (const auto& anchor : {NpuInitiatedCase{17, 16, Participants::kGpu},
                             NpuInitiatedCase{17, 16, Participants::kNpu},
                             NpuInitiatedCase{0, 16, Participants::kBoth},
                             NpuInitiatedCase{1, 1, Participants::kBoth},
                             NpuInitiatedCase{1, 16, Participants::kBoth},
                             NpuInitiatedCase{17, 1, Participants::kBoth},
                             NpuInitiatedCase{17, 16, Participants::kBoth}}) {
    cases.push_back(anchor);
  }
  return cases;
}

uint32_t SeedForPeerCause(uint32_t peer_cause) {
  // Held generation one's response is 3 * (seed + 257) + 1. Multiplication
  // by the inverse of three modulo 2^32 makes the peer's first cause match
  // the corresponding credit-window case, including every payload word.
  return (peer_cause - 772u) * 0xAAAAAAABu;
}

const char* ScheduleCaseSuffix(ServiceSchedule schedule) {
  switch (schedule) {
    case ServiceSchedule::kSplitPayload0PayloadFirst:
      return "Payload0PayloadFirst";
    case ServiceSchedule::kSplitPayload0ReadyFirst:
      return "Payload0ReadyFirst";
    case ServiceSchedule::kSplitPayload1PayloadFirst:
      return "Payload1PayloadFirst";
    case ServiceSchedule::kSplitPayload1ReadyFirst:
      return "Payload1ReadyFirst";
    default:
      break;
  }
  return schedule == ServiceSchedule::kWindow ||
                 schedule == ServiceSchedule::kRelayedWindow
             ? ""
         : schedule == ServiceSchedule::kHoldFirst ? "HoldFirst"
                                                   : "HoldSecond";
}

struct ExchangeCase {
  // Advertised construction role for all joint backing owners.
  amdf_memory_profile_roles_t role;
  // First participant submitted while both still await the startup decision.
  LaunchOrder order;
  // Number of complete dependent generations before final acknowledgement.
  uint32_t round_count;
  // Maximum requests admitted before waiting for a response.
  uint32_t credit_count;
  // Resident service topology and terminal-output route.
  ServiceSchedule schedule = ServiceSchedule::kWindow;
};

class ResidentExchangeTest
    : public ResidentGpuXdnaTest,
      public ::testing::WithParamInterface<ExchangeCase> {};

TEST_P(ResidentExchangeTest, CausalRoundTrip) {
  const auto& parameters = GetParam();
  // Both seeds exercise unsigned wrapping without an identity first request.
  uint32_t seed = parameters.round_count == 1 ? UINT32_MAX : 0x7FFFFF00u;
  if (parameters.schedule == ServiceSchedule::kHoldFirst ||
      parameters.schedule == ServiceSchedule::kHoldSecond) {
    seed = SeedForPeerCause(parameters.round_count == 17 ? 0xFFFFFFFEu : seed);
  }
  Run(parameters.role, parameters.round_count, seed, parameters.order,
      Participants::kBoth, {16, 16, parameters.credit_count},
      parameters.schedule);
}

std::vector<ExchangeCase> ExchangeCases(
    uint32_t credit_count, std::span<const uint32_t> round_counts,
    ServiceSchedule schedule = ServiceSchedule::kWindow) {
  std::vector<ExchangeCase> cases;
  for (amdf_memory_profile_roles_t role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_REGISTER}) {
    for (auto order : {LaunchOrder::kGpuFirst, LaunchOrder::kNpuFirst}) {
      for (uint32_t round_count : round_counts) {
        cases.push_back({role, order, round_count, credit_count, schedule});
      }
    }
  }
  return cases;
}

std::vector<ExchangeCase> IndependentExchangeCases() {
  std::vector<ExchangeCase> cases;
  for (auto parameters : ExchangeCases(1, std::array{1u, 17u, 257u})) {
    for (auto schedule :
         {ServiceSchedule::kHoldFirst, ServiceSchedule::kHoldSecond}) {
      parameters.schedule = schedule;
      cases.push_back(parameters);
    }
  }
  return cases;
}

std::string ExchangeCaseName(
    const ::testing::TestParamInfo<ExchangeCase>& info) {
  std::string name = info.param.role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                         ? "Registered"
                         : "Allocated";
  name += info.param.order == LaunchOrder::kGpuFirst ? "GpuFirst" : "NpuFirst";
  return name + "Rounds" + std::to_string(info.param.round_count) +
         ScheduleCaseSuffix(info.param.schedule);
}

struct PrestartAbortCase {
  // Advertised construction role for the accepted participant's joint backing.
  amdf_memory_profile_roles_t role;
  // Sole participant accepted before the host publishes ABORT.
  Participants participant;
  // Number of paired slots prepared before the partial startup.
  uint32_t credit_count;
  // Resident service topology and terminal-output route.
  ServiceSchedule schedule = ServiceSchedule::kWindow;
};

class ResidentPrestartAbortTest
    : public ResidentGpuXdnaTest,
      public ::testing::WithParamInterface<PrestartAbortCase> {};

TEST_P(ResidentPrestartAbortTest, DrainsWithoutPeer) {
  const auto& parameters = GetParam();
  const auto order = parameters.participant == Participants::kGpu
                         ? LaunchOrder::kGpuFirst
                         : LaunchOrder::kNpuFirst;
  Run(parameters.role, 17, 0x80000001u, order, parameters.participant,
      {16, 16, parameters.credit_count}, parameters.schedule);
}

std::vector<PrestartAbortCase> PrestartAbortCases(
    uint32_t credit_count,
    ServiceSchedule schedule = ServiceSchedule::kWindow) {
  std::vector<PrestartAbortCase> cases;
  for (amdf_memory_profile_roles_t role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_REGISTER}) {
    for (auto participant : {Participants::kGpu, Participants::kNpu}) {
      cases.push_back({role, participant, credit_count, schedule});
    }
  }
  return cases;
}

std::vector<PrestartAbortCase> IndependentPrestartAbortCases() {
  std::vector<PrestartAbortCase> cases;
  for (auto parameters : PrestartAbortCases(1)) {
    for (auto schedule :
         {ServiceSchedule::kHoldFirst, ServiceSchedule::kHoldSecond}) {
      parameters.schedule = schedule;
      cases.push_back(parameters);
    }
  }
  return cases;
}

std::string PrestartAbortCaseName(
    const ::testing::TestParamInfo<PrestartAbortCase>& info) {
  std::string name = info.param.role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                         ? "Registered"
                         : "Allocated";
  return name +
         (info.param.participant == Participants::kGpu ? "GpuOnly"
                                                       : "NpuOnly") +
         ScheduleCaseSuffix(info.param.schedule);
}

struct PayloadCase {
  // Advertised construction role for the complete joint slot backing.
  amdf_memory_profile_roles_t role;
  // Actual payload extent and placement relative to its generation word.
  ExchangeShape shape;
  // Resident service topology and terminal-output route.
  ServiceSchedule schedule = ServiceSchedule::kWindow;
};

class ResidentPayloadTest : public ResidentGpuXdnaTest,
                            public ::testing::WithParamInterface<PayloadCase> {
};

TEST_P(ResidentPayloadTest, PublishesCompleteResponse) {
  const auto& parameters = GetParam();
  const bool independent = parameters.schedule == ServiceSchedule::kHoldFirst ||
                           parameters.schedule == ServiceSchedule::kHoldSecond;
  const uint32_t seed =
      independent ? SeedForPeerCause(0xFFFFFFFEu) : 0xFFFFFFFEu;
  Run(parameters.role, 17, seed, LaunchOrder::kNpuFirst, Participants::kBoth,
      parameters.shape, parameters.schedule);
}

std::vector<PayloadCase> PayloadCases(
    uint32_t credit_count, std::span<const uint32_t> word_counts,
    ServiceSchedule schedule = ServiceSchedule::kWindow) {
  std::vector<PayloadCase> cases;
  for (amdf_memory_profile_roles_t role :
       {AMDF_MEMORY_PROFILE_ROLE_CREATE, AMDF_MEMORY_PROFILE_ROLE_REGISTER}) {
    for (uint32_t word_count : word_counts) {
      for (uint32_t word_offset : {1u, 16u}) {
        cases.push_back(
            {role, {word_count, word_offset, credit_count}, schedule});
      }
    }
  }
  return cases;
}

std::vector<PayloadCase> IndependentPayloadCases() {
  std::vector<PayloadCase> cases;
  for (auto parameters : PayloadCases(1, std::array{1u, 16u, 1024u})) {
    // The startup matrix already covers this complete shape in both orders.
    if (parameters.shape.word_count == 16 &&
        parameters.shape.word_offset == 16) {
      continue;
    }
    for (auto schedule :
         {ServiceSchedule::kHoldFirst, ServiceSchedule::kHoldSecond}) {
      parameters.schedule = schedule;
      cases.push_back(parameters);
    }
  }
  return cases;
}

std::string PayloadCaseName(const ::testing::TestParamInfo<PayloadCase>& info) {
  std::string name = info.param.role == AMDF_MEMORY_PROFILE_ROLE_REGISTER
                         ? "Registered"
                         : "Allocated";
  name += "Words" + std::to_string(info.param.shape.word_count);
  return name +
         (info.param.shape.word_offset == 1 ? "SharedFirstLine"
                                            : "SeparateFirstLine") +
         ScheduleCaseSuffix(info.param.schedule);
}

}  // namespace

#endif  // AMDF_CTS_INTEROP_GPU_XDNA_RECIPES_RESIDENT_CASES_H_
