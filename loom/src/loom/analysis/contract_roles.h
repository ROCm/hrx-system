// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared contraction operand roles.

#ifndef LOOM_ANALYSIS_CONTRACT_ROLES_H_
#define LOOM_ANALYSIS_CONTRACT_ROLES_H_

typedef enum loom_contract_operand_role_e {
  // Unknown or uninitialized operand role.
  LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN = 0,
  // Left-hand source operand.
  LOOM_CONTRACT_OPERAND_ROLE_LHS = 1,
  // Right-hand source operand.
  LOOM_CONTRACT_OPERAND_ROLE_RHS = 2,
  // Accumulator input operand.
  LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR = 3,
  // Result operand.
  LOOM_CONTRACT_OPERAND_ROLE_RESULT = 4,
} loom_contract_operand_role_t;

#endif  // LOOM_ANALYSIS_CONTRACT_ROLES_H_
