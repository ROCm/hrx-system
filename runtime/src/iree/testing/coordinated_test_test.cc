// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Self-test for the coordinated test harness.
//
// Validates the full lifecycle: process spawn, ready signaling, data exchange
// via temp directory files, exit code collection, and cleanup.
//
// The "writer" role writes a known string to a file and signals ready. The
// "reader" role reads the file and verifies the contents. Both roles exit 0
// on success. A separate test verifies that nonzero exit codes propagate.

#include "iree/testing/coordinated_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/testing/gtest.h"

//===----------------------------------------------------------------------===//
// Role entry functions
//===----------------------------------------------------------------------===//

static const char kExpectedMessage[] = "hello from writer";

static int writer_role(int argc, char** argv, const char* temp_directory) {
  // Write a message to a file in the shared temp directory.
  char file_path[512];
  iree_snprintf(file_path, sizeof(file_path), "%s/message.txt", temp_directory);
  FILE* file = fopen(file_path, "w");
  if (!file) {
    fprintf(stderr, "writer: failed to create %s\n", file_path);
    return 1;
  }
  fprintf(file, "%s", kExpectedMessage);
  fclose(file);

  // Signal that the file is ready for the reader.
  iree_coordinated_test_signal_ready(temp_directory);
  return 0;
}

static int reader_role(int argc, char** argv, const char* temp_directory) {
  // Read the message file written by the writer role.
  char file_path[512];
  iree_snprintf(file_path, sizeof(file_path), "%s/message.txt", temp_directory);
  FILE* file = fopen(file_path, "r");
  if (!file) {
    fprintf(stderr, "reader: failed to open %s\n", file_path);
    return 1;
  }
  char buffer[256] = {0};
  size_t bytes_read = fread(buffer, 1, sizeof(buffer) - 1, file);
  fclose(file);

  if (bytes_read == 0 || strcmp(buffer, kExpectedMessage) != 0) {
    fprintf(stderr, "reader: expected '%s', got '%s'\n", kExpectedMessage,
            buffer);
    return 1;
  }
  return 0;
}

static int failing_role(int argc, char** argv, const char* temp_directory) {
  (void)argc;
  (void)argv;
  (void)temp_directory;
  return 42;
}

static int argument_role(int argc, char** argv, const char* temp_directory) {
  (void)temp_directory;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--coordinated_test_payload=two words") == 0) {
      return 0;
    }
  }
  fprintf(stderr, "arguments: missing caller payload after launcher parsing\n");
  return 1;
}

//===----------------------------------------------------------------------===//
// Test configs
//===----------------------------------------------------------------------===//

// Writer signals ready, then reader starts and reads the file.
static const iree_test_role_t kDataExchangeRoles[] = {
    {"writer", writer_role, /*signals_ready=*/true},
    {"reader", reader_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kDataExchangeConfig = {
    .roles = kDataExchangeRoles,
    .role_count = 2,
};

// A role that exits with a nonzero code.
static const iree_test_role_t kFailingRoles[] = {
    {"failing", failing_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kFailingConfig = {
    .roles = kFailingRoles,
    .role_count = 1,
};

// A role that exits before publishing the readiness condition.
static const iree_test_role_t kFailsBeforeReadyRoles[] = {
    {"failing", failing_role, /*signals_ready=*/true},
};
static const iree_coordinated_test_config_t kFailsBeforeReadyConfig = {
    .roles = kFailsBeforeReadyRoles,
    .role_count = 1,
};

// All role functions must be reachable from the registered child dispatcher,
// even when individual tests launch only a subset of roles.
static const iree_test_role_t kAllRoles[] = {
    {"writer", writer_role, /*signals_ready=*/true},
    {"reader", reader_role, /*signals_ready=*/false},
    {"failing", failing_role, /*signals_ready=*/false},
    {"arguments", argument_role, /*signals_ready=*/false},
};
static const iree_coordinated_test_config_t kAllRolesConfig = {
    /*.roles=*/kAllRoles,
    /*.role_count=*/IREE_ARRAYSIZE(kAllRoles),
};
IREE_COORDINATED_TEST_REGISTER(kAllRolesConfig);

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST(CoordinatedTest, DataExchange) {
  // Writer writes a message, signals ready, reader verifies it.
  ASSERT_EQ(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kDataExchangeConfig));
}

TEST(CoordinatedTest, FailingRolePropagatesExitCode) {
  // A role that exits nonzero should cause run() to return nonzero.
  ASSERT_NE(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kFailingConfig));
}

TEST(CoordinatedTest, ExitBeforeReadyFailsWithoutDeadline) {
  ASSERT_NE(0, iree_coordinated_test_run(iree_coordinated_test_argc(),
                                         iree_coordinated_test_argv(),
                                         &kFailsBeforeReadyConfig));
}

TEST(CoordinatedTest, ForwardsArgumentsAfterLauncherParsing) {
  // The test invocation includes a gtest flag consumed before this body and
  // a caller-owned argument with spaces. The real child must receive the
  // caller payload and enter its role instead of recursively launching tests.
  const iree_test_role_t roles[] = {
      {"arguments", argument_role, /*signals_ready=*/false},
  };
  const iree_coordinated_test_config_t config = {roles, IREE_ARRAYSIZE(roles)};
  ASSERT_EQ(0,
            iree_coordinated_test_run(iree_coordinated_test_argc(),
                                      iree_coordinated_test_argv(), &config));
}
