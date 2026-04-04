// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0

#include <pyre_runtime.h>

int main(void) {
  int major = -1;
  int minor = -1;
  int patch = -1;
  pyre_runtime_version(&major, &minor, &patch);
  return major == PYRE_VERSION_MAJOR && minor == PYRE_VERSION_MINOR &&
          patch == PYRE_VERSION_PATCH
      ? 0
      : 1;
}
