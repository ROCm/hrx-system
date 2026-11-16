# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Providers shared by Loom corpus build and execution rules."""

LoomCorpusProgramInfo = provider(
    doc = "Build outputs and source identity for one corpus program.",
    fields = {
        "artifacts": "Depset of deployable artifacts across target profiles.",
        "compile_reports": "Depset of detailed compile reports.",
        "prepared_packages": "Depset of prepared execution-package directories.",
        "qualification_results": "Depset of batched diagnostic-xfail results.",
        "source_identity": "Canonical corpus source identity.",
        "sources": "Depset containing the qualified Loom source.",
    },
)

LoomCorpusBuildInfo = provider(
    doc = "Collected target-owned build outputs for Loom corpus programs.",
    fields = {
        "artifacts": "Depset of deployable artifacts across programs and profiles.",
        "compile_reports": "Depset of detailed compile reports.",
        "prepared_packages": "Depset of prepared execution-package directories.",
        "qualification_results": "Depset of batched diagnostic-xfail results.",
        "source_identities": "Depset of canonical corpus source identities.",
        "sources": "Depset of qualified Loom sources.",
    },
)
