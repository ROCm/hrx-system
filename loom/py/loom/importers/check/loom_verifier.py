# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Production Loom IR verification for importer check output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loom.importers.check.cases import CheckCase
from loom.importers.check.results import CheckResult
from loom.tools.loom_opt import LoomOpt


@dataclass(frozen=True, slots=True)
class LoomOutputVerifier:
    """Runs printed Loom IR through the production verifier."""

    loom_opt: LoomOpt

    @classmethod
    def resolve(cls, explicit_path: Path | None) -> LoomOutputVerifier | None:
        """Resolves the verifier tool for importer check execution."""

        loom_opt = LoomOpt.resolve(explicit_path)
        if loom_opt is None:
            return None
        return cls(loom_opt)

    def verify(self, case: CheckCase, loom_ir: str) -> CheckResult | None:
        """Returns a failed check result if `loom_ir` fails C verification."""

        result = self.loom_opt.verify_module_text(loom_ir)
        if result.succeeded:
            return None
        if result.invocation_error is not None:
            return CheckResult(
                path=case.path,
                case_index=case.index,
                returncode=1,
                stdout=loom_ir,
                stderr=f"{result.invocation_error}\n",
                input=case.input,
                expected=case.expected,
                mismatch="generated Loom IR could not be verified",
            )
        diagnostic_text = result.diagnostic_text() or result.stdout
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=result.returncode,
            stdout=loom_ir,
            stderr=diagnostic_text,
            input=case.input,
            expected=case.expected,
            mismatch="generated Loom IR failed verification",
        )
