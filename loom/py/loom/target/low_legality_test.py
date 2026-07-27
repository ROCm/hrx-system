# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from loom.dsl import ContractFamily, Dialect, Op, OpPhase
from loom.target.low_legality import TargetLowLegality, target_low_legality_by_name


def test_derives_target_low_legality_from_phase_and_contracts() -> None:
    dialect = Dialect("test", default_phase=OpPhase.EXECUTABLE)
    unsupported_dialect = Dialect("misc")
    source_dialect = Dialect("scf", default_phase=OpPhase.SOURCE_STRUCTURE)
    metadata_dialect = Dialect("target", default_phase=OpPhase.MODULE_METADATA)

    legality_by_name = target_low_legality_by_name(
        [
            (
                dialect,
                [
                    Op("test.add", group=dialect),
                    Op(
                        "test.contract",
                        group=dialect,
                        contracts=[ContractFamily.VECTOR_CONTRACTION],
                    ),
                ],
            ),
            (unsupported_dialect, [Op("misc.unsupported", group=unsupported_dialect)]),
            (source_dialect, [Op("scf.for", group=source_dialect)]),
            (metadata_dialect, [Op("target.generic", group=metadata_dialect)]),
        ]
    )

    assert legality_by_name["test.add"] == TargetLowLegality.CORE
    assert legality_by_name["test.contract"] == TargetLowLegality.PROVIDER
    assert legality_by_name["scf.for"] == TargetLowLegality.SOURCE_ONLY
    assert legality_by_name["target.generic"] == TargetLowLegality.MODULE_METADATA
    assert "misc.unsupported" not in legality_by_name
