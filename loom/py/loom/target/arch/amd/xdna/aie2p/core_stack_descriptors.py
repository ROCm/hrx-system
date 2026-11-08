# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native AIE2P stack state and frame-relative memory transfers."""

from dataclasses import dataclass

from loom.target.arch.amd.xdna.aie2p.core_descriptor_spec import _DescriptorSpec
from loom.target.arch.amd.xdna.aie2p.core_machine_data import (
    DIMENSION_FIELDS,
    STACK_COMPOSITE_FIELDS,
)
from loom.target.low_descriptors import DescriptorFlag, RegisterPart


@dataclass(frozen=True, slots=True)
class StackPiece:
    # Stable component name within the register carrier.
    name: str
    # Native load and store form prefixes before the addressing suffix.
    forms: tuple[str, str]
    # Memory bytes transferred by this component.
    byte_length: int
    # Byte offset within its component plane.
    byte_offset: int
    # Byte stride between logical register units in this component plane.
    unit_byte_stride: int
    # Bytes per complete slot unit preceding this component plane.
    plane_byte_stride: int = 0
    # Native itinerary bank suffix, or empty for an unspecialized itinerary.
    itinerary_bank: str = ""
    # Partial register view used by each native transfer.
    register_part: str | None = None
    # Component-specific load and store encoding adapters.
    adapters: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class StackRecipe:
    # Machine storage class used by the native transfer results and operands.
    register_class: str
    # Number of logical allocation units transferred by each recipe instance.
    unit_count: int
    # Native components, in store and reload construction order.
    pieces: tuple[StackPiece, ...]
    # Source storage classes supported, using a copy bridge when they differ.
    source_classes: tuple[str, ...]

    @property
    def name(self) -> str:
        suffix = f".x{self.unit_count}" if self.unit_count != 1 else ""
        return self.register_class.lower() + suffix


_SCALAR_FORMS = ("LDA_dms_lda", "ST_dms_sts")
_VECTOR_FORMS = ("VLDA_dmx_lda_x", "VST_dmx_sts_x")
_ACCUMULATOR_FORMS = ("VLDA_dmx_lda_bm", "VST_dmx_sts_bm")
_FIFO_FORMS = ("VLDA_dmx_lda_fifohl", "VST_dmx_sts_fifohl")


def _stack_recipes() -> tuple[StackRecipe, ...]:
    recipes = []
    for register_class, bank in (
        *(
            (name, "eR")
            for name in (
                "eR",
                "eRF2",
                "eRS16",
                "mR16_vcompare",
                "mR26_fifo_st",
                "mR26_lock",
                "mR27_select",
                "mR28_tlast",
                "mR29_insert",
                "mR31_divs",
                "mR31_scd",
            )
        ),
        *((name, name) for name in ("eDC", "eDJ", "eDN", "eM", "eP")),
        ("ePS", "eP"),
        ("mPfs", "eP"),
    ):
        recipes.append(
            StackRecipe(
                register_class,
                1,
                (StackPiece("", _SCALAR_FORMS, 4, 0, 4, itinerary_bank=bank),),
                (register_class, "eS", "mS2", "mS3")
                if register_class == "eR"
                else (register_class,),
            )
        )
    for units, forms, length in (
        (1, ("VLDA_dmw_lda_w", "VST_dmw_sts_w"), 32),
        (2, _VECTOR_FORMS, 64),
    ):
        recipes.append(
            StackRecipe(
                "VEC256", units, (StackPiece("", forms, length, 0, 32),), ("VEC256",)
            )
        )
    recipes.extend(
        StackRecipe(
            register_class,
            1,
            (StackPiece("", _ACCUMULATOR_FORMS, 64, 0, 64),),
            (register_class,),
        )
        for register_class in ("eBMLL", "mBMSm", "mBMs")
    )
    for register_class, fields in DIMENSION_FIELDS.items():
        prefix = f"aie2p.{register_class.lower()}"
        stride = 16 if register_class == "eD" else 32
        pieces = tuple(
            StackPiece(
                component,
                _SCALAR_FORMS,
                4,
                index * 4,
                stride,
                itinerary_bank=bank,
                register_part=f"{prefix}.{component}",
                adapters=tuple(
                    f"LOOM_{register_class}_{component}_{adapter}"
                    for adapter in ("OP_mLdaScl", "OP_mSclSt")
                ),
            )
            for index, (component, _, bank) in enumerate(fields)
        )
        recipes.append(StackRecipe(register_class, 1, pieces, (register_class,)))
    for register_class, fields in STACK_COMPOSITE_FIELDS.items():
        prefix = f"aie2p.{register_class.lower()}"
        accumulator = register_class == "eDM"
        length = 64 if accumulator else 4
        pieces = tuple(
            StackPiece(
                component,
                _ACCUMULATOR_FORMS if accumulator else _SCALAR_FORMS,
                length,
                index * length,
                length * len(fields),
                itinerary_bank="" if accumulator else "eR",
                register_part=f"{prefix}.{component}",
                adapters=tuple(
                    f"LOOM_{register_class}_{component}_{adapter}"
                    for adapter in adapters
                ),
            )
            for index, (component, _, adapters) in enumerate(fields)
        )
        recipes.append(StackRecipe(register_class, 1, pieces, (register_class,)))
    for register_class, banks in (
        ("eLdFifoReg", ("eLdFifoLReg", "eLdFifoHReg")),
        ("mStFifo", ("mStFifol", "mStFifoh")),
    ):
        prefix = f"aie2p.{register_class.lower()}"
        pieces = tuple(
            StackPiece(
                f"{half}512",
                _FIFO_FORMS,
                64,
                index * 64,
                128,
                itinerary_bank=bank,
                register_part=f"{prefix}.{half}512",
                adapters=(f"LOOM_{register_class}_{half}512",) * 2,
            )
            for index, (half, bank) in enumerate(
                zip(("low", "high"), banks, strict=True)
            )
        )
        recipes.append(StackRecipe(register_class, 1, pieces, (register_class,)))
    recipes.append(
        StackRecipe(
            "mEXa",
            1,
            (
                StackPiece(
                    "mantissas",
                    _VECTOR_FORMS,
                    64,
                    0,
                    64,
                    register_part="aie2p.mexa.mantissas",
                    adapters=(
                        "LOOM_mEXa_mantissas_OP_mXa",
                        "LOOM_mEXa_mantissas_OP_mXs",
                    ),
                ),
                *(
                    StackPiece(
                        f"exponents.{half}",
                        _SCALAR_FORMS,
                        4,
                        index * 4,
                        8,
                        64,
                        itinerary_bank="eElo",
                        register_part=f"aie2p.mexa.exponents.{half}",
                        adapters=tuple(
                            f"LOOM_mEXa_exponents_{half}_{adapter}"
                            for adapter in ("OP_mLdaScl", "OP_mSclSt")
                        ),
                    )
                    for index, half in enumerate(("low32", "high32"))
                ),
            ),
            ("mEXa",),
        )
    )
    return tuple(recipes)


STACK_RECIPES = _stack_recipes()
STACK_SPILLABLE_CLASSES = frozenset(
    name for recipe in STACK_RECIPES for name in recipe.source_classes
)
STACK_REGISTER_PARTS = tuple(
    RegisterPart(
        f"aie2p.{register_class.lower()}.{component}",
        f"aie2p.{register_class.lower()}",
        1 << index,
    )
    for register_class in ("mL8m", "eDM")
    for index, (component, _, _) in enumerate(STACK_COMPOSITE_FIELDS[register_class])
)


def stack_descriptor_key(
    recipe: StackRecipe,
    piece: StackPiece,
    operation: str,
    indexed: bool,
    continuation: bool = False,
) -> str:
    if recipe.register_class == "eR" and indexed:
        return f"amd.xdna.aie2p.{operation}.scalar.i32.indexed.immediate"
    component = f".{piece.name}" if piece.name else ""
    suffix = (".indexed" if indexed else "") + (".continue" if continuation else "")
    return f"amd.xdna.aie2p.stack.{operation}.{recipe.name}{component}{suffix}"


def _stack_transfer_descriptors() -> tuple[_DescriptorSpec, ...]:
    result = []
    for recipe in STACK_RECIPES:
        for piece in recipe.pieces:
            transfers = ((False, False), (True, False))
            if piece.register_part:
                transfers += ((False, True),)
            for store, continuation in transfers:
                operation, operand = ("store", "src") if store else ("load", "dst")
                for indexed in (False, True):
                    if recipe.register_class == "eR" and indexed:
                        continue
                    form = piece.forms[store] + ("_idx_imm" if indexed else "_spill")
                    bank = f"_{piece.itinerary_bank}" if piece.itinerary_bank else ""
                    component = f".{piece.name}" if piece.name else ""
                    mnemonic = (
                        ("st.stack" if store else "lda.stack")
                        if recipe.register_class == "eR"
                        else (
                            f"{operation}.stack.{recipe.name}{component}"
                            + (".indexed" if indexed else "")
                        )
                    )
                    if continuation:
                        mnemonic += ".continue"
                    result.append(
                        _DescriptorSpec(
                            form,
                            stack_descriptor_key(
                                recipe, piece, operation, indexed, continuation
                            ),
                            f"memory.{operation}.stack.{recipe.name}{component}",
                            f"II_{form}{bank}",
                            storage_overrides=((operand, recipe.register_class),),
                            operand_register_parts=((operand, piece.register_part),)
                            if piece.register_part
                            else (),
                            encoding_adapter_overrides=(
                                (operand, piece.adapters[store]),
                            )
                            if piece.adapters
                            else (),
                            preserves_unwritten_parts=continuation,
                            asm_mnemonic=mnemonic,
                            memory_width_bits=piece.byte_length * 8,
                        )
                    )
    return tuple(result)


STACK_DESCRIPTOR_SPECS = (
    _DescriptorSpec(
        "MOVXM",
        "amd.xdna.aie2p.state.stack-pointer",
        "state.write.stack-pointer",
        "II_MOVXM_mSPm",
        storage_overrides=(("dst", "mSPm"),),
        implicit_outputs=("dst",),
        asm_mnemonic="set.stack-pointer",
        flags=(DescriptorFlag.STATE_ASSIGNMENT,),
    ),
    _DescriptorSpec(
        "PADDXM_pstm_sp_imm",
        "amd.xdna.aie2p.stack.adjust",
        "stack.adjust",
        "II_PADDXM_pstm_sp_imm",
        asm_mnemonic="adjust.stack-pointer",
    ),
    _DescriptorSpec(
        "MOV_alu_mv_mv_mv_scl",
        "amd.xdna.aie2p.stack.address",
        "stack.address",
        "II_MOV_alu_mv_mv_mv_scl_eP_mSPm",
        storage_overrides=(("dst", "eP"), ("src", "mSPm")),
        implicit_inputs=("src",),
        asm_mnemonic="mov.stack-address",
    ),
    *_stack_transfer_descriptors(),
)
