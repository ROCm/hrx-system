// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_object.h"

#include <string.h>

#include "loom/target/arch/amd/xdna/aie2p/emit/relocation.h"

static const loom_storage_space_t kStorageSpaceOrder[] = {
    LOOM_STORAGE_SPACE_STACK,
    LOOM_STORAGE_SPACE_SCRATCH,
    LOOM_STORAGE_SPACE_PRIVATE,
    LOOM_STORAGE_SPACE_WORKGROUP,
};

bool loom_aie2p_leaf_may_write_register(
    const loom_aie2p_leaf_realization_t* realization,
    loom_aie2p_physical_register_id_t physical_register) {
  loom_aie2p_physical_register_info_t register_info;
  const bool found = loom_aie2p_machine_query_physical_register(
      physical_register, &register_info);
  if (!found) {
    IREE_ASSERT_UNREACHABLE("physical register ID must be valid");
    IREE_BUILTIN_UNREACHABLE();
  }
  for (uint8_t i = 0; i < register_info.atomic_unit_count; ++i) {
    const loom_aie2p_atomic_unit_id_t unit =
        loom_aie2p_machine_physical_register_atomic_unit(physical_register, i);
    if (unit == LOOM_AIE2P_ATOMIC_UNIT_ID_INVALID) {
      IREE_ASSERT_UNREACHABLE("physical register atomic unit must be valid");
      IREE_BUILTIN_UNREACHABLE();
    }
    if (realization->register_writes.words[unit / 64] &
        (UINT64_C(1) << (unit % 64))) {
      return true;
    }
  }
  return false;
}

const loom_aie2p_leaf_storage_requirement_t*
loom_aie2p_leaf_storage_requirement(
    const loom_aie2p_leaf_realization_t* realization,
    loom_storage_space_t storage_space) {
  switch (storage_space) {
    case LOOM_STORAGE_SPACE_STACK:
      return &realization->stack;
    case LOOM_STORAGE_SPACE_SCRATCH:
      return &realization->scratch;
    case LOOM_STORAGE_SPACE_PRIVATE:
      return &realization->private_storage;
    case LOOM_STORAGE_SPACE_WORKGROUP:
      return &realization->workgroup_storage;
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified function storage must use a known storage space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static loom_aie2p_leaf_storage_requirement_t*
loom_aie2p_leaf_object_storage_requirement_mutable(
    loom_aie2p_leaf_realization_t* realization,
    loom_storage_space_t storage_space) {
  return (loom_aie2p_leaf_storage_requirement_t*)
      loom_aie2p_leaf_storage_requirement(realization, storage_space);
}

static iree_string_view_t loom_aie2p_leaf_object_storage_space_name(
    loom_storage_space_t storage_space) {
  switch (storage_space) {
    case LOOM_STORAGE_SPACE_STACK:
      return IREE_SV("stack");
    case LOOM_STORAGE_SPACE_SCRATCH:
      return IREE_SV("scratch");
    case LOOM_STORAGE_SPACE_PRIVATE:
      return IREE_SV("private");
    case LOOM_STORAGE_SPACE_WORKGROUP:
      return IREE_SV("workgroup");
    default:
      IREE_ASSERT_UNREACHABLE(
          "verified function storage must use a known storage space");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static void loom_aie2p_leaf_object_copy_storage_requirements(
    const loom_aie2p_leaf_program_plan_t* plan,
    loom_aie2p_leaf_realization_t* realization) {
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    const loom_storage_space_t space = kStorageSpaceOrder[i];
    *loom_aie2p_leaf_object_storage_requirement_mutable(realization, space) =
        plan->storage_requirements[space];
  }
  realization->spill = plan->spill;
}

static iree_status_t loom_aie2p_leaf_object_copy_resources(
    const loom_aie2p_leaf_program_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_realization_t* realization) {
  const iree_host_size_t resource_count = plan->resource_import_count;
  if (resource_count == 0) {
    return iree_ok_status();
  }

  loom_aie2p_leaf_resource_import_t* resources = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*resources), (void**)&resources));
  memcpy(resources, plan->resource_imports,
         resource_count * sizeof(*resources));
  realization->resource_imports = resources;
  realization->resource_import_count = resource_count;
  realization->capability_flags |=
      LOOM_AIE2P_LEAF_CAPABILITY_FLAG_RESOURCE_IMPORTS;
  return iree_ok_status();
}

static iree_status_t loom_aie2p_leaf_object_copy_code_name(
    iree_string_view_t function_name, iree_arena_allocator_t* arena,
    iree_string_view_t* out_section_name, iree_string_view_t* out_symbol_name) {
  const iree_string_view_t prefix = IREE_SV(".text.");
  iree_host_size_t section_name_length = 0;
  if (!iree_host_size_checked_add(prefix.size, function_name.size,
                                  &section_name_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P leaf section name is too long");
  }
  char* section_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, section_name_length,
                                           (void**)&section_name_data));
  memcpy(section_name_data, prefix.data, prefix.size);
  memcpy(section_name_data + prefix.size, function_name.data,
         function_name.size);
  *out_section_name =
      iree_make_string_view(section_name_data, section_name_length);
  *out_symbol_name = iree_make_string_view(section_name_data + prefix.size,
                                           function_name.size);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_leaf_object_copy_storage_name(
    iree_string_view_t function_name, loom_storage_space_t storage_space,
    iree_arena_allocator_t* arena, iree_string_view_t* out_section_name,
    iree_string_view_t* out_symbol_name) {
  const iree_string_view_t prefix = IREE_SV(".storage.");
  const iree_string_view_t separator = IREE_SV(".");
  const iree_string_view_t space_name =
      loom_aie2p_leaf_object_storage_space_name(storage_space);
  iree_host_size_t section_name_length = 0;
  if (!iree_host_size_checked_add(prefix.size, function_name.size,
                                  &section_name_length) ||
      !iree_host_size_checked_add(section_name_length, separator.size,
                                  &section_name_length) ||
      !iree_host_size_checked_add(section_name_length, space_name.size,
                                  &section_name_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P storage section name is too long");
  }
  char* section_name_data = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(arena, section_name_length,
                                           (void**)&section_name_data));
  iree_host_size_t offset = 0;
  memcpy(section_name_data + offset, prefix.data, prefix.size);
  offset += prefix.size;
  memcpy(section_name_data + offset, function_name.data, function_name.size);
  offset += function_name.size;
  memcpy(section_name_data + offset, separator.data, separator.size);
  offset += separator.size;
  memcpy(section_name_data + offset, space_name.data, space_name.size);
  IREE_ASSERT_EQ(offset + space_name.size, section_name_length);
  *out_section_name =
      iree_make_string_view(section_name_data, section_name_length);
  *out_symbol_name = iree_make_string_view(section_name_data + prefix.size,
                                           section_name_length - prefix.size);
  return iree_ok_status();
}

iree_status_t loom_aie2p_leaf_object_emit(
    const loom_aie2p_leaf_program_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_leaf_contribution_t* out_contribution) {
  IREE_ASSERT_ARGUMENT(plan);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_contribution);
  *out_contribution = (loom_aie2p_leaf_contribution_t){0};

  uint8_t* code = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, plan->encoded_byte_length, (void**)&code));
  iree_host_size_t code_offset = 0;
  for (iree_host_size_t i = 0; i < plan->bundle_count; ++i) {
    const loom_aie2p_planned_bundle_t* bundle = &plan->bundles[i];
    loom_aie2p_encoded_slot_t
        encoded_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    for (uint8_t j = 0; j < bundle->slot_count; ++j) {
      encoded_slots[j] = plan->slots[bundle->slot_start + j].encoded_slot;
    }
    loom_aie2p_encoding_packet_t packet;
    IREE_RETURN_IF_ERROR(loom_aie2p_encoding_pack_bundle(
        bundle->format, encoded_slots, bundle->slot_count, &packet));
    // The shortest AIE2P NOP is the all-zero 16-bit bundle. Timing gaps stay
    // sparse in the plan and become these words directly in the code section.
    memset(code + code_offset, 0, bundle->byte_offset - code_offset);
    code_offset = bundle->byte_offset;
    IREE_ASSERT(code_offset + packet.data_length <= plan->encoded_byte_length);
    memcpy(code + code_offset, packet.data, packet.data_length);
    code_offset += packet.data_length;
  }
  IREE_ASSERT(code_offset == plan->encoded_byte_length);

  loom_aie2p_leaf_realization_t* realization = &out_contribution->realization;
  *realization = (loom_aie2p_leaf_realization_t){
      .register_writes = plan->register_writes,
      .target_identity = LOOM_AIE2P_LEAF_TARGET_IDENTITY,
      .abi_identity = LOOM_AIE2P_LEAF_ABI_IDENTITY,
      .entry_symbol_index = 0,
      .elf_machine = IREE_XDNA_ELF_MACHINE_AIE,
      .target_generation = IREE_XDNA_TARGET_GENERATION_AIE2P,
      .elf_flags = IREE_XDNA_ELF_AIE2P_FLAGS,
      .code =
          {
              .byte_length = plan->encoded_byte_length,
              .minimum_alignment = 16,
          },
  };
  loom_aie2p_leaf_object_copy_storage_requirements(plan, realization);

  iree_host_size_t storage_domain_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    if (loom_aie2p_leaf_storage_requirement(realization, kStorageSpaceOrder[i])
            ->byte_length != 0) {
      ++storage_domain_count;
    }
  }
  const iree_host_size_t section_count = storage_domain_count + 1u;
  loom_native_section_contribution_t* sections = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, section_count, sizeof(*sections), (void**)&sections));
  loom_native_object_symbol_t* symbols = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, section_count, sizeof(*symbols), (void**)&symbols));
  loom_aie2p_leaf_storage_domain_t* storage_domains = NULL;
  if (storage_domain_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, storage_domain_count,
                                                   sizeof(*storage_domains),
                                                   (void**)&storage_domains));
  }

  const iree_string_view_t function_name = plan->function_name;
  iree_string_view_t code_section_name;
  iree_string_view_t entry_symbol_name;
  IREE_RETURN_IF_ERROR(loom_aie2p_leaf_object_copy_code_name(
      function_name, arena, &code_section_name, &entry_symbol_name));
  sections[0] = (loom_native_section_contribution_t){
      .section_name = code_section_name,
      .section_type = LOOM_NATIVE_ELF_SECTION_TYPE_PROGBITS,
      .section_flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                       LOOM_NATIVE_ELF_SECTION_FLAG_EXECINSTR,
      .contribution_alignment = realization->code.minimum_alignment,
      .contents = iree_make_const_byte_span(code, plan->encoded_byte_length),
  };
  symbols[0] = (loom_native_object_symbol_t){
      .name = entry_symbol_name,
      .section_contribution_index = 0,
      .section_offset = 0,
      .size = plan->encoded_byte_length,
      .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_GLOBAL,
      .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_DEFAULT,
      .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_FUNCTION,
  };

  uint32_t storage_symbol_indices[LOOM_STORAGE_SPACE_COUNT_];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(storage_symbol_indices);
       ++i) {
    storage_symbol_indices[i] = UINT32_MAX;
  }
  iree_host_size_t domain_index = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    const loom_storage_space_t storage_space = kStorageSpaceOrder[i];
    const loom_aie2p_leaf_storage_requirement_t* requirement =
        loom_aie2p_leaf_storage_requirement(realization, storage_space);
    if (requirement->byte_length == 0) {
      continue;
    }
    IREE_ASSERT_GT(requirement->minimum_alignment, 0u);
    const uint32_t section_index = (uint32_t)(domain_index + 1u);
    iree_string_view_t storage_section_name;
    iree_string_view_t storage_symbol_name;
    IREE_RETURN_IF_ERROR(loom_aie2p_leaf_object_copy_storage_name(
        function_name, storage_space, arena, &storage_section_name,
        &storage_symbol_name));
    // NOBITS keeps uninitialized function storage compact. The retained
    // FUNCTION_STORAGE capability distinguishes it from semantic zero-fill.
    sections[section_index] = (loom_native_section_contribution_t){
        .section_name = storage_section_name,
        .section_type = LOOM_NATIVE_ELF_SECTION_TYPE_NOBITS,
        .section_flags = LOOM_NATIVE_ELF_SECTION_FLAG_ALLOC |
                         LOOM_NATIVE_ELF_SECTION_FLAG_WRITE,
        .contribution_alignment = requirement->minimum_alignment,
        .zero_fill_length = requirement->byte_length,
    };
    symbols[section_index] = (loom_native_object_symbol_t){
        .name = storage_symbol_name,
        .section_contribution_index = section_index,
        .section_offset = 0,
        .size = requirement->byte_length,
        .binding = LOOM_NATIVE_OBJECT_SYMBOL_BINDING_LOCAL,
        .visibility = LOOM_NATIVE_OBJECT_SYMBOL_VISIBILITY_HIDDEN,
        .kind = LOOM_NATIVE_OBJECT_SYMBOL_KIND_DATA,
    };
    storage_domains[domain_index] = (loom_aie2p_leaf_storage_domain_t){
        .storage_space = storage_space,
        .section_contribution_index = section_index,
        .symbol_index = section_index,
    };
    storage_symbol_indices[storage_space] = section_index;
    ++domain_index;
  }
  IREE_ASSERT_EQ(domain_index, storage_domain_count);
  realization->storage_domains = storage_domains;
  realization->storage_domain_count = storage_domain_count;

  iree_host_size_t fixup_count = 0;
  if (!iree_host_size_checked_add(plan->branch_fixup_count,
                                  plan->storage_fixup_count, &fixup_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AIE2P native fixup count exceeds host size");
  }
  loom_native_object_fixup_t* fixups = NULL;
  if (fixup_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, fixup_count, sizeof(*fixups), (void**)&fixups));
  }
  for (iree_host_size_t i = 0; i < plan->branch_fixup_count; ++i) {
    const loom_aie2p_planned_branch_fixup_t* branch_fixup =
        &plan->branch_fixups[i];
    IREE_ASSERT_LT(branch_fixup->bundle_index, plan->bundle_count);
    IREE_ASSERT_LT(branch_fixup->target_block_index, plan->block_count);
    fixups[i] = (loom_native_object_fixup_t){
        .section_contribution_index = 0,
        .section_offset = plan->bundles[branch_fixup->bundle_index].byte_offset,
        .relocation_kind =
            LOOM_AIE2P_NATIVE_RELOCATION_KIND_CORE_BRANCH_ABSOLUTE,
        .target_symbol_index = 0,
        .addend = plan->block_byte_offsets[branch_fixup->target_block_index],
    };
  }
  for (iree_host_size_t i = 0; i < plan->storage_fixup_count; ++i) {
    const loom_aie2p_planned_storage_fixup_t* storage_fixup =
        &plan->storage_fixups[i];
    IREE_ASSERT_LT(storage_fixup->bundle_index, plan->bundle_count);
    IREE_ASSERT(loom_storage_space_is_valid(storage_fixup->storage_space));
    const uint32_t target_symbol_index =
        storage_symbol_indices[storage_fixup->storage_space];
    IREE_ASSERT(target_symbol_index != UINT32_MAX &&
                "planned storage address must reference a placement domain");
    fixups[plan->branch_fixup_count + i] = (loom_native_object_fixup_t){
        .section_contribution_index = 0,
        .section_offset =
            plan->bundles[storage_fixup->bundle_index].byte_offset,
        .relocation_kind =
            LOOM_AIE2P_NATIVE_RELOCATION_KIND_LOCAL_ADDRESS_ABSOLUTE,
        .target_symbol_index = target_symbol_index,
        .addend = (int64_t)storage_fixup->byte_offset,
    };
  }

  out_contribution->object = (loom_native_object_contribution_t){
      .sections = sections,
      .section_count = section_count,
      .symbols = symbols,
      .symbol_count = section_count,
      .fixups = fixups,
      .fixup_count = fixup_count,
  };
  IREE_RETURN_IF_ERROR(
      loom_aie2p_leaf_object_copy_resources(plan, arena, realization));
  if (storage_domain_count != 0) {
    realization->capability_flags |=
        LOOM_AIE2P_LEAF_CAPABILITY_FLAG_FUNCTION_STORAGE;
  }
  if (realization->spill.byte_length != 0) {
    realization->capability_flags |=
        LOOM_AIE2P_LEAF_CAPABILITY_FLAG_MATERIALIZED_SPILLS;
  }
  if (out_contribution->object.fixup_count != 0) {
    realization->capability_flags |=
        LOOM_AIE2P_LEAF_CAPABILITY_FLAG_NATIVE_FIXUPS;
  }
  return iree_ok_status();
}
