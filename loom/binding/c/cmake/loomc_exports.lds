{
  global:
    loomc_*;
    /* Revision-coupled compiler interfaces used by external HAL adapters.
       Wildcards permit builds with the corresponding target disabled. */
    loom_aie2p_target_profile_select*;
    loom_aie2p_target_profile_type*;
    loom_amdgpu_artifact_key_append*;
    loom_amdgpu_target_info_find_target*;
    loom_amdgpu_target_info_is_generic*;
    loom_amdgpu_target_info_target_processor*;
    loom_amdgpu_target_profile_type*;
    loom_sanitizer_runtime_requirements_query*;
    loom_target_environment_lookup_profile_provider*;
    loom_target_environment_select_cpu_profile*;
    loom_target_profile_selection_release*;
  local:
    *;
};
