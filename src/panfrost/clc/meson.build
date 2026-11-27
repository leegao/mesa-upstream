# Copyright 2017 Intel Corporation
# Copyright 2024 Collabora
# SPDX-License-Identifier: MIT


if get_option('precomp-compiler') == 'system'
  prog_panfrost_compile = find_program('panfrost_compile', native : true)
else
  prog_panfrost_compile = executable(
    'panfrost_compile',
    ['pan_compile.c'],
    link_with : [libpanfrost_compiler],
    include_directories : [inc_include, inc_src],
    c_args : [c_msvc_compat_args, no_override_init_args, soong_compat_c_args],
    link_args : [ld_args_build_id],
    dependencies : [idep_vtn, idep_nir, idep_mesautil, libpanfrost_dep],
    # If we can run host binaries directly, just build panfrost_compile for the host.
    # Most commonly this happens when doing a cross compile from an x86_64 build
    # machine to an x86 host
    native : not meson.can_run_host_binaries(),
    install : get_option('install-precomp-compiler'),
  )
endif

# Standalone Vulkan SPIR-V -> Valhall/Bifrost ISA compiler.  Unlike
# panfrost_compile (OpenCL-only), this drives spirv_to_nir with
# NIR_SPIRV_VULKAN and links the full panfrost compiler backend, so it can
# compile Vertex/Fragment/Compute Vulkan shaders offline with no DRM/Wayland/
# libvulkan/libpanvk dependency.
#
# panvk_nir_lower_cooperative_matrix.c is a pure NIR pass that only depends on
# nir_builder.h, util/hash_table.h, and panvk_nir.h; compile it directly into
# the standalone executable without linking libpanvk.
#
# The tool is specialized for a single Valhall generation at a time via the
# PAN_ARCH compile-time define (valid 9..15, default 11).  To build for a
# different generation, change pan_vk_arch below.
pan_vk_arch = 11

libpanfrost_vk_compile = static_library(
  'panfrost_vk_compile',
  [
    'panfrost_vk_compile.c',
    '../vulkan/panvk_nir_lower_cooperative_matrix.c',
  ],
  include_directories : [
    inc_include,
    inc_src,
    inc_panfrost,
    include_directories('../vulkan'),
  ],
  c_args : [
    c_msvc_compat_args, no_override_init_args, soong_compat_c_args,
  ],
  dependencies : [idep_vtn, idep_nir, idep_mesautil, libpanfrost_dep],
  # Meson builds thin archives by default (they only reference the build-dir
  # .o files by path), which makes the shipped .a useless outside the build
  # tree.  Marking the library as installable forces a regular, self-contained
  # archive so it can be reused from the release tarball.
  install : true,
)

panfrost_vk_compile = executable(
  'panfrost_vk_compile',
  ['panfrost_vk_compile_cli.c'],
  include_directories : [
    inc_include,
    inc_src,
    inc_panfrost,
  ],
  link_with : [libpanfrost_vk_compile, libpanfrost_compiler],
  c_args : [
    c_msvc_compat_args, no_override_init_args, soong_compat_c_args,
  ],
  link_args : [ld_args_build_id],
  dependencies : [idep_vtn, idep_nir, idep_mesautil, libpanfrost_dep],
  build_by_default : true,
)
