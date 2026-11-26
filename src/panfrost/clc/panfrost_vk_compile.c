/*
 * Copyright 2024 Collabora Ltd
 * SPDX-License-Identifier: MIT
 *
 * Standalone Vulkan SPIR-V -> Valhall/Bifrost ISA compiler.
 *
 * This is a thin, self-contained driver modelled on panvk's shader compile
 * path (src/panfrost/vulkan/panvk_vX_shader.c) and the OpenCL-only
 * pan_compile.c.  It links directly against libpanfrost_compiler + libnir +
 * spirv_to_nir and therefore has no dependency on DRM, X11, Wayland or
 * libvulkan.  It accepts Vulkan SPIR-V for the Vertex, Fragment and Compute
 * stages and emits a raw Valhall/Bifrost binary blob.
 *
 * The tool is specialized for a single Valhall generation at a time via the
 * PAN_ARCH compile-time define (valid 9..15, default 11), mirroring how panvk
 * compiles panvk_vX_shader.c once per arch.  The pass ordering follows
 * panvk_preprocess_nir -> panvk_lower_nir -> panvk_lower_nir_io ->
 * panvk_compile_nir as closely as possible for v9+.
 *
 * The descriptor handling is intentionally minimal: it does not know about
 * VkDescriptorSetLayout, so descriptors are collapsed into packed immediates
 * instead of the full panvk descriptor-lowering machinery.  Likewise the
 * sysval and push-constant packing is simplified (no panvk_shader_variant).
 */

#include "compiler/glsl_types.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/compiler/pan_nir.h"
#include "shader_enums.h"
#include "util/macros.h"
#include "util/u_dynarray.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PAN_ARCH
#define PAN_ARCH 11
#endif

#if PAN_ARCH < 9 || PAN_ARCH > 15
#error "panfrost_vk_compile only supports PAN_ARCH in the range 9..15"
#endif

/* SPIR-V header word holding the version (major/minor in the low 16 bits). */
#define SPIRV_VERSION_WORD 1
#define SPIRV_VERSION_1_3 0x00010300u

/* glsl_type_size: number of attribute slots a type occupies, matching the
 * function panvk passes to nir_lower_io().
 */
static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* --- Ported verbatim (modulo panvk types) from panvk_vX_shader.c --------- */

/* NOTE: panvk lowers UBO/SSBO with nir_address_format_vec2_index_32bit_offset
 * and then runs valhall_lower_get_ssbo_size() + valhall_pack_buf_idx() to
 * fold the array size into the index and pack it back to a scalar.  That
 * requires the full panvk descriptor lowering to produce a 2-component
 * resource index.  Our simplified descriptor collapse below yields a
 * 1-component index, so we use nir_address_format_32bit_index_offset and skip
 * those two helpers. */

/* --- Simplified descriptor / sysval lowering (no panvk state) ------------ */

/* Collapse Vulkan descriptor intrinsics into packed immediates and drop the
 * texture/sampler derefs that the standalone flow cannot resolve.  The packed
 * descriptor follows the Valhall LD_PKA/LEA_PKA layout:
 *   ((4 + set) << 24) | binding
 */
static bool
lower_vk_descriptors_and_tex(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
         uint32_t set = nir_intrinsic_desc_set(intr);
         uint32_t binding = nir_intrinsic_binding(intr);
         uint32_t packed = ((4u + set) << 24) | (binding & 0xFFFFu);

         b->cursor = nir_before_instr(instr);
         nir_def *c;
         if (intr->def.num_components == 1)
            c = nir_imm_int(b, packed);
         else
            c = nir_imm_ivec2(b, packed, 0);

         nir_def_rewrite_uses(&intr->def, c);
         nir_instr_remove(instr);
         return true;
      }

      if (intr->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
         nir_def_rewrite_uses(&intr->def, intr->src[0].ssa);
         nir_instr_remove(instr);
         return true;
      }

      if (intr->intrinsic == nir_intrinsic_load_push_constant) {
         /* bi_emit_load_push_constant asserts base == 0 && range == 0; panvk
          * zeroes these after packing, so mirror that here. */
         nir_intrinsic_set_base(intr, 0);
         nir_intrinsic_set_range(intr, 0);
         return true;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      for (int i = (int)tex->num_srcs - 1; i >= 0; --i) {
         if (tex->src[i].src_type == nir_tex_src_texture_deref ||
             tex->src[i].src_type == nir_tex_src_sampler_deref) {
            nir_tex_instr_remove_src(tex, i);
         }
      }
      return true;
   }

   return false;
}

/* pan_postprocess_nir() injects viewport-transform sysvals for gl_Position, and
 * nir_lower_compute_system_values() (with has_base_workgroup_id) leaves a
 * load_base_workgroup_id behind.  Lower them to neutral values so the backend
 * does not see unhandled intrinsics (panvk does this in panvk_lower_sysvals). */
static bool
lower_post_sysvals(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_viewport_scale ||
       intr->intrinsic == nir_intrinsic_load_viewport_offset) {
      b->cursor = nir_before_instr(instr);
      float v = (intr->intrinsic == nir_intrinsic_load_viewport_scale) ? 1.0f
                                                                       : 0.0f;
      nir_def *c = nir_imm_vec3(b, v, v, v);
      nir_def_rewrite_uses(&intr->def, c);
      nir_instr_remove(instr);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_base_workgroup_id) {
      /* The base workgroup id is always 0. */
      b->cursor = nir_before_instr(instr);
      nir_def *c = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
      nir_def_rewrite_uses(&intr->def, c);
      nir_instr_remove(instr);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_load_noperspective_varyings_pan) {
      /* No noperspective varyings in the standalone flow. */
      b->cursor = nir_before_instr(instr);
      nir_def *c = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
      nir_def_rewrite_uses(&intr->def, c);
      nir_instr_remove(instr);
      return true;
   }

   return false;
}

/* --- SPIR-V front-end helpers ------------------------------------------- */

/* Read a NUL-terminated string packed 4 chars per word starting at word
 * index `start`.  Returns the offset just past the terminator. */
static unsigned
read_packed_string(const uint32_t *words, unsigned start, char *out,
                   size_t out_sz)
{
   size_t n = 0;
   unsigned i = start;
   for (;; i++) {
      uint32_t w = words[i];
      for (int c = 0; c < 4; c++) {
         char ch = (char)((w >> (c * 8)) & 0xff);
         if (ch == '\0')
            return i + 1;
         if (n + 1 < out_sz)
            out[n++] = ch;
      }
   }
}

/* Detect the shader stage and entry point name from the first OpEntryPoint. */
static mesa_shader_stage
detect_stage(const uint32_t *words, size_t word_count, char *entry,
             size_t entry_sz)
{
   mesa_shader_stage stage = MESA_SHADER_COMPUTE;
   strcpy(entry, "main");

   unsigned i = 5; /* skip magic, version, generator, bound, schema */
   while (i < word_count) {
      uint32_t inst = words[i];
      unsigned wc = inst >> 16;
      unsigned op = inst & 0xffff;
      if (wc == 0 || i + wc > word_count)
         break;

      if (op == SpvOpEntryPoint) {
         SpvExecutionModel model = (SpvExecutionModel)words[i + 1];
         /* word[i+2] is the entry-point function id; the name string starts
          * at word[i+3]. */
         read_packed_string(words, i + 3, entry, entry_sz);

         switch (model) {
         case SpvExecutionModelVertex:
            stage = MESA_SHADER_VERTEX;
            break;
         case SpvExecutionModelFragment:
            stage = MESA_SHADER_FRAGMENT;
            break;
         case SpvExecutionModelGLCompute:
         case SpvExecutionModelKernel:
            stage = MESA_SHADER_COMPUTE;
            break;
         default:
            fprintf(stderr,
                    "panfrost_vk_compile: unsupported execution model %d\n",
                    (int)model);
            stage = MESA_SHADER_COMPUTE;
            break;
         }
         return stage;
      }

      i += wc;
   }

   return stage;
}

static uint32_t *
read_spirv(const char *path, size_t *size_out)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "panfrost_vk_compile: cannot open %s\n", path);
      return NULL;
   }

   fseek(f, 0, SEEK_END);
   long sz = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (sz < 0) {
      fclose(f);
      return NULL;
   }

   uint32_t *words = malloc(sz);
   if (!words) {
      fclose(f);
      return NULL;
   }

   if (fread(words, 1, sz, f) != (size_t)sz) {
      fprintf(stderr, "panfrost_vk_compile: short read on %s\n", path);
      free(words);
      fclose(f);
      return NULL;
   }
   fclose(f);

   *size_out = (size_t)sz;
   return words;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr,
              "Usage: %s <input.spv> <output.bin>\n"
              "  (built for Valhall arch v%d)\n",
              argv[0], PAN_ARCH);
      return 1;
   }

   /* Default to the Bifrost/Valhall C backend.  The Rust Kraid backend is the
    * default for arch >= 9, but its standalone NIR path rejects the
    * lea_attr_pan intrinsic emitted by the vertex-input lowering, so vertex
    * shaders would abort.  Respect an explicit PAN_USE_KRAID if the caller
    * set one (overwrite == 0). */
   setenv("PAN_USE_KRAID", "0", 0);

   size_t spirv_size = 0;
   uint32_t *words = read_spirv(argv[1], &spirv_size);
   if (!words)
      return 1;

   if (spirv_size < 20 || words[0] != 0x07230203u) {
      fprintf(stderr, "panfrost_vk_compile: %s is not a SPIR-V binary\n",
              argv[1]);
      free(words);
      return 1;
   }

   /* SPIR-V 1.4+ requires StorageBuffer/Uniform OpVariables to be listed in
    * the OpEntryPoint interface.  Mali DDK tolerates their omission while Mesa
    * skips unlisted globals, so clamp the advertised version to 1.3 to keep
    * the interface list optional. */
   if (words[SPIRV_VERSION_WORD] > SPIRV_VERSION_1_3)
      words[SPIRV_VERSION_WORD] = SPIRV_VERSION_1_3;

   char entry[64];
   mesa_shader_stage stage =
      detect_stage(words, spirv_size / 4, entry, sizeof(entry));

   glsl_type_singleton_init_or_ref();

   /* NULL capabilities => use the full implemented_capabilities set.  The
    * address formats match panvk_get_spirv_options() for v9+, except that we
    * use the 1-component 32bit_index_offset for UBO/SSBO (see the note above
    * the descriptor lowering). */
   struct spirv_to_nir_options sopts = {
      .environment = NIR_SPIRV_VULKAN,
      .capabilities = NULL,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ubo_alignment = 16,
      .min_ssbo_alignment = 16,
   };

   const nir_shader_compiler_options *nopts =
      pan_get_nir_shader_compiler_options(PAN_ARCH, stage, false);

   nir_shader *nir =
      spirv_to_nir(words, spirv_size / 4, NULL, stage, entry, &sopts, nopts);
   free(words);
   if (!nir) {
      fprintf(stderr, "panfrost_vk_compile: spirv_to_nir failed\n");
      glsl_type_singleton_decref();
      return 1;
   }

   /* Early lowering.  The Vulkan driver performs the equivalent before
    * create_shader(); we need it here to handle multi-function shaders. */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_all);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);

   /* --- panvk_preprocess_nir (v9+ specialization) ---------------------- */
   if (pan_use_kraid(PAN_ARCH, stage, false))
      NIR_PASS(_, nir, nir_lower_mediump_vars,
               nir_var_shader_temp | nir_var_function_temp);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_vectorize_io_vars, nir_var_shader_out);

   /* NOTE: panvk runs nir_lower_io_vars_to_temporaries(entrypoint, out) here,
    * but that pass asserts on Vulkan interface blocks (e.g. gl_PerVertex).
    * nir_lower_io below handles blocks directly, so we skip it. */

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_combine_stores, nir_var_all);
   NIR_PASS(_, nir, nir_opt_loop);
   NIR_PASS(_, nir, nir_opt_barrier_modes);
   NIR_PASS(_, nir, nir_opt_acquire_release_barriers, SCOPE_DEVICE);

   NIR_PASS(_, nir, nir_lower_system_values);

   nir_lower_compute_system_values_options cs_opts = {
      .has_base_workgroup_id = true,
      .shuffle_local_ids_for_quad_derivatives = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &cs_opts);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_lower_wpos_center);

   pan_preprocess_nir(nir, PAN_ARCH);

   /* --- panvk_lower_nir (v9+ specialization) --------------------------- */
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   /* Full panvk descriptor lowering needs VkDescriptorSetLayout; collapse
    * descriptors to packed immediates instead.  This runs where panvk does
    * descriptor lowering (after pan_preprocess_nir's nir_lower_tex, before
    * explicit IO). */
   NIR_PASS(_, nir, nir_shader_instructions_pass,
            lower_vk_descriptors_and_tex, nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);

   if (mesa_shader_stage_uses_workgroup(stage)) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               glsl_get_cl_type_size_align);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* --- panvk_lower_nir_io (VS/FS only) -------------------------------- */
   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_FRAGMENT) {
      /* Assign driver locations so varyings get distinct bases (panvk does
       * this from the vertex-input / fragment state). */
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);

      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
      NIR_PASS(_, nir, nir_lower_io,
               nir_var_shader_in | nir_var_shader_out, glsl_type_size,
               stage == MESA_SHADER_FRAGMENT
                  ? nir_lower_io_use_interpolated_input_intrinsics
                  : 0);

      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   /* --- panvk_compile_nir ---------------------------------------------- */
   struct pan_compile_inputs inputs = {
      .arch = PAN_ARCH,
      .model = NULL, /* only the arch is known when compiling offline */
      .no_idvs = true,
      .fau = { .reserved = 64, .promote_immediates = true },
   };

   struct pan_varying_layout vlayout = {0};
   if (stage == MESA_SHADER_VERTEX) {
      pan_varying_collect_formats(&vlayout, nir, PAN_ARCH, false);
      pan_build_varying_layout_compact(&vlayout, nir, PAN_ARCH);
      inputs.varying_layout = &vlayout;
   }

   struct pan_shader_info info = {0};
   pan_postprocess_nir(nir, &inputs, &info);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_post_sysvals,
            nir_metadata_control_flow, NULL);

   struct util_dynarray binary;
   util_dynarray_init(&binary, NULL);
   pan_shader_compile(nir, &inputs, &binary, &info);

   FILE *out = fopen(argv[2], "wb");
   if (!out) {
      fprintf(stderr, "panfrost_vk_compile: cannot open %s for writing\n",
              argv[2]);
      util_dynarray_fini(&binary);
      ralloc_free(nir);
      glsl_type_singleton_decref();
      return 1;
   }

   fwrite(binary.data, 1, binary.size, out);
   fclose(out);

   uint32_t instrs = info.stats.valhall.instrs;
   fprintf(stderr,
           "panfrost_vk_compile: %s (%s) -> %s: %u bytes, %u instructions\n",
           entry, stage == MESA_SHADER_VERTEX   ? "vertex"
                   : stage == MESA_SHADER_FRAGMENT ? "fragment"
                                                   : "compute",
           argv[2], (unsigned)binary.size, instrs);

   util_dynarray_fini(&binary);
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return 0;
}