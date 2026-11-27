/*
 * Copyright 2024 Collabora Ltd
 * SPDX-License-Identifier: MIT
 *
 * Standalone Vulkan SPIR-V -> Valhall/Bifrost ISA compiler.
 *
 * This is a thin, self-contained driver modelled on panvk's shader compile
 * path (src/panfrost/vulkan/panvk_vX_shader.c) and pan_compile.c.  It links
 * directly against libpanfrost_compiler + libnir + spirv_to_nir (plus the
 * pure-NIR panvk_nir_lower_cooperative_matrix.c) and has no dependency on
 * DRM, panthor, libvulkan, or libpanvk.
 *
 * To match on-device panvk v11 and align with Mali DDK (libMali r55p0
 * Mali-G715, VEHW=0x1c/0x1d0000):
 *  - IDVS is enabled by default (no_idvs = false), matching both panvk on v9+
 *    and libMali's VERTEX POSITION SHADER, and eliminating the lea_attr_pan
 *    intrinsic so the Rust Kraid backend can compile VS/FS/CS.
 *  - Cooperative matrix lowering runs with pan_subgroup_size(PAN_ARCH) (16 on
 *    v9+).
 *  - Descriptor lowering infers a synthetic VkDescriptorSetLayout directly from
 *    the NIR shader, using nir_address_format_vec2_index_32bit_offset,
 *    valhall_lower_get_ssbo_size (table 62), and valhall_pack_buf_idx.
 *  - Optional --robust enables nir_lower_robust_access + bounds checks (off by
 *    default, matching libMali's cmpbe_has_robustness == 0).
 *  - panvk_can_merge_workgroups + panvk_lower_non_uniform handle workgroup
 *    merging and non-uniform descriptor indexing.
 *  - panvk_lower_sysvals + lower_load_push_consts pack sysvals (including
 *    viewport transform and FS 4-word blend constants) and user push constants
 *    into FAU, setting inputs.fau.reserved = total_count * 2 instead of a
 *    hardcoded 64.
 */
#include "panfrost_vk_compile.h"

#include "compiler/glsl_types.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "panfrost/compiler/bifrost/bifrost_compile.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/compiler/pan_nir.h"
#include "panfrost/vulkan/panvk_nir.h"
#include "shader_enums.h"
#include "util/bitset.h"
#include "util/macros.h"
#include "util/u_dynarray.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* SPIR-V header word holding the version (major/minor in the low 16 bits). */
#define SPIRV_VERSION_WORD 1
#define SPIRV_VERSION_1_3 0x00010300u

#define PANVK_DESCRIPTOR_SIZE 32u
#define MAX_SETS 7u
#define MAX_BINDINGS_PER_SET 128u
#define MAX_RTS 8u
#define MAX_PUSH_CONSTANTS_SIZE 256u
#define FAU_WORD_SIZE sizeof(uint64_t)
#define FAU_WORD_COUNT 64u
#define SYSVALS_PUSH_CONST_BASE MAX_PUSH_CONSTANTS_SIZE

/* glsl_type_size: number of attribute slots a type occupies, matching the
 * function panvk passes to nir_lower_io().
 */
static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
shared_type_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length;
   *align = comp_size * (length == 3 ? 4 : length);
}

/* --- Standalone panvk FAU / sysval layout (from panvk_shader.h) --------- */

typedef uint64_t aligned_u64 __attribute__((aligned(8)));

struct panvk_common_sysvals_inner {
   aligned_u64 push_uniforms;
   aligned_u64 printf_buffer_address;
   aligned_u64 constant_data;
} __attribute__((aligned(FAU_WORD_SIZE)));

struct panvk_common_sysvals {
   uint32_t _pad[4];
   struct panvk_common_sysvals_inner common;
} __attribute__((aligned(FAU_WORD_SIZE)));

struct panvk_graphics_sysvals {
   struct {
      float constants[4];
   } blend;

   struct panvk_common_sysvals_inner common;

   struct {
      struct {
         float x, y, z;
      } scale, offset;
   } viewport;

   struct {
      uint64_t vertex_params;
      uint64_t geometry_params;
      uint64_t pipeline_stats;
      uint32_t pipeline_stats_flags;
      uint32_t rasterization_stream;
      uint32_t is_api_gs;
      uint32_t provoking_vertex;
   } poly;

   struct {
      int32_t first_vertex;
      int32_t base_instance;
      uint32_t noperspective_varyings;
   } vs;

   struct {
      aligned_u64 blend_descs[MAX_RTS];
   } fs;

   struct {
      uint32_t target;
      uint32_t conv;
   } iam[MAX_RTS + 1];
} __attribute__((aligned(FAU_WORD_SIZE)));

struct panvk_compute_sysvals {
   struct {
      uint32_t x, y, z;
   } base;

   uint32_t _pad;

   struct panvk_common_sysvals_inner common;

   struct {
      uint32_t x, y, z;
   } num_work_groups;
   struct {
      uint32_t x, y, z;
   } local_group_size;
} __attribute__((aligned(FAU_WORD_SIZE)));

#define common_sysval_size(__name) \
   sizeof(((struct panvk_common_sysvals *)NULL)->common.__name)
#define graphics_sysval_size(__name) \
   sizeof(((struct panvk_graphics_sysvals *)NULL)->__name)
#define compute_sysval_size(__name) \
   sizeof(((struct panvk_compute_sysvals *)NULL)->__name)
#define sysval_size(__ptype, __name) __ptype##_sysval_size(__name)

#define common_sysval_offset(__name) \
   offsetof(struct panvk_common_sysvals, common.__name)
#define graphics_sysval_offset(__name) \
   offsetof(struct panvk_graphics_sysvals, __name)
#define compute_sysval_offset(__name) \
   offsetof(struct panvk_compute_sysvals, __name)
#define sysval_offset(__ptype, __name) __ptype##_sysval_offset(__name)

#define sysval_entry_size(__ptype, __name) \
   sizeof(((struct panvk_##__ptype##_sysvals *)NULL)->__name[0])
#define sysval_entry_offset(__ptype, __name, __idx) \
   (sysval_offset(__ptype, __name) +                \
    (sysval_entry_size(__ptype, __name) * (__idx)))

#define sysval_fau_start(__ptype, __name) \
   (sysval_offset(__ptype, __name) / FAU_WORD_SIZE)
#define sysval_fau_end(__ptype, __name) \
   ((sysval_offset(__ptype, __name) + sysval_size(__ptype, __name) - 1) / \
    FAU_WORD_SIZE)

#define MAX_COMPUTE_SYSVAL_FAUS \
   (sizeof(struct panvk_compute_sysvals) / FAU_WORD_SIZE)
#define MAX_GFX_SYSVAL_FAUS \
   (sizeof(struct panvk_graphics_sysvals) / FAU_WORD_SIZE)
#define MAX_SYSVAL_FAUS MAX2(MAX_COMPUTE_SYSVAL_FAUS, MAX_GFX_SYSVAL_FAUS)
#define MAX_PUSH_CONST_FAUS (MAX_PUSH_CONSTANTS_SIZE / FAU_WORD_SIZE)

struct standalone_fau_info {
   BITSET_DECLARE(used_sysvals, MAX_SYSVAL_FAUS);
   BITSET_DECLARE(used_push_consts, MAX_PUSH_CONST_FAUS);
   uint32_t sysval_count;
   uint32_t total_count;
};

#define shader_use_sysval(__fau, __ptype, __name) \
   BITSET_SET_RANGE((__fau)->used_sysvals,        \
                    sysval_fau_start(__ptype, __name), \
                    sysval_fau_end(__ptype, __name))

static void
fau_use_range(BITSET_WORD *bitset, unsigned max_words, uint32_t offset,
              uint32_t size)
{
   if (size == 0)
      return;
   uint32_t start = offset / FAU_WORD_SIZE;
   uint32_t end = (offset + size - 1) / FAU_WORD_SIZE;
   assert(end < max_words);
   BITSET_SET_RANGE(bitset, start, end);
}

#define shader_remapped_fau_offset(__fau, __kind, __offset)                  \
   ((FAU_WORD_SIZE *                                                         \
     BITSET_PREFIX_SUM((__fau)->used_##__kind, (__offset) / FAU_WORD_SIZE)) + \
    ((__offset) % FAU_WORD_SIZE))

#define shader_remapped_sysval_offset(__fau, __offset) \
   shader_remapped_fau_offset(__fau, sysvals, __offset)

#define shader_remapped_push_const_offset(__fau, __offset) \
   (((__fau)->sysval_count * FAU_WORD_SIZE) +              \
    shader_remapped_fau_offset(__fau, push_consts, __offset))

#define load_sysval(b, __ptype, __bit_size, __name)                    \
   nir_load_push_constant(                                             \
      b, sysval_size(__ptype, __name) / ((__bit_size) / 8),            \
      __bit_size, nir_imm_int(b, 0),                                   \
      .base = SYSVALS_PUSH_CONST_BASE + sysval_offset(__ptype, __name), \
      .range = sysval_size(__ptype, __name))

#define load_sysval_entry(b, __ptype, __bit_size, __name, __idx)             \
   nir_load_push_constant(                                                   \
      b, sysval_entry_size(__ptype, __name) / ((__bit_size) / 8),            \
      __bit_size,                                                            \
      nir_imul_imm(b, __idx, sysval_entry_size(__ptype, __name)),            \
      .base = SYSVALS_PUSH_CONST_BASE + sysval_offset(__ptype, __name),      \
      .range = sysval_size(__ptype, __name))

/* --- Synthetic descriptor-set layout & v9+ descriptor lowering ---------- */

enum synth_desc_kind {
   SYNTH_DESC_NONE = 0,
   SYNTH_DESC_BUFFER,
   SYNTH_DESC_TEX_OR_IMG,
   SYNTH_DESC_SAMPLER,
   SYNTH_DESC_COMBINED_IMAGE_SAMPLER,
};

struct synth_binding_info {
   enum synth_desc_kind kind;
   uint32_t desc_count;
   uint32_t desc_idx;
};

struct synth_set_layout {
   struct synth_binding_info bindings[MAX_BINDINGS_PER_SET];
   uint32_t max_binding;
   uint32_t total_descs;
};

struct synth_desc_ctx {
   struct synth_set_layout sets[MAX_SETS];
   bool add_bounds_checks;
};

static const char *
synth_desc_kind_name(enum synth_desc_kind kind)
{
   switch (kind) {
   case SYNTH_DESC_BUFFER:
      return "buffer";
   case SYNTH_DESC_TEX_OR_IMG:
      return "tex_or_img";
   case SYNTH_DESC_SAMPLER:
      return "sampler";
   case SYNTH_DESC_COMBINED_IMAGE_SAMPLER:
      return "combined_image_sampler";
   default:
      return "none";
   }
}

static void
record_synth_binding(struct synth_desc_ctx *ctx, uint32_t set, uint32_t binding,
                     enum synth_desc_kind kind, uint32_t count)
{
   if (set >= MAX_SETS || binding >= MAX_BINDINGS_PER_SET)
      return;

   struct synth_binding_info *b = &ctx->sets[set].bindings[binding];
   if (b->kind == SYNTH_DESC_NONE) {
      b->kind = kind;
   } else if ((b->kind == SYNTH_DESC_TEX_OR_IMG &&
               kind == SYNTH_DESC_SAMPLER) ||
              (b->kind == SYNTH_DESC_SAMPLER &&
               kind == SYNTH_DESC_TEX_OR_IMG)) {
      b->kind = SYNTH_DESC_COMBINED_IMAGE_SAMPLER;
   }

   if (count > b->desc_count)
      b->desc_count = count;
   if (binding + 1 > ctx->sets[set].max_binding)
      ctx->sets[set].max_binding = binding + 1;
}

static void
get_deref_binding_info(nir_deref_instr *deref, uint32_t *set_out,
                       uint32_t *binding_out, nir_def **index_ssa_out,
                       uint32_t *max_idx_out)
{
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (deref->deref_type == nir_deref_type_array && parent) {
      *index_ssa_out = deref->arr.index.ssa;
      uint32_t arr_sz = (uint32_t)glsl_array_size(parent->type);
      *max_idx_out = arr_sz > 0 ? arr_sz - 1 : UINT32_MAX;
      deref = parent;
   } else {
      *index_ssa_out = NULL;
      *max_idx_out = 0;
   }

   nir_variable *var = nir_deref_instr_get_variable(deref);
   assert(var != NULL);
   *set_out = var->data.descriptor_set;
   *binding_out = var->data.binding;
}

static bool
collect_synth_desc_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct synth_desc_ctx *ctx = data;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_vulkan_resource_index: {
         uint32_t set = nir_intrinsic_desc_set(intr);
         uint32_t binding = nir_intrinsic_binding(intr);
         record_synth_binding(ctx, set, binding, SYNTH_DESC_BUFFER, 1);
         break;
      }
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_deref_atomic_swap:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_samples: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         uint32_t cnt = max_idx == UINT32_MAX ? UINT32_MAX : max_idx + 1;
         record_synth_binding(ctx, set, binding, SYNTH_DESC_TEX_OR_IMG, cnt);
         break;
      }
      default:
         break;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      int tex_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (tex_idx >= 0) {
         nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_idx].src);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         uint32_t cnt = max_idx == UINT32_MAX ? UINT32_MAX : max_idx + 1;
         record_synth_binding(ctx, set, binding, SYNTH_DESC_TEX_OR_IMG, cnt);
      }
      int smp_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      if (smp_idx >= 0) {
         nir_deref_instr *deref = nir_src_as_deref(tex->src[smp_idx].src);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         uint32_t cnt = max_idx == UINT32_MAX ? UINT32_MAX : max_idx + 1;
         record_synth_binding(ctx, set, binding, SYNTH_DESC_SAMPLER, cnt);
      }
   }

   return false;
}

static void
assign_synth_desc_indices(nir_shader *nir, struct synth_desc_ctx *ctx)
{
   nir_foreach_variable_with_modes(var, nir,
                                   nir_var_uniform | nir_var_mem_ubo |
                                      nir_var_mem_ssbo | nir_var_image) {
      uint32_t cnt = 1;
      if (glsl_type_is_array(var->type)) {
         unsigned sz = glsl_get_aoa_size(var->type);
         cnt = sz > 0 ? sz : UINT32_MAX;
      }
      enum synth_desc_kind kind = SYNTH_DESC_BUFFER;
      const struct glsl_type *leaf = glsl_without_array(var->type);
      if (glsl_type_is_sampler(leaf)) {
         kind = glsl_sampler_type_is_shadow(leaf) ||
                      glsl_get_sampler_dim(leaf) != GLSL_SAMPLER_DIM_BUF
                   ? SYNTH_DESC_COMBINED_IMAGE_SAMPLER
                   : SYNTH_DESC_TEX_OR_IMG;
      } else if (glsl_type_is_image(leaf)) {
         kind = SYNTH_DESC_TEX_OR_IMG;
      }
      record_synth_binding(ctx, var->data.descriptor_set, var->data.binding,
                           kind, cnt);
   }

   nir_shader_instructions_pass(nir, collect_synth_desc_instr,
                                nir_metadata_all, ctx);

   for (uint32_t s = 0; s < MAX_SETS; s++) {
      uint32_t idx = 0;
      for (uint32_t b = 0; b < ctx->sets[s].max_binding; b++) {
         struct synth_binding_info *info = &ctx->sets[s].bindings[b];
         if (info->kind == SYNTH_DESC_NONE)
            continue;
         info->desc_idx = idx;
         uint32_t stride =
            (info->kind == SYNTH_DESC_COMBINED_IMAGE_SAMPLER) ? 2u : 1u;
         uint32_t cnt =
            (info->desc_count == 0 || info->desc_count == UINT32_MAX)
               ? 1u
               : info->desc_count;
         idx += cnt * stride;
      }
      ctx->sets[s].total_descs = idx;
   }
}

static nir_def *
synth_res_handle(nir_builder *b, const struct synth_desc_ctx *ctx,
                 uint32_t set, uint32_t binding, bool is_sampler,
                 nir_def *array_idx)
{
   assert(set < MAX_SETS && binding < MAX_BINDINGS_PER_SET);
   const struct synth_binding_info *info = &ctx->sets[set].bindings[binding];
   uint32_t sub_off =
      (is_sampler && info->kind == SYNTH_DESC_COMBINED_IMAGE_SAMPLER) ? 1u : 0u;
   uint32_t base = pan_res_handle(set + 1, info->desc_idx + sub_off);
   if (!array_idx)
      return nir_imm_int(b, base);

   uint32_t stride =
      (info->kind == SYNTH_DESC_COMBINED_IMAGE_SAMPLER) ? 2u : 1u;
   nir_def *scaled =
      stride > 1 ? nir_imul_imm(b, array_idx, stride) : array_idx;
   return nir_iadd_imm(b, scaled, base);
}

static bool
lower_vk_descriptors_v9(nir_builder *b, nir_instr *instr, void *data)
{
   const struct synth_desc_ctx *ctx = data;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      b->cursor = nir_before_instr(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_vulkan_resource_index: {
         uint32_t set = nir_intrinsic_desc_set(intr);
         uint32_t binding = nir_intrinsic_binding(intr);
         uint32_t desc_idx = 0;
         uint32_t array_size = 1;
         if (set < MAX_SETS && binding < MAX_BINDINGS_PER_SET) {
            const struct synth_binding_info *info =
               &ctx->sets[set].bindings[binding];
            desc_idx = pan_res_handle(set + 1, info->desc_idx);
            if (info->desc_count > 0)
               array_size = info->desc_count;
         } else {
            desc_idx = pan_res_handle(set + 1, binding);
         }

         nir_def *c = nir_vec3(b, nir_imm_int(b, desc_idx), intr->src[0].ssa,
                               nir_imm_int(b, array_size - 1));
         nir_def_replace(&intr->def, c);
         return true;
      }

      case nir_intrinsic_vulkan_resource_reindex: {
         nir_def *old = intr->src[0].ssa;
         nir_def *new_idx =
            nir_iadd(b, nir_channel(b, old, 1), intr->src[1].ssa);
         nir_def *c = nir_vec3(b, nir_channel(b, old, 0), new_idx,
                               nir_channel(b, old, 2));
         nir_def_replace(&intr->def, c);
         return true;
      }

      case nir_intrinsic_load_vulkan_descriptor: {
         nir_def *res_index = intr->src[0].ssa;
         nir_def *first_desc = nir_channel(b, res_index, 0);
         nir_def *array_idx = nir_channel(b, res_index, 1);
         nir_def *array_max = nir_channel(b, res_index, 2);
         if (ctx->add_bounds_checks)
            array_idx = nir_umin(b, array_idx, array_max);
         nir_def *addr =
            nir_vec3(b, first_desc, array_idx, nir_imm_int(b, 0));
         nir_def_replace(&intr->def, addr);
         return true;
      }

      case nir_intrinsic_load_constant: {
         nir_def *offset =
            nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
         nir_def *base = nir_load_constant_base_ptr(b, 1, 64);
         nir_def *addr = nir_iadd(b, base, nir_u2u64(b, offset));
         nir_def *val = nir_load_global_constant(
            b, intr->def.num_components, intr->def.bit_size, addr,
            .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE,
            .align_mul = nir_intrinsic_align_mul(intr),
            .align_offset = nir_intrinsic_align_offset(intr));
         nir_def_replace(&intr->def, val);
         return true;
      }

      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_deref_atomic_swap:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_samples: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         nir_def *handle =
            synth_res_handle(b, ctx, set, binding, false, idx_ssa);
         nir_rewrite_image_intrinsic(intr, handle,
                                     nir_image_intrinsic_type_default);
         return true;
      }

      default:
         return false;
      }
   } else if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      b->cursor = nir_before_instr(instr);

      int tex_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (tex_idx >= 0) {
         nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_idx].src);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         nir_def *handle =
            synth_res_handle(b, ctx, set, binding, false, idx_ssa);
         nir_tex_instr_remove_src(tex, tex_idx);
         nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);
      }

      int smp_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      if (smp_idx >= 0) {
         nir_deref_instr *deref = nir_src_as_deref(tex->src[smp_idx].src);
         uint32_t set, binding, max_idx;
         nir_def *idx_ssa;
         get_deref_binding_info(deref, &set, &binding, &idx_ssa, &max_idx);
         nir_def *handle =
            synth_res_handle(b, ctx, set, binding, true, idx_ssa);
         nir_tex_instr_remove_src(tex, smp_idx);
         nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, handle);
      }
      return tex_idx >= 0 || smp_idx >= 0;
   }

   return false;
}

static bool
is_robust_ssbo_intr(const nir_intrinsic_instr *intr, UNUSED const void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      return true;
   default:
      return false;
   }
}

static bool
valhall_lower_get_ssbo_size(struct nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_get_ssbo_size)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *res_handle = nir_channel(b, intr->src[0].ssa, 0);
   nir_def *table_idx = nir_ushr_imm(b, res_handle, 24);
   nir_def *res_idx = nir_iand_imm(b, res_handle, BITFIELD_MASK(24));
   nir_def *res_table = nir_ior_imm(b, table_idx, pan_res_handle(62, 0));
   nir_def *buf_idx = nir_iadd(b, res_idx, nir_channel(b, intr->src[0].ssa, 1));
   nir_def *desc_offset = nir_imul_imm(b, buf_idx, PANVK_DESCRIPTOR_SIZE);
   nir_def *size = nir_load_ubo(
      b, 1, 32, res_table, nir_iadd_imm(b, desc_offset, 4), .range = ~0u,
      .align_mul = PANVK_DESCRIPTOR_SIZE, .align_offset = 4);

   nir_def_replace(&intr->def, size);
   return true;
}

static bool
valhall_pack_buf_idx(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   unsigned index_src;

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      index_src = 0;
      break;
   case nir_intrinsic_store_ssbo:
      index_src = 1;
      break;
   default:
      return false;
   }

   nir_def *index = intrin->src[index_src].ssa;
   if (index->num_components == 1)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);
   nir_def *packed_index =
      nir_iadd(b, nir_channel(b, index, 0), nir_channel(b, index, 1));
   nir_src_rewrite(&intrin->src[index_src], packed_index);
   return true;
}

/* --- panvk_lower_non_uniform (from panvk_vX_shader.c) ------------------- */

static bool
mark_all_access_non_uniform(UNUSED nir_builder *b, nir_instr *instr,
                            UNUSED void *data)
{
   switch (instr->type) {
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      tex->texture_non_uniform = true;
      tex->sampler_non_uniform = true;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      switch (intrin->intrinsic) {
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_ssbo:
      case nir_intrinsic_store_ssbo:
      case nir_intrinsic_ssbo_atomic:
      case nir_intrinsic_ssbo_atomic_swap:
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_sparse_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
         nir_intrinsic_set_access(
            intrin, nir_intrinsic_access(intrin) | ACCESS_NON_UNIFORM);
         return true;
      default:
         return false;
      }
   }
   default:
      return false;
   }
}

static bool
standalone_can_merge_workgroups(nir_shader *nir, unsigned arch)
{
   if (!mesa_shader_stage_uses_workgroup(nir->info.stage))
      return false;

   bool allow_merging_workgroups = valhall_can_merge_workgroups(nir);
   if (allow_merging_workgroups) {
      nir->options = pan_get_nir_shader_compiler_options(
         arch, MESA_SHADER_COMPUTE, true);
      nir_foreach_function_impl(impl, nir)
         nir_progress(true, impl, ~nir_metadata_divergence);
   }
   return allow_merging_workgroups;
}

static void
standalone_lower_non_uniform(nir_shader *nir, unsigned arch,
                             bool allow_merging_workgroups)
{

   if (arch >= 10 && allow_merging_workgroups)
      NIR_PASS(_, nir, nir_shader_instructions_pass,
               mark_all_access_non_uniform,
               nir_metadata_control_flow | nir_metadata_live_defs |
                  nir_metadata_instr_index,
               NULL);

   enum nir_lower_non_uniform_access_type types =
      nir_lower_non_uniform_ubo_access |
      nir_lower_non_uniform_ssbo_access |
      nir_lower_non_uniform_texture_access |
      nir_lower_non_uniform_texture_query |
      nir_lower_non_uniform_image_access |
      nir_lower_non_uniform_image_query |
      nir_lower_non_uniform_get_ssbo_size;

   if (allow_merging_workgroups ||
       nir_has_non_uniform_access(nir, types)) {
      NIR_PASS(_, nir, nir_opt_non_uniform_access);
      struct nir_lower_non_uniform_access_options opts = {
         .types = types,
      };
      NIR_PASS(_, nir, nir_lower_non_uniform_access, &opts);
   }
}

/* --- panvk_lower_sysvals & lower_load_push_consts ----------------------- */

static bool
panvk_lower_sysvals(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   unsigned arch = *(const unsigned *)data;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned bit_size = intr->def.bit_size;
   nir_def *val = NULL;
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_base_workgroup_id:
      val = load_sysval(b, compute, bit_size, base);
      break;
   case nir_intrinsic_load_num_workgroups:
      val = load_sysval(b, compute, bit_size, num_work_groups);
      break;
   case nir_intrinsic_load_workgroup_size:
      val = load_sysval(b, compute, bit_size, local_group_size);
      break;
   case nir_intrinsic_load_viewport_scale:
      val = load_sysval(b, graphics, bit_size, viewport.scale);
      break;
   case nir_intrinsic_load_viewport_offset:
      val = load_sysval(b, graphics, bit_size, viewport.offset);
      break;
   case nir_intrinsic_load_first_vertex:
      val = load_sysval(b, graphics, bit_size, vs.first_vertex);
      break;
   case nir_intrinsic_load_base_instance:
      val = load_sysval(b, graphics, bit_size, vs.base_instance);
      break;
   case nir_intrinsic_load_noperspective_varyings_pan:
      assert(b->shader->info.stage == MESA_SHADER_VERTEX);
      val = load_sysval(b, graphics, bit_size, vs.noperspective_varyings);
      break;
   case nir_intrinsic_load_view_index:
      val = nir_imm_zero(b, 1, 32);
      break;
   case nir_intrinsic_load_layer_id:
      val = nir_load_frame_arg_pan(b);
      val = nir_extract_u8_imm(b, nir_u2u32(b, val), 0);
      break;
   case nir_intrinsic_load_draw_id:
      if (arch >= 10)
         return false;
      assert(b->shader->info.stage == MESA_SHADER_VERTEX);
      val = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_printf_buffer_address:
      val = load_sysval(b, common, bit_size, printf_buffer_address);
      break;
   case nir_intrinsic_load_blend_descriptor_pan: {
      uint32_t loc = nir_intrinsic_base(intr);
      val = load_sysval(b, graphics, bit_size, fs.blend_descs[loc]);
      break;
   }
   case nir_intrinsic_load_input_attachment_target_pan: {
      nir_def *ia_info =
         load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);
      val = nir_channel(b, ia_info, 0);
      break;
   }
   case nir_intrinsic_load_input_attachment_conv_pan: {
      nir_def *ia_info =
         load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);
      val = nir_channel(b, ia_info, 1);
      break;
   }
   case nir_intrinsic_load_ro_sink_address_poly:
   case nir_intrinsic_load_stat_query_address_poly:
      val = nir_imm_int64(b, PAN_SHADER_OOB_ADDRESS);
      break;
   case nir_intrinsic_load_vertex_param_buffer_poly:
      val = load_sysval(b, graphics, bit_size, poly.vertex_params);
      break;
   case nir_intrinsic_load_geometry_param_buffer_poly:
      val = load_sysval(b, graphics, bit_size, poly.geometry_params);
      break;
   case nir_intrinsic_load_rasterization_stream:
      val = load_sysval(b, graphics, bit_size, poly.rasterization_stream);
      break;
   case nir_intrinsic_load_provoking_last:
      val = load_sysval(b, graphics, bit_size, poly.provoking_vertex);
      break;
   default:
      return false;
   }

   assert(val->num_components == intr->def.num_components);
   b->cursor = nir_after_instr(instr);
   nir_def_rewrite_uses(&intr->def, val);
   return true;
}

static bool
collect_push_constant(UNUSED struct nir_builder *b, nir_intrinsic_instr *intr,
                      void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   struct standalone_fau_info *fau = data;
   uint32_t base = nir_intrinsic_base(intr);
   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;
   uint32_t offset, size;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   if (!nir_src_is_const(intr->src[0])) {
      offset = base;
      size = nir_intrinsic_range(intr);
      shader_use_sysval(fau, common, push_uniforms);
   } else {
      offset = base + nir_src_as_uint(intr->src[0]);
      size = (intr->def.bit_size / 8) * intr->def.num_components;
   }

   if (is_sysval)
      fau_use_range(fau->used_sysvals, MAX_SYSVAL_FAUS, offset, size);
   else
      fau_use_range(fau->used_push_consts, MAX_PUSH_CONST_FAUS, offset, size);

   return true;
}

static bool
move_push_constant(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   const struct standalone_fau_info *fau = data;
   unsigned base = nir_intrinsic_base(intr);
   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   b->cursor = nir_before_instr(&intr->instr);

   if (nir_src_is_const(intr->src[0])) {
      unsigned offset = base + nir_src_as_uint(intr->src[0]);
      offset = is_sysval ? shader_remapped_sysval_offset(fau, offset)
                         : shader_remapped_push_const_offset(fau, offset);

      nir_src_rewrite(&intr->src[0], nir_imm_int(b, offset));
      nir_intrinsic_set_base(intr, 0);
      nir_intrinsic_set_range(intr, 0);
   } else {
      unsigned push_const_buf_offset = shader_remapped_sysval_offset(
         fau, sysval_offset(common, push_uniforms));
      nir_def *push_const_buf = nir_load_push_constant(
         b, 1, 64, nir_imm_int(b, push_const_buf_offset));
      unsigned push_const_offset =
         is_sysval ? shader_remapped_sysval_offset(fau, base)
                   : shader_remapped_push_const_offset(fau, base);
      nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, push_const_offset);
      unsigned align = nir_combined_align(nir_intrinsic_align_mul(intr),
                                          nir_intrinsic_align_offset(intr));
      align = MIN2(align, FAU_WORD_SIZE);
      nir_def *value = nir_load_global(
         b, intr->def.num_components, intr->def.bit_size,
         nir_iadd(b, push_const_buf, nir_u2u64(b, offset)),
         .align_mul = align);
      nir_def_replace(&intr->def, value);
   }

   return true;
}

static void
lower_load_push_consts(nir_shader *nir, struct standalone_fau_info *fau)
{
   bool progress = false;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
   } while (progress);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      STATIC_ASSERT(
         offsetof(struct panvk_graphics_sysvals, blend.constants) == 0 &&
         sizeof(((struct panvk_graphics_sysvals *)NULL)->blend.constants) ==
            16);
      shader_use_sysval(fau, graphics, blend.constants);
   }

   progress = false;
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, collect_push_constant,
            nir_metadata_all, fau);

   fau->sysval_count = BITSET_COUNT(fau->used_sysvals);
   assert(fau->sysval_count <= FAU_WORD_COUNT - 32 && "too many sysval FAUs");
   fau->total_count = fau->sysval_count + BITSET_COUNT(fau->used_push_consts);
   assert(fau->total_count <= FAU_WORD_COUNT &&
          "asking for more FAUs than the hardware has to offer");

   if (!progress)
      return;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, move_push_constant,
            nir_metadata_control_flow, fau);
}

/* --- Metadata JSON export (for c_8c7eadb161a76028 / spirv2valhall) ------- */

static const char *
stage_name(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return "vertex";
   case MESA_SHADER_FRAGMENT:
      return "fragment";
   default:
      return "compute";
   }
}

static void
write_info_json(const char *path, unsigned arch, mesa_shader_stage stage,
                const char *entry, bool use_kraid, bool idvs, bool robust,
                bool merge_wg, uint32_t core_max_id, size_t code_size,
                const struct pan_shader_info *info,
                const struct standalone_fau_info *fau,
                const struct synth_desc_ctx *desc_ctx)
{
   FILE *f = fopen(path, "w");
   if (!f) {
      fprintf(stderr, "panfrost_vk_compile: cannot open %s for writing\n",
              path);
      return;
   }

   fprintf(f, "{\n");
   fprintf(f, "  \"arch\": %d,\n", arch);
   fprintf(f, "  \"stage\": \"%s\",\n", stage_name(stage));
   fprintf(f, "  \"entry\": \"%s\",\n", entry);
   fprintf(f, "  \"backend\": \"%s\",\n", use_kraid ? "kraid" : "bifrost");
   fprintf(f, "  \"idvs\": %s,\n", idvs ? "true" : "false");
   fprintf(f, "  \"robust\": %s,\n", robust ? "true" : "false");
   fprintf(f, "  \"merge_workgroups\": %s,\n", merge_wg ? "true" : "false");
   fprintf(f, "  \"core_max_id\": %u,\n", core_max_id);
   fprintf(f, "  \"code_size\": %zu,\n", code_size);
   fprintf(f, "  \"instrs\": %u,\n", info->stats.valhall.instrs);
   fprintf(f, "  \"work_reg_count\": %u,\n", info->work_reg_count);
   fprintf(f, "  \"tls_size\": %u,\n", info->tls_size);
   fprintf(f, "  \"wls_size\": %u,\n", info->wls_size);

   fprintf(f, "  \"fau\": {\n");
   fprintf(f, "    \"sysval_slots\": %u,\n", fau->sysval_count);
   fprintf(f, "    \"driver_slots\": %u,\n", fau->total_count);
   fprintf(f, "    \"reserved_words\": %u,\n", info->fau.reserved);
   fprintf(f, "    \"total_words\": %u,\n", info->fau.count);
   fprintf(f, "    \"used_sysval_words\": [");
   bool first = true;
   for (unsigned i = 0; i < MAX_SYSVAL_FAUS; i++) {
      if (BITSET_TEST(fau->used_sysvals, i)) {
         fprintf(f, "%s%u", first ? "" : ", ", i);
         first = false;
      }
   }
   fprintf(f, "],\n");
   fprintf(f, "    \"used_push_const_words\": [");
   first = true;
   for (unsigned i = 0; i < MAX_PUSH_CONST_FAUS; i++) {
      if (BITSET_TEST(fau->used_push_consts, i)) {
         fprintf(f, "%s%u", first ? "" : ", ", i);
         first = false;
      }
   }
   fprintf(f, "],\n");
   fprintf(f, "    \"immediates\": [");
   first = true;
   pan_fau_foreach_imm(&info->fau, i) {
      fprintf(f, "%s{\"word\": %u, \"value\": \"0x%08x\"}",
              first ? "" : ", ", i, info->fau.words[i].constant);
      first = false;
   }
   fprintf(f, "],\n");
   fprintf(f, "    \"ubo_relocs\": [");
   first = true;
   pan_fau_foreach_reloc(&info->fau, i) {
      fprintf(f, "%s{\"word\": %u, \"ubo\": %u, \"offset\": %u}",
              first ? "" : ", ", i, info->fau.words[i].relocation.ubo,
              info->fau.words[i].relocation.offset);
      first = false;
   }
   fprintf(f, "]\n");
   fprintf(f, "  },\n");

   fprintf(f, "  \"descriptors\": [\n");
   first = true;
   for (uint32_t s = 0; s < MAX_SETS; s++) {
      for (uint32_t b = 0; b < desc_ctx->sets[s].max_binding; b++) {
         const struct synth_binding_info *bi = &desc_ctx->sets[s].bindings[b];
         if (bi->kind == SYNTH_DESC_NONE)
            continue;
         fprintf(
            f,
            "%s    {\"set\": %u, \"table\": %u, \"binding\": %u, "
            "\"kind\": \"%s\", \"desc_idx\": %u, \"desc_count\": %u}",
            first ? "" : ",\n", s, s + 1, b, synth_desc_kind_name(bi->kind),
            bi->desc_idx, bi->desc_count);
         first = false;
      }
   }
   fprintf(f, "\n  ]\n");
   fprintf(f, "}\n");
   fclose(f);
}

/* --- SPIR-V front-end helpers ------------------------------------------- */

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

static mesa_shader_stage
detect_stage(const uint32_t *words, size_t word_count, char *entry,
             size_t entry_sz)
{
   mesa_shader_stage stage = MESA_SHADER_COMPUTE;
   strcpy(entry, "main");

   unsigned i = 5;
   while (i < word_count) {
      uint32_t inst = words[i];
      unsigned wc = inst >> 16;
      unsigned op = inst & 0xffff;
      if (wc == 0 || i + wc > word_count)
         break;

      if (op == SpvOpEntryPoint) {
         SpvExecutionModel model = (SpvExecutionModel)words[i + 1];
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

/* Mirrors vk_physical_device_get_spirv_capabilities() (generated from
 * vk.xml) for the subset of caps panvk's get_physical_device_features()
 * actually enables, selected by PAN_ARCH.  No panvk/libvulkan dependency.
 *
 * Device-dependent caps (shader clock) are taken as parameters because the
 * standalone tool has no real device to query; pass the same values panvk
 * would derive from kmod props (gpu_can_query_timestamp /
 * timestamp_device_coherent).
 */
static struct spirv_capabilities
get_spirv_capabilities(unsigned arch,
                       bool can_query_timestamp,
                       bool timestamp_device_coherent)
{
   struct spirv_capabilities caps = { false, };

   /* --- Vulkan 1.0 core features (panvk: all true except noted) --- */
   caps.Int16 = true;   /* shaderInt16 = true */
   caps.Int64 = true;   /* shaderInt64 = true */
   caps.Float64 = false;/* shaderFloat64 = false (all archs) */
   caps.Int8 = true;    /* shaderInt8 = true (VK12 / KHR_shader_float16_int8) */

   /* shaderFloat16 = PAN_ARCH >= 10 */
   caps.Float16 = arch >= 10;

   /* shaderStorageImageMultisample = false -> no MS storage/array caps */
   /* (there is no StorageImageMS enumerant; MS is gated via ImageMSArray) */
   caps.ImageMSArray = false; /* shaderStorageImageMultisample = false */

   caps.StorageImageReadWithoutFormat = true;  /* shaderStorageImageReadWithoutFormat */
   caps.StorageImageWriteWithoutFormat = true; /* shaderStorageImageWriteWithoutFormat */

   /* --- Vulkan 1.1 --- */
   caps.StorageBuffer16BitAccess = true;            /* storageBuffer16BitAccess */
   caps.UniformAndStorageBuffer16BitAccess = true;  /* uniformAndStorageBuffer16BitAccess */
   caps.StoragePushConstant16 = true;               /* storagePushConstant16 */
   caps.StorageInputOutput16 = true;                /* storageInputOutput16 */
   caps.VariablePointers = true;                    /* variablePointers */
   caps.VariablePointersStorageBuffer = true;       /* variablePointersStorageBuffer */

   /* --- Vulkan 1.2 / KHR_8bit_storage --- */
   caps.StorageBuffer8BitAccess = true;             /* storageBuffer8BitAccess */
   caps.UniformAndStorageBuffer8BitAccess = true;   /* uniformAndStorageBuffer8BitAccess */
   caps.StoragePushConstant8 = true;                /* storagePushConstant8 */

   /* descriptorIndexing = PAN_ARCH >= 9; non-uniform indexing features all
    * true.  The non-uniform cap is ShaderNonUniform; dynamic indexing is
    * implied by the per-class *ArrayDynamicIndexing caps (all true here). */
   caps.ShaderNonUniform = arch >= 9;               /* descriptorIndexing */
   caps.RuntimeDescriptorArray = true;              /* runtimeDescriptorArray */

   /* bufferDeviceAddress = true */
   caps.PhysicalStorageBufferAddresses = true;

   /* vulkanMemoryModel + DeviceScope = true */
   caps.VulkanMemoryModel = true;
   caps.VulkanMemoryModelDeviceScope = true;

   /* --- Vulkan 1.3 --- */
   caps.DemoteToHelperInvocation = true;            /* shaderDemoteToHelperInvocation */

   /* --- VK_KHR_shader_clock (device-dependent) --- */
   caps.ShaderClockKHR = can_query_timestamp;       /* KHR_shader_clock ext */

   /* --- Subgroup ops (from subgroupSupportedOperations property) ---
    * panvk sets BASIC|VOTE|ARITHMETIC|BALLOT|SHUFFLE|SHUFFLE_RELATIVE|
    * CLUSTERED|QUAD|ROTATE|ROTATE_CLUSTERED for all archs.
    */
   caps.GroupNonUniform = true;                     /* BASIC_BIT */
   caps.GroupNonUniformVote = true;                 /* VOTE_BIT */
   caps.GroupNonUniformArithmetic = true;           /* ARITHMETIC_BIT */
   caps.GroupNonUniformBallot = true;               /* BALLOT_BIT */
   caps.GroupNonUniformShuffle = true;              /* SHUFFLE_BIT */
   caps.GroupNonUniformShuffleRelative = true;      /* SHUFFLE_RELATIVE_BIT */
   caps.GroupNonUniformClustered = true;            /* CLUSTERED_BIT */
   caps.GroupNonUniformQuad = true;                 /* QUAD_BIT */

   /* --- VK_KHR_cooperative_matrix: PAN_ARCH >= 11 --- */
   caps.CooperativeMatrixKHR = arch >= 11;

   /* --- VK_KHR_shader_integer_dot_product: shaderIntegerDotProduct = true --- */
   caps.DotProductKHR = true;

   (void)timestamp_device_coherent; /* only affects shaderDeviceClock, not a cap */
   return caps;
}

int
panfrost_vk_compile(const struct panfrost_vk_compile_options *opts)
{
   const char *in_path = opts->in_path;
   const char *out_path = opts->out_path;
   unsigned arch = opts->arch;
   const char *entry_override = opts->entry_override;
   const char *info_json_path = opts->info_json_path;
   int stage_override = opts->stage_override;
   bool robust = opts->robust;
   bool no_idvs = opts->no_idvs;
   bool promote_imm = opts->promote_imm;
   bool clamp_version = opts->clamp_version;
   bool dump_nir = opts->dump_nir;
   bool disasm = opts->disasm;
   uint32_t core_max_id = opts->core_max_id;

   if (!in_path || !out_path) {
      fprintf(stderr,
              "panfrost_vk_compile: in_path and out_path are required\n");
      return 1;
   }

   size_t spirv_size = 0;
   uint32_t *words = read_spirv(in_path, &spirv_size);
   if (!words)
      return 1;

   if (spirv_size < 20 || words[0] != 0x07230203u) {
      fprintf(stderr, "panfrost_vk_compile: %s is not a SPIR-V binary\n",
              in_path);
      free(words);
      return 1;
   }

   if (clamp_version && words[SPIRV_VERSION_WORD] > SPIRV_VERSION_1_3)
      words[SPIRV_VERSION_WORD] = SPIRV_VERSION_1_3;

   char entry[64];
   mesa_shader_stage stage =
      detect_stage(words, spirv_size / 4, entry, sizeof(entry));
   if (entry_override)
      snprintf(entry, sizeof(entry), "%s", entry_override);
   if (stage_override >= 0)
      stage = (mesa_shader_stage)stage_override;

   glsl_type_singleton_init_or_ref();

   struct spirv_capabilities spirv_caps =
         get_spirv_capabilities(arch,
                                /* can_query_timestamp = */ true,
                                /* timestamp_device_coherent = */ true);

   struct spirv_to_nir_options sopts = {
      .environment = NIR_SPIRV_VULKAN,
      .capabilities = &spirv_caps,
      .group_non_uniform_subgroup_size = pan_subgroup_size(arch),
      .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ubo_alignment = 16,
      .min_ssbo_alignment = 16,
   };

   const nir_shader_compiler_options *nopts =
      pan_get_nir_shader_compiler_options(arch, stage, false);

   nir_shader *nir =
      spirv_to_nir(words, spirv_size / 4, NULL, stage, entry, &sopts, nopts);
   free(words);
   if (!nir) {
      fprintf(stderr, "panfrost_vk_compile: spirv_to_nir failed\n");
      glsl_type_singleton_decref();
      return 1;
   }

   /* Early lowering before panvk_preprocess_nir(). */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_all);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);

   /* --- panvk_preprocess_nir (v9+ specialization) ---------------------- */
   if (pan_use_kraid(arch, stage, false))
      NIR_PASS(_, nir, nir_lower_mediump_vars,
               nir_var_shader_temp | nir_var_function_temp);

   if (stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_vectorize_io_vars, nir_var_shader_out);

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

   NIR_PASS(_, nir, nir_inline_sysval,
            nir_intrinsic_load_core_max_id_arm, core_max_id);

   pan_preprocess_nir(nir, arch);

   /* --- panvk_lower_nir (v9+ specialization) --------------------------- */
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   NIR_PASS(_, nir, panvk_nir_lower_cooperative_matrix,
            pan_subgroup_size(arch));

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   /* NOTE: nir_vk_lower_ycbcr_tex is intentionally omitted when no
    * VkSamplerYcbcrConversion state is supplied; without app-created YCbCr
    * conversions, lookup_ycbcr_conversion returns NULL and it is a no-op. */

   struct synth_desc_ctx desc_ctx = {
      .add_bounds_checks = robust,
   };
   assign_synth_desc_indices(nir, &desc_ctx);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_vk_descriptors_v9,
            nir_metadata_control_flow, &desc_ctx);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            nir_address_format_vec2_index_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            nir_address_format_vec2_index_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);

   if (robust) {
      NIR_PASS(_, nir, nir_lower_robust_access, is_robust_ssbo_intr, NULL);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, valhall_lower_get_ssbo_size,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_instructions_pass, valhall_pack_buf_idx,
            nir_metadata_control_flow, NULL);

   if (mesa_shader_stage_uses_workgroup(stage)) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               shared_type_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory,
               nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* --- panvk_lower_nir_io (VS/FS only) -------------------------------- */
   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_FRAGMENT) {
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);

      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
               nir_shader_get_entrypoint(nir), nir_var_shader_out);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      // nir_lower_global_vars_to_local generates meta-instructions that need to be elided
      // these have already run before, but we need to run them here too
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
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
      .arch = arch,
      .model = NULL,
      .no_idvs = no_idvs,
      .robust_modes =
         robust ? (nir_var_mem_ubo | nir_var_mem_ssbo) : 0,
      .robust_descriptors = arch >= 10,
      .fau = { .reserved = 0, .promote_immediates = promote_imm },
   };

   struct pan_varying_layout vlayout = {0};
   if (stage == MESA_SHADER_VERTEX) {
      pan_varying_collect_formats(&vlayout, nir, arch, false);
      pan_build_varying_layout_compact(&vlayout, nir, arch);
      inputs.varying_layout = &vlayout;
   }

   bool allow_merging_workgroups = standalone_can_merge_workgroups(nir, arch);
   standalone_lower_non_uniform(nir, arch, allow_merging_workgroups);

   struct pan_shader_info info = {0};
   pan_postprocess_nir(nir, &inputs, &info);

   struct standalone_fau_info fau = {0};
   NIR_PASS(_, nir, nir_shader_instructions_pass, panvk_lower_sysvals,
            nir_metadata_control_flow, &arch);
   lower_load_push_consts(nir, &fau);

   inputs.fau.reserved = fau.total_count * 2;
   inputs.fau.promote_immediates = promote_imm;

   if (dump_nir)
      nir_print_shader(nir, stderr);

   struct util_dynarray binary;
   util_dynarray_init(&binary, NULL);
   pan_shader_compile(nir, &inputs, &binary, &info);

   if (disasm) {
      pan_disassemble_bi(stderr, binary.data, binary.size, arch,
                           /* verbose = */ true);
      fprintf(stderr, "\n");
   }

   FILE *out = fopen(out_path, "wb");
   if (!out) {
      fprintf(stderr, "panfrost_vk_compile: cannot open %s for writing\n",
              out_path);
      util_dynarray_fini(&binary);
      ralloc_free(nir);
      glsl_type_singleton_decref();
      return 1;
   }

   fwrite(binary.data, 1, binary.size, out);
   fclose(out);

   bool used_kraid = pan_use_kraid(arch, stage, false);
   if (info_json_path) {
      write_info_json(info_json_path, arch, stage, entry, used_kraid, !no_idvs,
                      robust, allow_merging_workgroups, core_max_id,
                      binary.size, &info, &fau, &desc_ctx);
   }

   uint32_t instrs = info.stats.valhall.instrs;
   fprintf(stderr,
           "panfrost_vk_compile: %s (%s, %s, fau=%u) -> %s: %u bytes, "
           "%u instructions\n",
           entry, stage_name(stage), used_kraid ? "kraid" : "bifrost",
           fau.total_count, out_path, (unsigned)binary.size, instrs);

   util_dynarray_fini(&binary);
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return 0;
}
