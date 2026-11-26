# `panfrost_vk_compile` — Standalone Vulkan SPIR-V to Valhall ISA Compiler

`panfrost_vk_compile` (`src/panfrost/clc/panfrost_vk_compile.c`) is a host-side, self-contained Vulkan SPIR-V compiler built on Mesa's `spirv_to_nir`, `libnir`, `libpanfrost_compiler`, and `panvk_nir_lower_cooperative_matrix.c`. It has **zero runtime or link dependencies** on DRM, `panthor`, `libvulkan`, or `libpanvk`, and mirrors `panvk_vX_shader.c` for Valhall (`PAN_ARCH` 9–15, default `11`).

---

## 1. Building

### Non-Rust Build (C Bifrost/Valhall Backend Only)

```sh
meson setup build . \
  -Dbuildtype=release \
  -Dplatforms= -Dglx=disabled -Degl=disabled -Dgallium-drivers= \
  -Dvulkan-drivers=panfrost \
  -Dprecomp-compiler=enabled \
  -Dpanfrost-rust=false \
  -Dllvm=enabled
ninja -C build src/panfrost/clc/panfrost_vk_compile
```

### Rust / Kraid Build (Debian 12 Bookworm & Linux Hosts)

Mesa's Rust Kraid backend requires `rustc >= 1.85` and `bindgen == 0.71.1` (`0.72.0` and `0.73.x` are rejected by `meson.build`):

```sh
meson setup build . \
  -Dbuildtype=release \
  -Dplatforms= -Dglx=disabled -Degl=disabled -Dgallium-drivers= \
  -Dvulkan-drivers=panfrost \
  -Dprecomp-compiler=enabled \
  -Dpanfrost-rust=true \
  -Dllvm=enabled
ninja -C build src/panfrost/clc/panfrost_vk_compile
```

To target a different Valhall architecture, change `pan_vk_arch` in `src/panfrost/clc/meson.build` (valid `9..15`).

---

## 2. CLI Usage & Flags

```sh
panfrost_vk_compile [options] <input.spv> <output.bin>
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--kraid` | env / Mesa default | Sets `PAN_USE_KRAID=all` to compile via the Rust Kraid backend (effective when built with `-Dpanfrost-rust=true`). |
| `--no-kraid` | — | Sets `PAN_USE_KRAID=0` to force the C Bifrost/Valhall backend. |
| `--robust` | off (or `PANVK_ROBUST=1`) | Enables `nir_lower_robust_access` (`get_ssbo_size` bounds checks) and descriptor array bounds clamping. Off by default to match `libMali` (`cmpbe_has_robustness == 0`). |
| `--no-idvs` | off (`idvs = true`) | Disables Index-Driven Vertex Shading (`inputs.no_idvs = true`). Keeping IDVS enabled aligns with `panvk` and `libMali` on v11 and avoids `lea_attr_pan` (which Kraid rejects). |
| `--entry <name>` | first `OpEntryPoint` | Overrides the SPIR-V entry point name. |
| `--stage <vert\|frag\|comp>` | inferred from `OpEntryPoint` | Overrides the detected shader stage. |
| `--core-max-id <N>` | `0` | Sets the inlined `load_core_max_id_arm` sysval (`util_last_bit(shader_present) - 1`). |
| `--no-promote-imm` | off (`promote_immediates = true`) | Disables backend immediate promotion into FAU. |
| `--no-clamp-version` | off (clamps `>= 1.4` to `1.3`) | Keeps the original SPIR-V version word instead of clamping `>= 1.4` to `1.3` for unlisted globals. |
| `--dump-nir` | off | Prints the final lowered NIR shader to `stderr` immediately before `pan_shader_compile()`. |
| `--disasm` | off | Sets `BIFROST_MESA_DEBUG=shaders` and `PAN_MESA_DEBUG=shaders` to print Valhall disassembly to `stderr`. |
| `--info-json <file>` | off | Writes a JSON report with stage, backend, register/TLS/WLS counts, FAU word allocations (`sysval_slots`, `immediates`, `ubo_relocs`), and synthetic descriptor table mappings. |
| `-h`, `--help` | — | Prints usage and exits `0`. |

---

## 3. Pass Pipeline & `libMali` v11 Alignment

1. **Cooperative Matrix**: Runs `panvk_nir_lower_cooperative_matrix(nir, pan_subgroup_size(PAN_ARCH))` (`16` on v9+) by compiling `../vulkan/panvk_nir_lower_cooperative_matrix.c` directly into the binary.
2. **Synthetic Descriptors & SSBO Size**: Infers per-set `(set, binding)` layouts from NIR variables/derefs/intrinsics, assigns sequential `desc_idx` slots (`COMBINED_IMAGE_SAMPLER` uses 2 slots), lowers UBO/SSBO to `nir_address_format_vec2_index_32bit_offset` (`pan_res_handle(set + 1, desc_idx)`), lowers `load_constant` to `load_constant_base_ptr`, and runs `valhall_lower_get_ssbo_size` (table `62`) + `valhall_pack_buf_idx`.
3. **Non-Uniform & Workgroup Merging**: Runs `valhall_can_merge_workgroups` + `nir_opt_non_uniform_access` + `nir_lower_non_uniform_access` (marking accesses non-uniform on `PAN_ARCH >= 10` when workgroups are merged).
4. **Sysvals & FAU Packing**: Runs `panvk_lower_sysvals` + `lower_load_push_consts` using `panvk`'s FAU layout (reserving 4 FAU words for FS `blend.constants`, lowering dynamic `load_push_constant` to `nir_load_global` from `common.push_uniforms`, and setting `inputs.fau.reserved = total_count * 2`).
5. **YCbCr**: `nir_vk_lower_ycbcr_tex` is omitted when no `VkSamplerYcbcrConversion` state is supplied (`lookup_ycbcr_conversion` returns `NULL`).

---

## 4. Integration with `valhall2spirv` / `spirv2valhall`

For automated sweeps (`panfrost_vs_ddk.py`) and `defau` devirtualization, invoke with `--info-json`:

```sh
panfrost_vk_compile --kraid --info-json shader_info.json shader.spv shader.bin
```

The generated `shader_info.json` provides:
- `fau.sysval_slots`, `fau.reserved_words`, `fau.immediates[]`, and `fau.ubo_relocs[]` (eliminating hardcoded `pan_shader_info` struct offsets).
- `descriptors[]`: exact `(set, table = set + 1, binding, kind, desc_idx, desc_count)` mapping for matching `LD_PKA` / `LEA_PKA` handles back to SPIR-V bindings.
