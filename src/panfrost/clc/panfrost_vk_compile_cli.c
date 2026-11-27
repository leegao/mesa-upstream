/*
 * Copyright 2024 Collabora Ltd
 * SPDX-License-Identifier: MIT
 *
 * Command-line front-end for panfrost_vk_compile().  All real work lives in
 * the static library; this file only parses flags into an options struct.
 */

#include "panfrost_vk_compile.h"
#include "shader_enums.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef PAN_ARCH
#define PAN_ARCH 11
#endif

static void
print_usage(const char *prog, FILE *stream)
{
   fprintf(
      stream,
      "Usage: %s [options] <input.spv> <output.bin>\n"
      "Options:\n"
      "  --arch <N|vN>         Target Valhall generation 9..15 (default: 11)\n"
      "  --kraid               Force Rust Kraid backend (PAN_USE_KRAID=all)\n"
      "  --no-kraid            Force C Bifrost/Valhall backend (PAN_USE_KRAID=0)\n"
      "  --robust              Enable SSBO/descriptor robustness bounds checks\n"
      "  --no-idvs             Disable IDVS for vertex shaders\n"
      "  --entry <name>        Override SPIR-V entry point name\n"
      "  --stage <vert|frag|comp>  Override detected shader stage\n"
      "  --core-max-id <N>     Set load_core_max_id_arm sysval (default: 0)\n"
      "  --no-promote-imm      Disable FAU immediate promotion\n"
      "  --no-clamp-version    Do not clamp SPIR-V >= 1.4 header to 1.3\n"
      "  --dump-nir            Print final lowered NIR to stderr\n"
      "  --disasm              Disassemble emitted Valhall ISA to stderr\n"
      "  --info-json <file>    Write FAU, descriptor, and shader stats JSON\n"
      "  -h, --help            Show this help message\n",
      prog);
}

int
main(int argc, char **argv)
{
   struct panfrost_vk_compile_options opts = {
      .arch = 11,
      .stage_override = -1,
      .robust = getenv("PANVK_ROBUST") && atoi(getenv("PANVK_ROBUST")) != 0,
      .promote_imm = true,
      .clamp_version = true,
   };

   for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
         print_usage(argv[0], stdout);
         return 0;
      } else if (strcmp(argv[i], "--arch") == 0 && i + 1 < argc) {
         const char *s = argv[++i];
         if (s[0] == 'v' || s[0] == 'V')
            s++;
         char *end;
         long a = strtol(s, &end, 10);
         if (*end || a < 9 || a > 15) {
            fprintf(stderr,
                     "panfrost_vk_compile: invalid --arch '%s' (must be 9..15)\n",
                     argv[i]);
            return 1;
         }
         opts.arch = (unsigned)a;
      } else if (strcmp(argv[i], "--robust") == 0) {
         opts.robust = true;
      } else if (strcmp(argv[i], "--no-idvs") == 0) {
         opts.no_idvs = true;
      } else if (strcmp(argv[i], "--kraid") == 0) {
         setenv("PAN_USE_KRAID", "all", 1);
      } else if (strcmp(argv[i], "--no-kraid") == 0) {
         setenv("PAN_USE_KRAID", "0", 1);
      } else if (strcmp(argv[i], "--no-promote-imm") == 0) {
         opts.promote_imm = false;
      } else if (strcmp(argv[i], "--no-clamp-version") == 0) {
         opts.clamp_version = false;
      } else if (strcmp(argv[i], "--dump-nir") == 0) {
         opts.dump_nir = true;
      } else if (strcmp(argv[i], "--disasm") == 0) {
         opts.disasm = true;
      } else if (strcmp(argv[i], "--entry") == 0 && i + 1 < argc) {
         opts.entry_override = argv[++i];
      } else if (strcmp(argv[i], "--info-json") == 0 && i + 1 < argc) {
         opts.info_json_path = argv[++i];
      } else if (strcmp(argv[i], "--core-max-id") == 0 && i + 1 < argc) {
         opts.core_max_id = (uint32_t)strtoul(argv[++i], NULL, 0);
      } else if (strcmp(argv[i], "--stage") == 0 && i + 1 < argc) {
         const char *s = argv[++i];
         if (strcmp(s, "vert") == 0 || strcmp(s, "vertex") == 0) {
            opts.stage_override = MESA_SHADER_VERTEX;
         } else if (strcmp(s, "frag") == 0 || strcmp(s, "fragment") == 0) {
            opts.stage_override = MESA_SHADER_FRAGMENT;
         } else if (strcmp(s, "comp") == 0 || strcmp(s, "compute") == 0) {
            opts.stage_override = MESA_SHADER_COMPUTE;
         } else {
            fprintf(stderr, "panfrost_vk_compile: unknown stage '%s'\n", s);
            return 1;
         }
      } else if (argv[i][0] == '-') {
         fprintf(stderr, "panfrost_vk_compile: unknown option '%s'\n", argv[i]);
         return 1;
      } else if (!opts.in_path) {
         opts.in_path = argv[i];
      } else if (!opts.out_path) {
         opts.out_path = argv[i];
      } else {
         fprintf(stderr, "panfrost_vk_compile: unexpected argument %s\n",
                 argv[i]);
         return 1;
      }
   }

   if (!opts.in_path || !opts.out_path) {
      print_usage(argv[0], stderr);
      return 1;
   }

   return panfrost_vk_compile(&opts);
}
