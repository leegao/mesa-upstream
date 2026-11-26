# predownloaded/ — airgapped build bundle for `panfrost_vk_compile`

This folder packages every tool needed to build `panfrost_vk_compile` on a
**network-isolated Debian 12 (bookworm) (gvisor) sandbox**. Populate it on a
connected Debian 12 machine, copy the whole folder into the sandbox, then
install offline.

> bookworm's default repos only ship LLVM 14, so `fetch.sh` adds the official
> LLVM apt repo (apt.llvm.org) to obtain `llvm-15-dev` / `libclang-15-dev` /
> `libclc-15-dev` / `libllvmspirvlib-15-dev` (LLVM 15 is the minimum the CLC
> path accepts).

## Layout

```
predownloaded/
  debs/      # .deb packages (build-essential, llvm-15, libdrm, bison, flex, ...)
  wheels/    # Python wheels (meson, ninja, mako, packaging, PyYAML)
  bin/       # standalone binaries (bindgen)
  fetch.sh         # run on a CONNECTED bookworm machine to populate debs/ wheels/ bin/
  fetch-bookworm.sh# run on Ubuntu/other to fetch bookworm packages via a container
  install.sh       # run INSIDE the airgapped sandbox to install everything
```

## Usage

### On a Debian 12 (bookworm) machine

```sh
cd predownloaded
./fetch.sh            # downloads everything into debs/ wheels/ bin/
```

### On Ubuntu (or any non-bookworm host)

`apt-get` on Ubuntu resolves **Ubuntu** packages, not bookworm, and `fetch.sh`
would add the bookworm LLVM repo to the host's apt. Use the container wrapper
instead — it runs `fetch.sh` inside a `debian:bookworm` container so the
`.deb`s are the bookworm ones:

```sh
cd predownloaded
./fetch-bookworm.sh   # needs docker or podman
```

`fetch.sh` refuses to run on non-bookworm unless `ALLOW_NON_BOOKWORM=1` is set.

Transfer the `predownloaded/` folder into the sandbox, then inside it:

```sh
cd predownloaded
sudo ./install.sh     # installs .debs, wheels, and the bindgen binary
```

Then build the tool as usual:

```sh
meson setup build . -Dbuildtype=release -Dplatforms= -Dglx=disabled \
  -Degl=disabled -Dgallium-drivers= -Dvulkan-drivers=panfrost \
  -Dprecomp-compiler=enabled -Dpanfrost-rust=false -Dllvm=enabled
ninja -C build src/panfrost/clc/panfrost_vk_compile
```

## Notes

- The `.deb` set is resolved with `apt-get install --download-only`, so it
  includes transitive dependencies. Run `fetch.sh` on **Debian 12 (bookworm)**
  so the package versions match the sandbox. `fetch.sh` adds the LLVM 15 apt
  repo automatically (bookworm ships only LLVM 14 by default, and the CLC path
  requires LLVM >= 15).
- The Rust/Kraid variant additionally needs `bindgen` (provided in `bin/`).
  The default `non-rust` variant does not need bindgen or the Rust toolchain.
- `glslang-tools` is only used by the CI smoke test; it is optional for the
  build itself.