#!/bin/sh
# nvcc host-compiler wrapper.
#
# In this WSL + conda toolchain, nvcc injects its own bin dir (which holds an
# `ld` linked against a different glibc) into the host compiler's `-B` search
# path, so CUDA link steps fail with `__nptl_change_stack_perm@GLIBC_PRIVATE`.
# Prepending `-B<system binutils>` makes the compatible system `ld` win for both
# compile and link. Override the host compiler / binutils dir via env vars.
exec "${DFINE_HOST_CXX:-/usr/bin/g++}" -B"${DFINE_BINUTILS_DIR:-/usr/bin}" "$@"
