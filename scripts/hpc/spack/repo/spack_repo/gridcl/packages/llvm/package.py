# LLVM for CubeCL's tracel-llvm-bundler on Myriad (ADR-0004). The bundler 23.1.0-3 ships a
# prebuilt LLVM 23.1.0 that needs glibc >= 2.34; this recipe builds the same version on RHEL 7.
# The builtin recipe stops at 22.1.5; this adds 23.1.0 and turns zlib off, like tracel's bundle
# (llvm-config --system-libs then reports no -lz, so the Rust link needs no extra search path).
# A CubeCL upgrade that changes the bundler version adds the matching LLVM version here.
from spack_repo.builtin.packages.llvm.package import Llvm as BuiltinLlvm

from spack.package import *


class Llvm(BuiltinLlvm):
    version("23.1.0", sha256="d8657b2a7291e518407bf13c4b41c85ef2cded2d4354097a2f451644dfc817b0")

    def cmake_args(self):
        args = super().cmake_args()
        args.append(self.define("LLVM_ENABLE_ZLIB", False))
        return args
