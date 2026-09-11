#!/bin/bash
# =============================================================================
# Setup dependencies to process Tahoe-100M data efficiently
# =============================================================================

set -e
set -u
set -o pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "biopert" ]; then
    echo "Error: the 'biopert' conda env is not active — run 'conda activate biopert' first" >&2
    exit 1
fi

BIOPERT_ROOT="$(git rev-parse --show-toplevel)"
echo "BIOPERT_ROOT = $BIOPERT_ROOT"
mkdir -p "$BIOPERT_ROOT/deps"


# Install conda dependencies
# =====================================================================
conda install -y -c conda-forge gcc=12 gxx=12 boost-cpp snappy thrift-cpp rapidjson \
                                zlib brotli lz4-c zstd re2 libutf8proc flatbuffers xsimd


# Build Apache Arrow (C++)
# =====================================================================
ARROW_VERSION="25.0.1"
echo "Apache Arrow version: $ARROW_VERSION"

ARROW_TARBALL="apache-arrow-${ARROW_VERSION}.tar.gz"
ARROW_DIR="apache-arrow-${ARROW_VERSION}"
ARROW_URL="https://downloads.apache.org/arrow/arrow-${ARROW_VERSION}/${ARROW_TARBALL}"

cd "$BIOPERT_ROOT/deps"
ARROW_INSTALL_DIR="$(pwd)/arrow-install"

if [ -d "$ARROW_INSTALL_DIR" ] && [ -f "$ARROW_INSTALL_DIR/lib/libarrow.a" ]; then
    echo "================================================================="
    echo "Arrow ${ARROW_VERSION} already built, skipping."
    echo "================================================================="
else
    wget -c "$ARROW_URL"
    tar -xzf "$ARROW_TARBALL"
    rm "$ARROW_TARBALL"

    mkdir -p "$ARROW_INSTALL_DIR"

    cd "${ARROW_DIR}/cpp"
    rm -rf build            # Wipe cache to prevent stale CMake state
    mkdir -p build
    cd build

    cmake .. \
        -DCMAKE_C_COMPILER="$(which gcc)" \
        -DCMAKE_CXX_COMPILER="$(which g++)" \
        -DCMAKE_INSTALL_PREFIX="$ARROW_INSTALL_DIR" \
        -DARROW_WITH_SNAPPY=ON \
        -DARROW_PARQUET=ON \
        -DARROW_CSV=OFF \
        -DARROW_JSON=OFF \
        -DARROW_WITH_UTF8PROC=OFF \
        -DARROW_COMPUTE=OFF \
        -DARROW_DATASET=OFF \
        -DARROW_FILESYSTEM=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20

    make -j"$(nproc)"
    make install

    echo "================================================================="
    echo "Arrow ${ARROW_VERSION} installed to:"
    echo "  $ARROW_INSTALL_DIR"
    echo "================================================================="
fi


# Build libcxxwrap-julia
# =====================================================================
cd "$BIOPERT_ROOT/deps"

if [ ! -d "libcxxwrap-julia" ]; then
    git clone https://github.com/JuliaInterop/libcxxwrap-julia.git
fi

rm -rf build/libcxxwrap-julia       # Always rebuild to avoid stale Julia version mismatch
mkdir -p build/libcxxwrap-julia
cd build/libcxxwrap-julia

CXXWRAP_INSTALL_DIR="$(pwd)/libcxxwrap-julia-install"
mkdir -p "$CXXWRAP_INSTALL_DIR"

JULIA_PREFIX="$(julia -e 'using Libdl; println(Sys.BINDIR |> dirname)')"

cmake ../../libcxxwrap-julia \
    -DJulia_PREFIX="${JULIA_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="$CXXWRAP_INSTALL_DIR" \
    -DJLCXX_BUILD_TESTS=OFF \
    -DJLCXX_BUILD_EXAMPLES=OFF

make -j"$(nproc)"
make install

echo "================================================================="
echo "libcxxwrap-julia installed to:"
echo "  $CXXWRAP_INSTALL_DIR"
echo "================================================================="


# Build ArrowWrap (CxxWrap module)
# =====================================================================
cd "$BIOPERT_ROOT/cxx/ArrowWrap"
rm -rf build          # wipe cache to prevent stale CMake state
mkdir -p build
cd build

cmake .. \
    -DCMAKE_C_COMPILER="$(which gcc)" \
    -DCMAKE_CXX_COMPILER="$(which g++)" \
    -DARROW_INSTALL_DIR="$ARROW_INSTALL_DIR" \
    -DJLCXX_INSTALL_DIR="$CXXWRAP_INSTALL_DIR" \
    -DJulia_PREFIX="$JULIA_PREFIX"

make -j"$(nproc)"

echo "================================================================="
echo "ArrowWrap built successfully"
echo "================================================================="


# Done
# =====================================================================
echo "================================================================="
echo "Setup complete!"
echo "Arrow:            $ARROW_INSTALL_DIR"
echo "libcxxwrap-julia: $CXXWRAP_INSTALL_DIR"
echo "ArrowWrap:        $BIOPERT_ROOT/cxx/ArrowWrap/build"
echo "================================================================="