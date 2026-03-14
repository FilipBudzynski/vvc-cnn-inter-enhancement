#!/bin/bash
# fetch_vtm.sh for Arch WSL

# 1. Clean start
if [ ! -d "vtm" ]; then
    echo "Cloning VTM..."
    git clone --depth 1 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
fi

cd vtm
rm -rf build && mkdir build && cd build

# 2. Configure for Linux x86_64
# ENABLE_TRACING is what gives us the CSV metadata for those 7 extra channels
echo "Configuring VTM for Arch Linux..."
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_TRACING=ON \
         -DCMAKE_CXX_FLAGS="-DK0149_BLOCK_STATISTICS=1"

# 3. Build the DecoderAnalyserApp
echo "Compiling DecoderAnalyserApp..."
CORES=$(nproc)
make DecoderAnalyserApp -j$CORES

cd ../..
echo "Build complete. Binary is at: ./vtm/bin/DecoderAnalyserApp"
##! /bin/bash
#
#build_vtm() {
#    # 1. Clone
#    git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
#    cd vtm
#
#    # 2. Patch source for Mac ARM (Forward Declarations)
#    # Using sed -i '' for macOS compatibility
#    sed -i '' '1460i\
#struct CodingUnit;\
#struct PredictionUnit;\
#struct TransformUnit;\
#' source/Lib/CommonLib/TypeDef.h
#
#    sed -i '' '38i\
#struct CodingUnit;\
#struct PredictionUnit;\
#struct TransformUnit;\
#' source/Lib/CommonLib/dtrace.h
#
#    # 3. Build
#    mkdir -p build && cd build
#    cmake .. -DCMAKE_BUILD_TYPE=Release \
#             -DENABLE_TRACING=ON \
#             -DCMAKE_CXX_FLAGS="-DK0149_BLOCK_STATISTICS=1 -w -Wno-everything" \
#             -DCMAKE_C_FLAGS="-w -Wno-everything"
#
#    make DecoderAnalyserApp -j8
#
#    # 4. Cleanup/Link
#    # Moves the binary to the root for easy access
#    ln -sf $(find . -name "DecoderAnalyserApp" -type f) ../DecoderAnalyserApp
#    cd ../..
#}
#
#[ -d "vtm" ] || build_vtm $1

#!/bin/bash
# fetch_vtm_pc.sh
