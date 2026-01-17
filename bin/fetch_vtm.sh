#! /bin/bash

build_vtm() {
    # 1. Clone
    git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
    cd vtm
    
    # 2. Patch source for Mac ARM (Forward Declarations)
    # Using sed -i '' for macOS compatibility
    sed -i '' '1460i\
struct CodingUnit;\
struct PredictionUnit;\
struct TransformUnit;\
' source/Lib/CommonLib/TypeDef.h

    sed -i '' '38i\
struct CodingUnit;\
struct PredictionUnit;\
struct TransformUnit;\
' source/Lib/CommonLib/dtrace.h

    # 3. Build
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DENABLE_TRACING=ON \
             -DCMAKE_CXX_FLAGS="-DK0149_BLOCK_STATISTICS=1 -w -Wno-everything" \
             -DCMAKE_C_FLAGS="-w -Wno-everything"
    
    make DecoderAnalyserApp -j8
    
    # 4. Cleanup/Link
    # Moves the binary to the root for easy access
    ln -sf $(find . -name "DecoderAnalyserApp" -type f) ../DecoderAnalyserApp
    cd ../..
}

[ -d "vtm" ] || build_vtm $1
