#!/bin/bash

# Script to build VTM with research-grade tracing enabled
build_vtm() {
    echo "Cloning VTM (VVC Test Model)..."
    git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
    mkdir -p vtm/build
    cd vtm/build

    echo "Configuring with Tracing enabled..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DSET_ENABLE_TRACING=ON
    
    echo "Compiling VTM (this may take a while)..."
    make -j$(nproc)
    cd ../..
    echo "VTM build complete."
}

[ -d "vtm" ] || build_vtm
