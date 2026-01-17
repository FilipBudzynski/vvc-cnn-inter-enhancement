#!/bin/bash

# 1. Clean start
[ -d "vtm" ] || git clone --depth 1 https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git vtm
cd vtm
rm -rf build
mkdir build
cd build

# 2. Use the official CMake flag defined in the repo's CMakeLists.txt
# The repository uses 'SET_ENABLE_TRACING' as the standard toggle.
# This will trigger the ARM-specific optimizations recently added to the cmake/ directory.
echo "Configuring VTM using repository-native ARM fixes..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DSET_ENABLE_TRACING=ON

# 3. Build the DecoderAnalyserApp
# Note: For research and block statistics, the VTM manual recommends the 
# 'Analyser' variant as it is pre-configured for data extraction.
echo "Compiling DecoderAnalyserApp..."
CORES=$(sysctl -n hw.logicalcpu)
make DecoderAnalyserApp -j$CORES

cd ../..
echo "Build complete. Binary: ./vtm/bin/DecoderAnalyserApp"
