#! /bin/bash

build_vvenc() {
	git clone https://github.com/fraunhoferhhi/vvenc
	mkdir -p vvenc/build
	cd vvenc/build
	cmake .. -DCMAKE_BUILD_TYPE=Release
	make $1
	cd ../..
}

# build_vvdec() {
# 	git clone https://github.com/fraunhoferhhi/vvdec
# 	mkdir -p vvdec/build
# 	cd vvdec/build
# 	cmake .. -DCMAKE_BUILD_TYPE=Release
# 	make $1
# 	cd ../..
# }

[ -d "vvenc" ] || build_vvenc $1
# [ -d "vvdec" ] || build_vvdec $1
