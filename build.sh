#!/bin/bash   

MY_DIR="$(dirname "$(readlink -f "${0}")")"
cd ${MY_DIR}

make -j16 MY_DIR=${MY_DIR}