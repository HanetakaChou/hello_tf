#!/bin/bash   

MY_DIR="$(dirname "$(readlink -f "${0}")")"
cd ${MY_DIR}

LIB_DIR=/usr/local/lib #ubuntu

make -j16 MY_DIR=${MY_DIR} LIB_DIR=${LIB_DIR} 