#!/bin/bash   

MY_DIR="$(dirname "$(readlink -f "${0}")")"
cd ${MY_DIR}

g++ -c hello_tf.cpp \
-fPIE -fPIC -g -O0 \
-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 \
-I "${HOME}/.local/lib/python3.5/site-packages/tensorflow/include" \
-I "${MY_DIR}" \
-o hello_tf.o  

g++ -pie hello_tf.o -g -O0 \
"${HOME}/.local/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so" \
"${HOME}/.local/lib/python3.5/site-packages/tensorflow/libtensorflow_framework.so.2" \
"${HOME}/.local/lib/python3.5/site-packages/_solib_k8/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib/libmklml_intel.so" \
"${HOME}/.local/lib/python3.5/site-packages/_solib_k8/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib/libiomp5.so" \
-lpython3.5m \
-Wl,--enable-new-dtags \
-Wl,-rpath ${HOME}/.local/lib/python3.5/site-packages/tensorflow/python \
-Wl,-rpath ${HOME}/.local/lib/python3.5/site-packages/tensorflow \
-Wl,-rpath ${HOME}/.local/lib/python3.5/site-packages/_solib_k8/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib \
-o hello_tf