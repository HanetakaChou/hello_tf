hello_tf : hello_tf.o  
	g++ -pie hello_tf.o -g -O0 \
	"${LIB_DIR}/python3.5/dist-packages/tensorflow/python/_pywrap_tensorflow_internal.so" \
	"${LIB_DIR}/python3.5/dist-packages/tensorflow/libtensorflow_framework.so.2" \
	-L"${LIB_DIR}/python3.5/dist-packages/_solib_k8/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib" \
	-lmklml_intel \
	-liomp5 \
	-lpython3.5m \
	-Wl,--enable-new-dtags \
	-Wl,-rpath ${LIB_DIR}/python3.5/dist-packages/tensorflow/python \
	-Wl,-rpath ${LIB_DIR}/python3.5/dist-packages/tensorflow \
	-Wl,-rpath ${LIB_DIR}/python3.5/dist-packages/_solib_k8/_U@mkl_Ulinux_S_S_Cmkl_Ulibs_Ulinux___Uexternal_Smkl_Ulinux_Slib \
	-o hello_tf

hello_tf.o : hello_tf.cpp  
	g++ -c hello_tf.cpp \
	-fPIE -fPIC -g -O0 \
	-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 \
	-I "${LIB_DIR}/python3.5/dist-packages/tensorflow/include" \
	-I "${MY_DIR}" \
	-o hello_tf.o  