#########################################################################
# Setup a cuda and gcc version that works with CUDA 3.2 and gcc 4.9.2
#########################################################################
source /usr/local/gpgpu-sim-setup/common/export_gcc_version.sh 4.9.2
export CUDAHOME=/home/mkhairy/pWork/cuda3.2/cuda/
export PATH=$PATH:/home/mkhairy/pWork/cuda3.2/cuda/bin/
export NVIDIA_CUDA_SDK_LOCATION=/home/mkhairy/pWork/cuda3.2/SDK/C/
