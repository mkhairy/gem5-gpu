#########################################################################
# Setup a cuda and gcc version that works with CUDA 4.2
#########################################################################
source /usr/local/gpgpu-sim-setup/common/export_gcc_version.sh 4.4.7
#export CUDAHOME=/usr/local/cuda-4.2/
#export PATH=$PATH:/usr/local/cuda-4.2/bin/
export CUDAHOME=/home/mkhairy/pWork/cuda3.2/cuda/
export PATH=$PATH:/home/mkhairy/pWork/cuda3.2/cuda/bin/
export NVIDIA_CUDA_SDK_LOCATION=/home/mkhairy/pWork/cuda3.2/SDK/C/
