
# MEGO

This repo is the source code for the paper _Learning Mixture-of-Experts for General-Purpose Black-Box Discrete Optimization_. 
This page will tell you how to config the environment for the source code and run it.

## Quick Start

### Setup Environment

#### Python Environment
```shell
conda create -n test_env -q -y python=3.8
conda activate test_env
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Binary Lib compile
```shell
# You need to set the $MEGO as the root path of this project
# export MEGO="The Path of this project on your server"
# gcc we used is 11.2/11.4
# Install some package by apt
sudo apt install make gcc g++ libeigen3-dev libssl-dev swig git libboost-dev libasio-dev
# Download and install cmake>=3.14
cd $MEGO #The root path of the project
wget https://cmake.org/files/v3.22/cmake-3.22.4.tar.gz
tar xf cmake-3.22.4.tar.gz
cd cmake-3.22.4
./bootstrap --parallel=48
make -j 255
sudo make install 

#Download and install pybind11
cd $MEGO #The root path of the project
git clone https://github.com/pybind/pybind11.git
cd  pybind11
mkdir build
cd build
cmake ..
make check -j 255
sudo make install

#Download and install Boost
cd $MEGO #The root path of the project
wget https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
tar xf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh
sudo ./b2 install --prefix=/usr toolset=gcc threading=multi

# Update Dynamic Lib list 
sudo ldconfig
# Build the binary lib
# Modify the Python Environments !!! Important!!!
conda activate test_env
cd $MEGO/src/cpp/anchor_selection
cmake -DCMAKE_BUILD_TYPE=Release && make

cd $MEGO/src/cpp/com_imp/
cmake -DCMAKE_BUILD_TYPE=Release && make
```


### Dataset
There are 6 problem classes in this repo and 3 of them are generated according to the existing datasets. The datasets of 
anchor selection problem and complementary influence maximization problem have upload into the folder `data/dataset`, the dataset 
of compiler arguments optimization problem need to be downloaded by `ck` package which is installed while seting the environment.

#### Anchor Selection Problem
The dataset of ETH3D for Anchor Selection Problem is located in the folder **data/dataset/anchor_selection**.

#### Complementary Influence Maximization Problem
The dataset of Facebook/Wiki for Complementary Influence Maximization Problem is located in the folder **data/dataset/com_imp**.

#### Compiler Arguments Optimization 
You need to run the following instructions to download the dataset for compiler arguments optimization.
```shell
ck pull repo:ck-env
ck pull repo:ck-autotuning
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min
```

### Set the PYTHONPATH
Set the env_variable PYTHONPATH as: 
```shell
# You need to set the $MEGO as the root path of this project
# export MEGO="The Path of this project on your server"

export PYTHONPATH=$MEGO:$MEGO/src
```
While `$MEGO` is the root path of this project

### Generate Problem Instance
Run the `experiment_problem.py` in the `src` path
```shell
cd $MEGO/src
python experiment_problem.py
```

### Build Experts Models for Training Instances

```shell
cd $MEGO/src
python experiment_build_surrogate.py
```

### Build Decoder Mapping from Training Instance to Test Instance
```shell
cd $MEGO/src
python experiment_decoder_mapping.py
```

### Test Performance for the Initial Solutions
Test the performance of the initial solution for each test instance get by 
MEGO, and the result will be dumped to a pickle file.
```shell
cd $MEGO/src
python experiment_initial_solution.py
```

### Test Performance for the Baseline Methods
Test the performance of the baseline methods, the result will be
dumped to a picke file.

#### GA
```shell
cd $MEGO/src
python GA/experiment_GA.py
```

#### Hill Climbing
```shell
cd $MEGO/src
python hill_climbing/experiment_hill_climbing.py
```

#### Bayesian Optimization
```shell
cd $MEGO/src
python SMAC/experiment_SMAC.py
```

#### SMARTEST
```shell
cd $MEGO/src
python SMARTEST/experiment_SMARTEST.py
```

### A Reclassify of the Problem Class
Corresponding to the section `New perspective for problem classification` of the paper
```shell
cd $MEGO/src
python experiment_problem_reclassify.py
```


### Something Need Attention:
1. The default `max_thread_num` of this project is big, if your machine don't have enough CPU and memory, you need to decrease this number. This number is always set by a variable caled `max_parallel_num`, you can find them in the `experiment_X.py` files.
2. The experiments of `Compiler Arguments Optimization` will cost much time, because each evaluation of it need to compile a source program to a binary file and save it in the hard disk.
   - If you want to accelerate these experiments, you can create a `memory virtual hard disk` and create a soft link from it to the path `$MEGO/tmp`. It will cost less than `1G` RAM.
   - If you want to skip these experiments, you can delete or comment the `line 28` of `$MEGO/src/experiment_problem.py`.
