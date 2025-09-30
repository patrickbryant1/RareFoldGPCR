#Install conda env
micromamba env create -f environment.yml

wait
micromamba activate rarefold
pip install -q --no-warn-conflicts numpy==1.26.4
pip install -q --no-warn-conflicts 'jax[cuda12_pip]'==0.4.35 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
micromamba deactivate

## Get network parameters for RareFoldGPCR (a few minutes)
ZENODO=https://zenodo.org/records/15180406/files
wget $ZENODO/complex_params26500.npy
mkdir data/params
mv complex_params26500.npy data/params/


wait
## Get Uniclust30 (10-20 minutes depending on bandwidth)
# 25 Gb download, 87 Gb extracted
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mkdir data/uniclust30
mv uniclust30_2018_08_hhsuite.tar.gz data
cd data
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
cd ..

wait
## Install HHblits (a few minutes)
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ../..

wait
