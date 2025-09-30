# RareFoldGPCR
## Agonist Design Beyond Natural Amino Acids 

By transfer learning from [RareFold](https://github.com/patrickbryant1/RareFold) on
high-quality structurtes from [GPCRdb](https://gpcrdb.org/), we can learn how to incorporate
noncanonical amino acids (NCAAs) seen in RareFold to the structure prediction of GPCRs without ever having
seen NCAA-based GPCR modulators.

RareFoldGPCR (RFG) supports 49 different amino acid types.\
The 20 regular ones, and 29 **RARE** ones:
MSE, TPO, MLY, CME, PTR, SEP,SAH, CSO, PCA, KCX, CAS, CSD, MLZ, OCS, ALY, CSS, CSX, HIC, HYP, YCM, YOF, M3L, PFF, CGU,FTR, LLP, CAF, CMH, MHO

<img src="./data/RFG.svg"/>

# LICENSE
RareFoldGPCR is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  \
The RareFoldGPCR parameters for prediction and design are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). \
The design protocol is made available under the terms of the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

**You may not use these files except in compliance with the licenses.**


# Installation
The entire installation takes <1 hour on a standard computer. \
We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \
The runtime will depend on the GPU you have available and the size of the protein you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.

First install micromamba (you can also build from conda or mamba if you already have these installed)
https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

```
bash install_dependencies.sh
```

1. Install the RareFold environment
2. Get the RareFold parameters for single-chain structure prediction
3. Get the EvoBindRare parameters for binder design
4. Get Uniclust for MSA search
5. Install HHblits for MSA search


# Design using RareFoldGPCR
Run the test case (a few minutes)
```
micromamba activate rarefold
bash design.sh
```

If you want to use your target, simply replace the structure file for processing
in "design.sh" with a path for a new one. This will take care of all feature generation.
Note that RareFoldGPCR is purely sequence based, but can use structural input for e.g.
scaffolding or for biasing the design towards certain states. \
We have included the structure as a starting point here to make it easy for you to start designing.


# Citation
If you use RareFoldGPCR in your research, please cite



Li Q, Daumiller D, Bryant P. RareFold: Structure prediction and design of proteins with noncanonical amino acids. bioRxiv. 2025. p. 2025.05.19.654846. doi:10.1101/2025.05.19.654846
[link to paper](https://www.biorxiv.org/content/10.1101/2025.05.19.654846v1)

# Data
https://zenodo.org/uploads/15180406
