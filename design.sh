#Here is an example for designing towards GLP1R using the
#PDB file 6X18. The structural information can be used from this
#file by setting what part of the natural peptide agonist (GLP-1) to scaffold.
#It can also be ignored (e.g. for de novo cyclic design) by setting scaffold to "NA"

##########First, let's process the input we need for the design##########
PRED_ID=6X18
DATA_DIR=./data/gpcrdb/design_test_case/$PRED_ID
GPCR_COMPLEX=$DATA_DIR/6X18.cif #https://www.rcsb.org/3d-view/6X18

#1. Extract the chains you want from the complex
REC_CHAIN=R
PEP_CHAIN=P #If not using - leave empty

python3 ./src/preprocess/read_write_complex.py
#2. Extract fasta sequence from the target for design





MSA_FEATS=$DATA_DIR/msa_features.pkl
NUM_REC=3 #For difficult receptors (low plDDT) - run with 8
BIND_SEQ='HIS-ALA-GLU-GLY-THR-PHE-THR-SER-ASP-VAL-SER-SER-TYR-LEU-GLU-GLY-GLN-ALA-ALA-LYS-GLU-PHE-ILE-ALA-TRP-LEU-VAL-LYS-GLY-ARG'
BIND_LENGTH=30
NITER=1000
SCAFFOLD='nterm' #centre/nterm/cterm/NA - decides what part of the native peptide (chain B) to scaffold
NUM_SC_RESIS=5
STRUCT_FEATS=$DATA_DIR/complex/complex_structure_features.pkl
BS=1
PARAMS=/home/bryant/Desktop/dual_design/gpcrdb/params/params26500.npy
RARE_AAS="MSE,MLY,PTR,SEP,TPO,MLZ,ALY,HIC,HYP,M3L,PFF,MHO"
CYC_OFFSET=False
NUM_CLUSTS=128
OUTDIR=../../data/gpcrdb/design_test_case/

python3 ./design_scaffold_struc_feats.py --predict_id $PRED_ID \
--MSA_feats $MSA_FEATS \
--num_recycles $NUM_REC \
--binder_sequence $BIND_SEQ \
--binder_length $BIND_LENGTH \
--num_iterations $NITER \
--scaffold_mode $SCAFFOLD \
--num_resis_to_scaffold $NUM_SC_RESIS \
--structure_feats $STRUCT_FEATS \
--batch_size $BS \
--params $PARAMS \
--rare_AAs $RARE_AAS \
--cyclic_offset $CYC_OFFSET \
--num_clusters $NUM_CLUSTS \
--outdir $OUTDIR
