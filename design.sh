#Here is an example for designing towards GLP1R using the
#PDB file 6X18. The structural information can be used from this
#file by setting what part of the natural peptide agonist (GLP-1) to scaffold.
#It can also be ignored (e.g. for de novo cyclic design) by setting scaffold to "NA"
BASE=. #Change this depending on your local path
##########First, let's process the input we need for the design##########
PRED_ID=6X18
DATA_DIR=$BASE/data/design_test_case/$PRED_ID
GPCR_COMPLEX=$DATA_DIR/6X18.cif #https://www.rcsb.org/3d-view/6X18

#1. Extract the chains you want from the complex
#This also writes fasta files for MSA generation
REC_CHAIN=R
PEP_CHAIN=P #If not using - leave empty
python3 $BASE/src/preprocess/read_write_complex.py --structure $GPCR_COMPLEX \
--rec_chain $REC_CHAIN --pep_chain $PEP_CHAIN --outdir $DATA_DIR/

#2. Create an MSA with HHblits
REC_FASTA=$DATA_DIR'/receptor.fasta'
REC_MSA=$DATA_DIR'/receptor.a3m'
HHBLITSDB=$BASE/data/uniclust30_2018_08/uniclust30_2018_08
if test -f $REC_MSA; then
	echo $REC_MSA exists
else
	$BASE/hh-suite/build/bin/hhblits -i $REC_FASTA -d $HHBLITSDB -E 0.001 -all -n 2 -oa3m $REC_MSA
fi

#3. Make iput structural feats (if using) - these are the feats used for trainging as well
python3 $BASE/src/make_complex_gt_structural_feats.py --input_pdb $DATA_DIR/extracted_complex.pdb \
--outdir $DATA_DIR/

#4. Make MSA feats
python3 $BASE/src/make_msa_seq_feats.py --input_fasta_path $REC_FASTA \
--input_msas $REC_MSA --outdir $DATA_DIR/


MSA_FEATS=$DATA_DIR/receptor_msa_features.pkl
NUM_REC=3 #For difficult receptors (low plDDT) - run with 8
#You can find the extracted bind sec and binder length in the peptide fasta (./data/design_test_case/6X18/peptide.fasta)
#Keep this empty if not using or set the scaffold type to "NA"
BIND_SEQ='HIS-ALA-GLU-GLY-THR-PHE-THR-SER-ASP-VAL-SER-SER-TYR-LEU-GLU-GLY-GLN-ALA-ALA-LYS-GLU-PHE-ILE-ALA-TRP-LEU-VAL-LYS-GLY-ARG'
BIND_LENGTH=30 #This information is also in ./data/design_test_case/6X18/peptide.fasta
NITER=1000 #How many iterations to run
SCAFFOLD='nterm' #centre/nterm/cterm/NA - decides what part of the native peptide (chain B) to scaffold
NUM_SC_RESIS=20
STRUCT_FEATS=$DATA_DIR/complex_structure_features.pkl
BS=1 #Batch size - you can usually run many independent threads on one GPU
PARAMS=$BASE/data/params/complex_params26500.npy
RARE_AAS="MSE,MLY,PTR,SEP,TPO,MLZ,ALY,HIC,HYP,M3L,PFF,MHO" #Pick from MSE, TPO, MLY, CME, PTR, SEP,SAH, CSO, PCA, KCX, CAS, CSD, MLZ, OCS, ALY, CSS, CSX, HIC, HYP, YCM, YOF, M3L, PFF, CGU,FTR, LLP, CAF, CMH, MHO
CYC_OFFSET=False #Cyclic or not - bad idea to use with scaffolding
NUM_CLUSTS=128
OUTDIR=../../data/gpcrdb/design_test_case/

python3 $BASE/src/design_scaffold_struc_feats.py --predict_id $PRED_ID \
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
