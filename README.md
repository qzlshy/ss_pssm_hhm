# second_structure_pssm_hhm
A context model to prediction protein secondary structure.<br>
The *.npy files in train_context_cullpdb are date set we used. They are uploaded with lfs, so you should install git-lfs to download *.npy files.<br>
The *.npy files can be loaded with python as:<br>
```
import numpy as np
data=np.load('filename.npy').item()
name=data['name']
seq=data['seq']
pssm=data['pssm']
dssp=data['dssp']
hhm=data['hhm']
```
Name is the protein structures name, seq is the sequence of structures, pssm is the psi-blasts profiles of sequences, dssp is the second structure of sequcences, hhm is the HHBlits profiles.<br>
All the sequences in Cullpdb, CB513, CASP12 and CASP13 datasets are culled with CD-HIT server, sequence that had more than 25\% identity to any sequences in the datasets was removed. So the sequence identity in all the datasets are less than 25\%.<br>
