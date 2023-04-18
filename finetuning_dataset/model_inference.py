from ase import Atoms
import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
from schnetpack.data import ASEAtomsData
from datamodule import AtomsDataModule

# load model
forcetut = '/data/second_trainset'
model_path = os.path.join(forcetut, "forcetut_init9")
best_model = torch.load(model_path)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32
)



MoS2_1T = AtomsDataModule(
    os.path.join('MoS2_1T.db'),
    batch_size=100,
    num_train=200,
    num_val=0,
    num_test=0,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        
        trn.CastTo32()
    ],
    num_workers=1,
    pin_memory=True, # set to false, when not using a GPU
)


MoS2_1T.setup()

# create atoms object from dataset
result1 = []
structure1 = []
for  i  in range(200):
        structure = MoS2_1T.train_dataset[i]
        atoms = Atoms(numbers=structure[spk.properties.Z], positions=structure[spk.properties.R])
        atoms.set_pbc((True,True,True))
        atoms.set_cell([[6.39818,0,0],[-3.19909,5.540988 ,0],[0,0,12.42199]])
        inputs = converter(atoms)
        results = best_model(inputs)
        structure['energy'] = torch.tensor(structure['energy'],dtype = torch.float64)
        print(i)
        result1.append(float(results['energy']))
        structure1.append(float(structure['energy']))
        
        
filename = open('ft.txt', 'w')
for value in result1:
  filename.write(str(value))
  filename.write('\n')
filename.close()

filename = open('origin.txt', 'w')
for value in structure1:
  filename.write(str(value))
  filename.write('\n')
filename.close()
