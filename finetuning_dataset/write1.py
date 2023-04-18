from ase import Atoms
import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
import numpy as np
from schnetpack.data import ASEAtomsData,AtomsDataModule
import ase

# load model



MoS2_1T = AtomsDataModule(
    os.path.join('MoS2_1T.db'),
    batch_size=100,
    num_train=8000,
    num_val=1000,
    num_test=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        
        trn.CastTo32()
    ],
    num_workers=1,
    pin_memory=True, # set to false, when not using a GPU
)


MoS2_1T.setup()

        
        
  
lattice = "Lattice=\"%f %f %f %f %f %f %f %f %f\""%(6.39818,0,0,-3.19909,5.540988 ,0,0,0,12.42199)


for i in range(200):
  print(i)

  structure = MoS2_1T.train_dataset[i]
  
  
  energy = structure[spk.properties.energy]*627.5#*27.2114
  numbers=structure[spk.properties.Z]
  positions=structure[spk.properties.R] 
  path = 'for_mace/'+str(i)+'.xyz'
  fout = open (path, 'w')
  
  positions,energy =  np.array(positions), float(energy)
  print(energy)
  force = np.random.randn(24,3)
  force = np.around(force,3)
  xyz = ""

  
  

  for  i  in range(24):
    



      if int(numbers[i]) == 42:
        token = 'Mo'
      else:
        token = 'S'
      
      xyz = xyz+"%s %s %s %s"%(token,positions[i][0],positions[i][1],positions[i][2])+" %s %s %s"%(force[i][0],force[i][1],force[i][2])+'\n'

    
    
  
  fout.write('24'+'\n')
  fout.write("energy="+str(energy)+" ")

  fout.write (lattice+" Properties=species:S:1:pos:R:3:forces:R:3")
  fout.write(' pbc="T T T"\n')

  
  fout.write(xyz)
  xyz = ""
  
