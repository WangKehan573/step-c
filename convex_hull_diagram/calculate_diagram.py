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
from ase import io
import pandas as pd

# load model
forcetut = './forcetut7'
model_path = forcetut
best_model = torch.load(model_path)

from ase import io
def readTxt1(filename):
    data = []
    with open(filename,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            #print(line)
            line = line.split()
                      
            data.append(line)
    #print(data)
    return data


filename = os.path.join('Mo3S4.txt')
print(filename)
#output_name =  os.path.join('C:/Users/kehan.wang/Desktop/trainset', member,'position.csv')
list1 = readTxt1(filename)
name = ["id","type","charge","fx","fy","fz"]
table = pd.DataFrame(columns=name,data=list1)
table = np.array(table)  
print(table)

position = table[:,3:]
position = np.float64(position)
print(position)

type1 = table[:,1]

type1[type1=='1'] = 42

type1[type1=='2'] = 16
type1 = np.float64(type1)
print(type1)

from ase import io
#atoms = io.read('/data/2H_inference/forcetut/ase_calcs/optimization.extxyz')
a = torch.tensor(type1)
b = torch.tensor(position)
atoms = Atoms(
    numbers=a, positions = b )
atoms.set_pbc((True,True,True))

#atoms.set_cell([[3.171846,0,0 ],[ 0,  3.171846,0],[0,0 , 3.171846  ]])  #Mo
#atoms.set_cell([[  6.081385,0,0 ],[   0,  3.233692 ,0],[0,-1.807431 ,   8.396760   ]])  #mo2s3
atoms.set_cell([[ 9.208688,0,0 ],[ -4.604344 ,  7.974958,0],[0,0 , 10.969303   ]]) 



#set up calculator

calculator = spk.interfaces.SpkCalculator(
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key='energy',
    force_key='forces',
    energy_unit="kcal/mol",
    position_unit="Ang",
)

atoms.set_calculator(calculator)

print("Prediction:")
print("energy:", atoms.get_total_energy())
print("forces:", atoms.get_forces())



# Generate a directory for the ASE computations
ase_dir =  'ase_Mo3S4'

if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

# Write a sample molecule
molecule_path = os.path.join(ase_dir, 'mos21.xyz')
io.write(molecule_path, atoms, format='xyz')


#ASEinterface
mos2_ase = spk.interfaces.AseInterface(
    molecule_path,
    ase_dir,
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key='energy',
    force_key='forces',
    energy_unit="kcal/mol",
    position_unit="Ang",
    device="cpu",
    dtype=torch.float64,
)


mos2_ase.optimize(fmax=4e-2)
