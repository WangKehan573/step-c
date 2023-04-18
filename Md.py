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


# load model
forcetut = './forcetut'
#model_path = os.path.join(forcetut, "best1_inference_model")
model_path = '/data/second_trainset/forcetut7'
best_model = torch.load(model_path)

# set up converter

from ase import io
atoms = io.read('2.xyz')



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

#load configuration file
from ase import io

# Generate a directory for the ASE computations
ase_dir = os.path.join(forcetut, 'ase_calcs_wet2')

if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)

# Write a sample molecule
molecule_path = os.path.join(ase_dir, 'mos2.xyz')
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

mos2_ase.optimize(fmax=1e-4)


#mos2_ase.compute_normal_modes()
#MD_simulation
mos2_ase.init_md('simulation_300K',
    temp_bath=300,
    reset=True)
mos2_ase.run_md(20000)

skip_initial = 5000

# Load logged results
results = np.loadtxt(os.path.join(ase_dir, 'simulation_300K.log'), skiprows=1)

# Determine time axis
time = results[skip_initial:,0]
#0.02585
# Load energies
energy_tot = results[skip_initial:,1]
energy_pot = results[skip_initial:,2]

# Construct figure
plt.figure(figsize=(14,6))

# Plot energies
plt.subplot(2,1,1)
plt.plot(time, energy_tot, label='Total energy')
plt.plot(time, energy_pot, label='Potential energy')
plt.ylabel('Energies [eV]')
plt.legend()

# Plot Temperature
temperature = results[skip_initial:,4]

# Compute average temperature
print('Average temperature: {:10.2f} K'.format(np.mean(temperature)))

plt.subplot(2,1,2)
plt.plot(time, temperature, label='Simulation')
plt.ylabel('Temperature [K]')
plt.xlabel('Time [ps]')
plt.plot(time, np.ones_like(temperature)*300, label='Target')
plt.legend()
plt.show()
