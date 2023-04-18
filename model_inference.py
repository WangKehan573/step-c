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
import torch 
import crystal
from torch.autograd import Variable

# load model
forcetut = '/common-data/kehanwang_data/second_trainset/'
model_path = os.path.join(forcetut, "forcetut7")
best_model = torch.load(model_path)


# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32
)


#early stopping
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


#setup trainset and valset
geo_list = os.listdir('/data/second_trainset/Mo/')
#print(geo_list)
'''
geo_list = ['Mo_0%',
'Mo_100%_2',
'Mo_100%_1',
'Mo_62.5%',
'Mo_50%_1',
'Mo_50%_2',
'Mo_37.5%',
'Mo_25%_1',
'Mo_25%_2']

valgeo_list = ['S_100%',
'S_50%_1',
'S_50%_2',
'S_87.5%',
'S_75%_1',
'S_75%_2',
'S_75%_3',
'S_75%_4',
'S_62.5%_1',
'S_62.5%_2',
'S_37.5%_1',
'S_37.5%_2']
'''

atoms_list = []
DFT_energy_list = []
def calculate_energy(geo_list):
  energy_dict={}
  atoms_dict ={}
  for geo in geo_list:
    
    file_name = os.path.join('/data/second_trainset/Mo/', geo)

    i = open (file_name, 'r')

    fin = i.readlines()

    positions = []
    count = 0
    type1 = []
    
   
    for line in fin:

      token = line.split()

     
      if (line.find ("DESCRP") >= 0):
        header = "XYZ %s\n"%(token[1])
        count = 0
    

      elif (line.find ("CRYSTX") >= 0):
        a = token[1]
        b = token[2]
        c = token[3]
        
        alpha = token[4]
        beta = token[5]
        gamma = token[6]
        
        if float(gamma) == 90.0:
          cell = np.array([[float(a),0,0],[0,float(b),0],[0,0,float(c)]])
        else:
          cry = crystal.Crystal(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)

          cell = np.array([[cry.vecb[1],cry.vecb[0],cry.veca[2]],[cry.veca[1],cry.veca[0],cry.vecb[2]],[cry.vecc[0],cry.vecc[1],cry.vecc[2]]])
     

      elif (line.find ("HETATM") >= 0):
         positions.append([float(token[3]),float(token[4]),float(token[5])])
         if token[2] == 'Mo':
             type1.append(42)
         else:
             type1.append(16)
  
      elif token!= [] and token[0] == 'ENERGY':
          DFT_energy = float(token[-1])
      
      
    positions = np.array(positions)
    type1 = np.array(type1)
    #print(geo,cell)



    a = torch.tensor(type1)
    b = torch.tensor(positions)
    c = torch.tensor(cell)
    atoms = Atoms(
        numbers=a, positions = b )
    atoms.set_pbc((True,True,True))

    atoms.set_cell(c)

    geo = os.path.splitext(geo)[0]
    atoms_dict[geo] = atoms
    
    energy_dict[geo] = DFT_energy
    

  return atoms_dict,energy_dict

 
atoms_list,DFT_energy_list = calculate_energy(geo_list)
#val_list,val_energy_list = calculate_energy(valgeo_list)
#print(atoms_list)

'''
for name, param in best_model.named_parameters():
      if param.requires_grad:
          print(name)
'''


#preprocess_trainset_line

def preprocess_trainset_line(line):
    # to make sure everything is nicely seperated by a space
    line = line.replace('/', ' / ')
    # it changes the weight as well, so removed
    #line = line.replace('+', ' + ')
    #line = line.replace('-', ' - ')

    return line


def read_train_set(train_in):
    f = open(train_in, 'r')
    training_items = {}
    training_items_str = {}
    energy_flag = 0
    
    energy_items = []


    
    energy_items_str = []

    for line in f:
        #print(line)
        line = line.strip()
        # ignore everything after #
        line = line.split('#', 1)[0]
        line = line.split('!', 1)[0]
        if len(line) == 0 or line.startswith("#"):
            continue
        elif line.startswith("ENERGY"):
            energy_flag = 1


        elif line.startswith("ENDENERGY"):
            #training_items['ENERGY'] = energy_items
            energy_flag = 0

        elif energy_flag == 1:
            line = preprocess_trainset_line(line)
            split_line = line.split()
            num_ref_items = int((len(split_line) - 2) / 4) # w and energy + 4 items per ref. item

            name_list = []
            multiplier_list = []

            w = float(split_line[0])
            for i in range(num_ref_items):
                div = float(split_line[4 * i + 4].strip())
                mult = 1/div
                if split_line[1 + 4*i].strip() == '+':
                    multiplier_list.append(mult)
                else:
                    multiplier_list.append(-mult)


                name_list.append(split_line[4 * i + 2].strip())

            energy = float(split_line[-1])

            energy_items.append((name_list,w,multiplier_list, energy))
            energy_items_str.append(line)


    if len(energy_items) > 0:
        training_items = energy_items
        training_items_str = energy_items_str



    return training_items,training_items_str


    

valid_items,valid_items_str = read_train_set('trainset1.in')


optimizer = torch.optim.AdamW(best_model.parameters(), lr=0.001)



criterion = torch.nn.MSELoss()
early_stopping = EarlyStopping(patience=2)
 
  

def val_loss(model,atoms_list,DFT_energy_list,valid_items):
    valid_loss = 0
    optimizer.zero_grad()
   
    num = len(valid_items)
    print(num)
    atom_dict = {}
    for key, value in atoms_list.items():   
      inputs1 = converter(value)      
      y_hat1 = model(inputs1)   
      atom_dict[key] = y_hat1#['energy']
      

    for i in range(num):
      name_list,w,multiplier_list, energy = valid_items[i]
      j = len(name_list)
      gd_difference = 0
      pred_difference = 0
      for k in range(j):
        geo = name_list[k]
        gd_difference += multiplier_list[k]*DFT_energy_list[geo] 
        
       
        pred_difference += multiplier_list[k]* atom_dict[geo]['energy']
      gd_difference = torch.tensor(gd_difference,dtype = torch.float64)
      
      
      #print(name_list,pred_difference,gd_difference)
      loss = criterion(pred_difference,gd_difference)
      print(name_list,float(loss))
      
      valid_loss += loss
    loss_mean = valid_loss/(num) 
    
    print('Loss: {:.6f}'.format(loss_mean.item()))
   
  
    return valid_loss
      
    

(val_loss(best_model,atoms_list,DFT_energy_list,valid_items)) #val_list,val_energy_list

        

MoS2_1T = AtomsDataModule(
    os.path.join('/data/2H_inference/MoS2_1T.db'),
    batch_size=1,
    num_train=1,
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
structure = MoS2_1T.train_dataset[0]
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)
atoms.set_pbc((True,True,True))
atoms.set_cell([[6.39818,0,0],[-3.19909,5.540988 ,0],[0,0,12.42199]])
# convert atoms to SchNetPack inputs and perform prediction
inputs = converter(atoms)
results = best_model(inputs)

print(results['energy'])
print(structure['energy'])


print(results['forces'])
print(structure['forces'])
  
        
