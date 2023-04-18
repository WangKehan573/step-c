import numpy as onp

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

def structure_energy_training_data(name_dict, training_items):
    import copy
    max_len = 5

    all_weights = []
    all_energy_vals = []

    sys_list_of_lists = []
    multip_list_of_lists = []
    for i, item in enumerate(training_items):

        name_list, w, multiplier_list, energy = item
        # deep copy not to affect the orig. data structures
        multiplier_list = copy.deepcopy(multiplier_list)
        index_list = []
        new_energy = energy
        exist = True
        for multip,name in zip(multiplier_list,name_list):
            if name not in name_dict:
                exist = False
                print("{} does not exist in the geo file, skipping!", name)
                break
        if exist:
            for multip,name in zip(multiplier_list,name_list):
                ind = name_dict[name]


                index_list.append(ind)
                # just to have fixed size length, filler ones will be zeroed out
            if len(index_list) <= max_len:
                cur_len = len(index_list)
                for _ in range(max_len - cur_len):
                    index_list.append(0)
                    multiplier_list.append(0)
                sys_list_of_lists.append(index_list)
                multip_list_of_lists.append(multiplier_list)
            all_weights.append(w)
            all_energy_vals.append(new_energy)
    return onp.array(sys_list_of_lists,dtype=onp.int32), onp.array(multip_list_of_lists,dtype=TYPE), onp.array(all_weights,dtype=TYPE), onp.array(all_energy_vals,dtype=TYPE)
    

def preprocess_trainset_line(line):
    # to make sure everything is nicely seperated by a space
    line = line.replace('/', ' / ')
    # it changes the weight as well, so removed
    #line = line.replace('+', ' + ')
    #line = line.replace('-', ' - ')

    return line
    
training_items,training_items_str = read_train_set('trainset1.in')
print(training_items,training_items_str )
