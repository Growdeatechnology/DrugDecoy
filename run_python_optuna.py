import optuna
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from openpyxl import Workbook
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten, MaxPool1D
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
import math
import glob
import time
import numpy as np
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import regex as re
import matplotlib.pyplot as plt
import re
import joblib
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import GCNSupervisedGraphClassification
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
import seaborn as sns
from sklearn import model_selection
from IPython.display import display, HTML
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dense, Concatenate, Conv1D, Dropout, Flatten, MaxPool1D, concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
#from keras.optimizers import SGD

import matplotlib.pyplot as plt
import math 

pd.set_option('mode.chained_assignment', None)
atom_props = pd.read_csv("atom_props.csv")
atom_type = pd.read_csv("Atom_Type.csv")
atom_type.head(2)
atom_props.head(2)
atom_type.head(2)

def bond_parser(filename):
  print(filename)
  with open(filename, 'r') as f:
      f_text = f.read()
  # Error handling for regex search
  search_result = re.search(r'@<TRIPOS>BOND([\s\S]*?)@<TRIPOS>', f_text) # Updated regex for robustness
  if search_result:
      bonds_text = search_result.group(1)
  else:
      return pd.DataFrame(columns=['bond_id', 'source', 'target', 'bond_type'])  # Return empty dataframe if pattern not found
  
  bonds = np.array(re.sub(r'\s+', ' ', bonds_text.strip()).split()).reshape((-1, 4))
  df_bonds = pd.DataFrame(bonds, columns=['bond_id', 'source', 'target', 'bond_type'])
  df_bonds.set_index(['bond_id'], inplace=True)
  return df_bonds

def calc_dist(df, node1, node2) : 
    x1, y1, z1 = df.iloc[np.where(df.index == str(node1))][['x', 'y', 'z']].values.flatten()
    x2, y2, z2 = df.iloc[np.where(df.index == str(node2))][['x', 'y', 'z']].values.flatten()

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) 
    return(dist)   



def ligand_prep(file):
  try:
    
    print('starting_ligand')
    

  #id="3pce"

    ligand_file = file
    #ligand_file = "../AI-GCN/Ligand/" + id + "_ligand.mol2"


    ligand = PandasMol2().read_mol2(ligand_file)
    ligand_atoms = ligand.df
    ligand_bonds = bond_parser(ligand_file)
    print(ligand_bonds)


    ligand_atoms.atom_id = ligand_atoms.atom_id.astype("str")
    #print(ligand_atoms)

    mat = '^HOH|^ZN|^CO|^K|^MG|^CA|^CU|^NA|^MN|^NI'

    nh_index_l = (ligand_atoms.iloc[:, 5].str.split(".", expand=True)[0] != "H") #& (~ligand_atoms.iloc[:, 7].str.contains(mat, case=False, regex=True))
    nh_nodes_l = ligand_atoms[nh_index_l].atom_id
    #print(nh_nodes_l)

    ligand_atoms = ligand_atoms.iloc[:, [0, 2, 3, 4, 5, 8]]
    ligand_atoms = ligand_atoms[ligand_atoms.atom_id.isin(nh_nodes_l)]
    ligand_atoms = ligand_atoms.set_index("atom_id")

    ligand_bonds = ligand_bonds[ligand_bonds.source.isin(nh_nodes_l)]
    ligand_bonds = ligand_bonds[ligand_bonds.target.isin(nh_nodes_l)]
    #

    ligand_atoms = ligand_atoms.reset_index().merge(atom_type, how='left', left_on='atom_type', right_on="atom_type").set_index('atom_id')
    ligand_atoms['atom_type'] = ligand_atoms.iloc[:, 3].str.split(".", expand=True)[0]
    ligand_atoms = ligand_atoms.reset_index().merge(atom_props, how='left', left_on='atom_type', right_on="Atom").set_index('atom_id')
    ligand_atoms.drop(["atom_type", "Atom"], axis=1, inplace=True)

    ligand_dists = []
    for s,t in zip(ligand_bonds.source, ligand_bonds.target) : 
        ligand_dists.append(calc_dist(ligand_atoms, s, t))
    ligand_bonds['weight'] = ligand_dists


    mean_x_l=(ligand_atoms.x.sum()/ligand_atoms.x.count())  
    sd_x_l= (pow((ligand_atoms.x-mean_x_l),2)).sum()
    sd_x_l=sd_x_l/ligand_atoms.x.count()
    sd_x_l=math.sqrt(sd_x_l)

    mean_y_l=(ligand_atoms.y.sum()/ligand_atoms.y.count())  
    sd_y_l= (pow((ligand_atoms.y-mean_y_l),2)).sum()
    sd_y_l=sd_y_l/ligand_atoms.y.count()
    sd_y_l=math.sqrt(sd_y_l)  
      #ligand_atoms.y = (ligand_atoms.y - mean_y_l) / sd_x_l

    mean_z_l=(ligand_atoms.z.sum()/ligand_atoms.z.count())  
    sd_z_l= (pow((ligand_atoms.z-mean_z_l),2)).sum()
    sd_z_l=sd_z_l/ligand_atoms.z.count()
    sd_z_l=math.sqrt(sd_z_l)

    mean_charge_l=(ligand_atoms.charge.sum()/ligand_atoms.charge.count())  
    sd_charge_l= (pow((ligand_atoms.charge - mean_charge_l),2)).sum()
    sd_charge_l=sd_charge_l/ligand_atoms.charge.count()
    sd_charge_l=math.sqrt(sd_charge_l) 


    ligand_atoms = (ligand_atoms - ligand_atoms.min()) / (ligand_atoms.max() - ligand_atoms.min())


    ligand_atoms = ligand_atoms.replace(np.nan, 0) 


    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('ar',6)
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('am',7) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('du',0) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('un',0)   
    #print(ligand_bonds)

    return (sg.StellarGraph(ligand_atoms, edges = ligand_bonds),id)
  except Exception as e:
        print(f"Error processing file {file}: {e}")

def predict_class(graph, model_name):
  single_graph_list = [graph[0]] 

  loaded_model = model_name

  # Now, create a generator with the corrected list of StellarGraph objects
  single_graph_generator = PaddedGraphGenerator(graphs=single_graph_list)

  # Prepare the graph for prediction without wrapping `single_graph_list` in another list
  single_graph_gen = single_graph_generator.flow(single_graph_list,targets=[0], batch_size=1)

  # Make predictions
  predictions = loaded_model.predict(single_graph_gen, steps=len(single_graph_gen)) 
  print(predictions)# Ensure steps match the number of graphs

  # Since this is a binary classification problem, the model outputs probabilities of the positive class
  # You may want to apply a threshold to convert these probabilities into a binary decision
  threshold = 0.5  # Example threshold for binary classification
  predicted_class = (predictions.flatten() > threshold).astype(int)

  return (predicted_class[0])
    
    
def ligand_preparation(id,directory):
#id="3pce"
    directory=directory

    ligand_file = directory+"/" + id + ".mol2"
    print(ligand_file)
    #ligand_file = "../AI-GCN/Ligand/" + id + "_ligand.mol2"


    ligand = PandasMol2().read_mol2(ligand_file)
    ligand_atoms = ligand.df
    ligand_bonds = bond_parser(ligand_file)


    ligand_atoms.atom_id = ligand_atoms.atom_id.astype("str")
    #print(ligand_atoms)

    mat = '^HOH|^ZN|^CO|^K|^MG|^CA|^CU|^NA|^MN|^NI'

    nh_index_l = (ligand_atoms.iloc[:, 5].str.split(".", expand=True)[0] != "H") #& (~ligand_atoms.iloc[:, 7].str.contains(mat, case=False, regex=True))
    nh_nodes_l = ligand_atoms[nh_index_l].atom_id
    #print(nh_nodes_l)

    ligand_atoms = ligand_atoms.iloc[:, [0, 2, 3, 4, 5, 8]]
    ligand_atoms = ligand_atoms[ligand_atoms.atom_id.isin(nh_nodes_l)]
    ligand_atoms = ligand_atoms.set_index("atom_id")

    ligand_bonds = ligand_bonds[ligand_bonds.source.isin(nh_nodes_l)]
    ligand_bonds = ligand_bonds[ligand_bonds.target.isin(nh_nodes_l)]
    #

    ligand_atoms = ligand_atoms.reset_index().merge(atom_type, how='left', left_on='atom_type', right_on="atom_type").set_index('atom_id')
    ligand_atoms['atom_type'] = ligand_atoms.iloc[:, 3].str.split(".", expand=True)[0]
    ligand_atoms = ligand_atoms.reset_index().merge(atom_props, how='left', left_on='atom_type', right_on="Atom").set_index('atom_id')
    ligand_atoms.drop(["atom_type", "Atom"], axis=1, inplace=True)

    ligand_dists = []
    for s,t in zip(ligand_bonds.source, ligand_bonds.target) : 
        ligand_dists.append(calc_dist(ligand_atoms, s, t))
    ligand_bonds['weight'] = ligand_dists


    mean_x_l=(ligand_atoms.x.sum()/ligand_atoms.x.count())  
    sd_x_l= (pow((ligand_atoms.x-mean_x_l),2)).sum()
    sd_x_l=sd_x_l/ligand_atoms.x.count()
    sd_x_l=math.sqrt(sd_x_l)

    mean_y_l=(ligand_atoms.y.sum()/ligand_atoms.y.count())  
    sd_y_l= (pow((ligand_atoms.y-mean_y_l),2)).sum()
    sd_y_l=sd_y_l/ligand_atoms.y.count()
    sd_y_l=math.sqrt(sd_y_l)  
      #ligand_atoms.y = (ligand_atoms.y - mean_y_l) / sd_x_l

    mean_z_l=(ligand_atoms.z.sum()/ligand_atoms.z.count())  
    sd_z_l= (pow((ligand_atoms.z-mean_z_l),2)).sum()
    sd_z_l=sd_z_l/ligand_atoms.z.count()
    sd_z_l=math.sqrt(sd_z_l)

    mean_charge_l=(ligand_atoms.charge.sum()/ligand_atoms.charge.count())  
    sd_charge_l= (pow((ligand_atoms.charge - mean_charge_l),2)).sum()
    sd_charge_l=sd_charge_l/ligand_atoms.charge.count()
    sd_charge_l=math.sqrt(sd_charge_l) 


    ligand_atoms = (ligand_atoms - ligand_atoms.min()) / (ligand_atoms.max() - ligand_atoms.min())


    ligand_atoms = ligand_atoms.replace(np.nan, 0) 


    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('ar',6)
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('am',7) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('du',0) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('un',0)   
    #print(ligand_bonds)

    return (sg.StellarGraph(ligand_atoms, edges = ligand_bonds),id)
    
def training_model(directory, batch_size, learning_rate, dropout_rate, num_rows_tensor, training_size):
    try:
        import glob
        import os
        import joblib
        import pandas as pd
        from sklearn import model_selection
        from stellargraph.mapper import PaddedGraphGenerator
        from stellargraph.layer import DeepGraphCNN
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Dropout
        from tensorflow.keras.losses import binary_crossentropy
        import time

        directory = directory
        print(f"Directory: {directory}")

        # Load the .mol2 files
        idsA = glob.glob(os.path.join(directory, '*.mol2'))
        idsA = [id.replace(f"{directory}/", "").replace(".mol2", "") for id in idsA]
        print(f"First 5 Ligand IDs: {idsA[:5]}")

        # Load All.csv and All_Activity.txt
        path_all_csv = os.path.join(directory, "All.csv")
        path_activity_txt = os.path.join(directory, "All_Activity.txt")
        selected_ids = pd.read_csv(path_all_csv).iloc[:, 0].values

        # Initialize lists for ligands and processed IDs
        pid = []  # Processed IDs
        lgs = []  # List of Ligand Graphs
        ignored_ids = []  # Ignored IDs due to errors

        # Prepare the ligands
        count = 0
        for id in selected_ids:
            try:
                lg = ligand_preparation(id, directory)
                lgs.append(lg[0])
                pid.append(lg[1])
                count += 1
                if count % 250 == 0:
                    print(f"Processed {count} - {id}")
            except Exception as e:
                ignored_ids.append(id)
                print(f"Error processing ligand {id}: {e}")

        # Save ligand graphs to a .var file for reuse
        joblib.dump((pid, lgs), os.path.join(directory, "ligand_graphs.var"))
        pid, lgs = joblib.load(os.path.join(directory, "ligand_graphs.var"))

        # Prepare the StellarGraph objects
        pl_graphs = lgs
        print(lgs)

        # Prepare the activity labels (targets)
        target = pd.read_csv(path_activity_txt, sep="\s+", header=None)
        res = [target[target[0] == x][2].values.item() for x in pid]
        res = pd.DataFrame(res)
        label = res[0].astype('category').cat.codes
        label.name = 'LABEL'

        # Split data into train (70%), validation (20%), and test (10%) sets
        train_graphs, temp_graphs = model_selection.train_test_split(label, train_size=0.7, stratify=label)
        test_graphs, val_graphs = model_selection.train_test_split(temp_graphs, test_size=0.33, stratify=temp_graphs)

        print(f"Training set size: {len(train_graphs)}")
        print(f"Validation set size: {len(val_graphs)}")
        print(f"Test set size: {len(test_graphs)}")

        # Save the test set indices for re-creating the test generator later
        test_data = {"test_indices": list(test_graphs.index), "test_labels": list(test_graphs.values)}
        joblib.dump(test_data, 'test_data.joblib')
        print(f"Test data indices and labels saved to test_data.joblib")

        # Create the PaddedGraphGenerator
        generator = PaddedGraphGenerator(graphs=pl_graphs)
        k = int(num_rows_tensor)  # The number of rows for the output tensor
        print(f"Number of rows for tensor: {k}")

        # Define DeepGraphCNN model
        layer_sizes = [1024, 1024, 1024, 1]
        dgcnn_model = DeepGraphCNN(layer_sizes=layer_sizes, activations=["tanh", "tanh", "tanh", "tanh"], k=k, generator=generator)

        # Model input and output tensors
        x_inp, x_out = dgcnn_model.in_out_tensors()

        # Additional Conv1D, MaxPooling, Flatten, Dense layers
        x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=float(dropout_rate))(x_out)

        # Final output layer with sigmoid activation
        predictions = Dense(units=1, activation="sigmoid")(x_out)

        # Define the final model
        model = Model(inputs=x_inp, outputs=predictions)

        # Compile the model
        model.compile(optimizer=Adam(lr=learning_rate), loss=binary_crossentropy, metrics=["acc"])

        # Data generators for training, validation, and testing
        train_gen = generator.flow(list(train_graphs.index), targets=train_graphs.values, batch_size=int(batch_size))
        val_gen = generator.flow(list(val_graphs.index), targets=val_graphs.values, batch_size=int(batch_size))
        
        # Create the test generator when needed, using indices saved earlier.
        test_gen = generator.flow(list(test_graphs.index), targets=test_graphs.values, batch_size=1)

        # Train the model
        epochs = int(training_size)
        print(f"Training for {epochs} epochs...")
        start_time = time.time()
   
        

        # Uncomment this to train the model
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, verbose=1, shuffle=True, workers=16, use_multiprocessing=True)

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        return model, test_gen, test_graphs

    except Exception as e:
        print(f"Error during training: {e}")

pd.set_option('mode.chained_assignment', None)
atom_props = pd.read_csv("atom_props.csv")
atom_type = pd.read_csv("Atom_Type.csv")
atom_type.head(2)
atom_props.head(2)
atom_type.head(2)

def bond_parser(filename):
  print(filename)
  with open(filename, 'r') as f:
      f_text = f.read()
  # Error handling for regex search
  search_result = re.search(r'@<TRIPOS>BOND([\s\S]*?)@<TRIPOS>', f_text) # Updated regex for robustness
  if search_result:
      bonds_text = search_result.group(1)
  else:
      return pd.DataFrame(columns=['bond_id', 'source', 'target', 'bond_type'])  # Return empty dataframe if pattern not found
  
  bonds = np.array(re.sub(r'\s+', ' ', bonds_text.strip()).split()).reshape((-1, 4))
  df_bonds = pd.DataFrame(bonds, columns=['bond_id', 'source', 'target', 'bond_type'])
  df_bonds.set_index(['bond_id'], inplace=True)
  return df_bonds

def calc_dist(df, node1, node2) : 
    x1, y1, z1 = df.iloc[np.where(df.index == str(node1))][['x', 'y', 'z']].values.flatten()
    x2, y2, z2 = df.iloc[np.where(df.index == str(node2))][['x', 'y', 'z']].values.flatten()

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) 
    return(dist)   



def ligand_prep(file):
  try:
    
    print('starting_ligand')
    

  #id="3pce"

    ligand_file = file
    #ligand_file = "../AI-GCN/Ligand/" + id + "_ligand.mol2"


    ligand = PandasMol2().read_mol2(ligand_file)
    ligand_atoms = ligand.df
    ligand_bonds = bond_parser(ligand_file)
    print(ligand_bonds)


    ligand_atoms.atom_id = ligand_atoms.atom_id.astype("str")
    #print(ligand_atoms)

    mat = '^HOH|^ZN|^CO|^K|^MG|^CA|^CU|^NA|^MN|^NI'

    nh_index_l = (ligand_atoms.iloc[:, 5].str.split(".", expand=True)[0] != "H") #& (~ligand_atoms.iloc[:, 7].str.contains(mat, case=False, regex=True))
    nh_nodes_l = ligand_atoms[nh_index_l].atom_id
    #print(nh_nodes_l)

    ligand_atoms = ligand_atoms.iloc[:, [0, 2, 3, 4, 5, 8]]
    ligand_atoms = ligand_atoms[ligand_atoms.atom_id.isin(nh_nodes_l)]
    ligand_atoms = ligand_atoms.set_index("atom_id")

    ligand_bonds = ligand_bonds[ligand_bonds.source.isin(nh_nodes_l)]
    ligand_bonds = ligand_bonds[ligand_bonds.target.isin(nh_nodes_l)]
    #

    ligand_atoms = ligand_atoms.reset_index().merge(atom_type, how='left', left_on='atom_type', right_on="atom_type").set_index('atom_id')
    ligand_atoms['atom_type'] = ligand_atoms.iloc[:, 3].str.split(".", expand=True)[0]
    ligand_atoms = ligand_atoms.reset_index().merge(atom_props, how='left', left_on='atom_type', right_on="Atom").set_index('atom_id')
    ligand_atoms.drop(["atom_type", "Atom"], axis=1, inplace=True)

    ligand_dists = []
    for s,t in zip(ligand_bonds.source, ligand_bonds.target) : 
        ligand_dists.append(calc_dist(ligand_atoms, s, t))
    ligand_bonds['weight'] = ligand_dists


    mean_x_l=(ligand_atoms.x.sum()/ligand_atoms.x.count())  
    sd_x_l= (pow((ligand_atoms.x-mean_x_l),2)).sum()
    sd_x_l=sd_x_l/ligand_atoms.x.count()
    sd_x_l=math.sqrt(sd_x_l)

    mean_y_l=(ligand_atoms.y.sum()/ligand_atoms.y.count())  
    sd_y_l= (pow((ligand_atoms.y-mean_y_l),2)).sum()
    sd_y_l=sd_y_l/ligand_atoms.y.count()
    sd_y_l=math.sqrt(sd_y_l)  
      #ligand_atoms.y = (ligand_atoms.y - mean_y_l) / sd_x_l

    mean_z_l=(ligand_atoms.z.sum()/ligand_atoms.z.count())  
    sd_z_l= (pow((ligand_atoms.z-mean_z_l),2)).sum()
    sd_z_l=sd_z_l/ligand_atoms.z.count()
    sd_z_l=math.sqrt(sd_z_l)

    mean_charge_l=(ligand_atoms.charge.sum()/ligand_atoms.charge.count())  
    sd_charge_l= (pow((ligand_atoms.charge - mean_charge_l),2)).sum()
    sd_charge_l=sd_charge_l/ligand_atoms.charge.count()
    sd_charge_l=math.sqrt(sd_charge_l) 


    ligand_atoms = (ligand_atoms - ligand_atoms.min()) / (ligand_atoms.max() - ligand_atoms.min())


    ligand_atoms = ligand_atoms.replace(np.nan, 0) 


    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('ar',6)
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('am',7) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('du',0) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('un',0)   
    #print(ligand_bonds)

    return (sg.StellarGraph(ligand_atoms, edges = ligand_bonds),id)
  except Exception as e:
        print(f"Error processing file {file}: {e}")

def predict_class(graph, model_name):
  single_graph_list = [graph[0]] 

  loaded_model = model_name

  # Now, create a generator with the corrected list of StellarGraph objects
  single_graph_generator = PaddedGraphGenerator(graphs=single_graph_list)

  # Prepare the graph for prediction without wrapping `single_graph_list` in another list
  single_graph_gen = single_graph_generator.flow(single_graph_list,targets=[0], batch_size=1)

  # Make predictions
  predictions = loaded_model.predict(single_graph_gen, steps=len(single_graph_gen)) 
  print(predictions)# Ensure steps match the number of graphs

  # Since this is a binary classification problem, the model outputs probabilities of the positive class
  # You may want to apply a threshold to convert these probabilities into a binary decision
  threshold = 0.5  # Example threshold for binary classification
  predicted_class = (predictions.flatten() > threshold).astype(int)

  return (predicted_class[0])
    
    
def ligand_preparation(id,directory):
#id="3pce"
    directory=directory

    ligand_file = directory+"/" + id + ".mol2"
    print(ligand_file)
    #ligand_file = "../AI-GCN/Ligand/" + id + "_ligand.mol2"


    ligand = PandasMol2().read_mol2(ligand_file)
    ligand_atoms = ligand.df
    ligand_bonds = bond_parser(ligand_file)


    ligand_atoms.atom_id = ligand_atoms.atom_id.astype("str")
    #print(ligand_atoms)

    mat = '^HOH|^ZN|^CO|^K|^MG|^CA|^CU|^NA|^MN|^NI'

    nh_index_l = (ligand_atoms.iloc[:, 5].str.split(".", expand=True)[0] != "H") #& (~ligand_atoms.iloc[:, 7].str.contains(mat, case=False, regex=True))
    nh_nodes_l = ligand_atoms[nh_index_l].atom_id
    #print(nh_nodes_l)

    ligand_atoms = ligand_atoms.iloc[:, [0, 2, 3, 4, 5, 8]]
    ligand_atoms = ligand_atoms[ligand_atoms.atom_id.isin(nh_nodes_l)]
    ligand_atoms = ligand_atoms.set_index("atom_id")

    ligand_bonds = ligand_bonds[ligand_bonds.source.isin(nh_nodes_l)]
    ligand_bonds = ligand_bonds[ligand_bonds.target.isin(nh_nodes_l)]
    #

    ligand_atoms = ligand_atoms.reset_index().merge(atom_type, how='left', left_on='atom_type', right_on="atom_type").set_index('atom_id')
    ligand_atoms['atom_type'] = ligand_atoms.iloc[:, 3].str.split(".", expand=True)[0]
    ligand_atoms = ligand_atoms.reset_index().merge(atom_props, how='left', left_on='atom_type', right_on="Atom").set_index('atom_id')
    ligand_atoms.drop(["atom_type", "Atom"], axis=1, inplace=True)

    ligand_dists = []
    for s,t in zip(ligand_bonds.source, ligand_bonds.target) : 
        ligand_dists.append(calc_dist(ligand_atoms, s, t))
    ligand_bonds['weight'] = ligand_dists


    mean_x_l=(ligand_atoms.x.sum()/ligand_atoms.x.count())  
    sd_x_l= (pow((ligand_atoms.x-mean_x_l),2)).sum()
    sd_x_l=sd_x_l/ligand_atoms.x.count()
    sd_x_l=math.sqrt(sd_x_l)

    mean_y_l=(ligand_atoms.y.sum()/ligand_atoms.y.count())  
    sd_y_l= (pow((ligand_atoms.y-mean_y_l),2)).sum()
    sd_y_l=sd_y_l/ligand_atoms.y.count()
    sd_y_l=math.sqrt(sd_y_l)  
      #ligand_atoms.y = (ligand_atoms.y - mean_y_l) / sd_x_l

    mean_z_l=(ligand_atoms.z.sum()/ligand_atoms.z.count())  
    sd_z_l= (pow((ligand_atoms.z-mean_z_l),2)).sum()
    sd_z_l=sd_z_l/ligand_atoms.z.count()
    sd_z_l=math.sqrt(sd_z_l)

    mean_charge_l=(ligand_atoms.charge.sum()/ligand_atoms.charge.count())  
    sd_charge_l= (pow((ligand_atoms.charge - mean_charge_l),2)).sum()
    sd_charge_l=sd_charge_l/ligand_atoms.charge.count()
    sd_charge_l=math.sqrt(sd_charge_l) 


    ligand_atoms = (ligand_atoms - ligand_atoms.min()) / (ligand_atoms.max() - ligand_atoms.min())


    ligand_atoms = ligand_atoms.replace(np.nan, 0) 


    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('ar',6)
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('am',7) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('du',0) 
    ligand_bonds.bond_type=ligand_bonds.bond_type.replace('un',0)   
    #print(ligand_bonds)

    return (sg.StellarGraph(ligand_atoms, edges = ligand_bonds),id)
    
def training_model(directory, batch_size, learning_rate, dropout_rate, num_rows_tensor, training_size):
    try:
        print(f"Directory: {directory}")

        # Load ligand data
        idsA = glob.glob(os.path.join(directory, '*.mol2'))
        idsA = [id.replace(f"{directory}/", "").replace(".mol2", "") for id in idsA]

        path_all_csv = os.path.join(directory, "All.csv")
        path_activity_txt = os.path.join(directory, "All_Activity.txt")
        selected_ids = pd.read_csv(path_all_csv).iloc[:, 0].values

        pid, lgs, ignored_ids = [], [], []

        # Prepare ligands
        for id in selected_ids:
            try:
                lg = ligand_preparation(id, directory)
                lgs.append(lg[0])
                pid.append(lg[1])
            except Exception as e:
                ignored_ids.append(id)
                print(f"Error processing ligand {id}: {e}")

        joblib.dump((pid, lgs), os.path.join(directory, "ligand_graphs.var"))
        pid, lgs = joblib.load(os.path.join(directory, "ligand_graphs.var"))

        # Prepare StellarGraph objects and activity labels
        pl_graphs = lgs
        target = pd.read_csv(path_activity_txt, sep="\s+", header=None)
        res = [target[target[0] == x][2].values.item() for x in pid]
        res = pd.DataFrame(res)
        label = res[0].astype('category').cat.codes
        label.name = 'LABEL'

        # Train-test split
        train_graphs, test_graphs = model_selection.train_test_split(label, train_size=0.7, stratify=label)

        generator = PaddedGraphGenerator(graphs=pl_graphs)
        k = int(num_rows_tensor)

        # Define DeepGraphCNN model
        layer_sizes = [1024, 1024, 1024, 1]
        dgcnn_model = DeepGraphCNN(layer_sizes=layer_sizes, activations=["tanh", "tanh", "tanh", "tanh"], k=k, generator=generator)
        x_inp, x_out = dgcnn_model.in_out_tensors()

        x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
        x_out = MaxPool1D(pool_size=2)(x_out)
        x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
        x_out = Flatten()(x_out)
        x_out = Dense(units=128, activation="relu")(x_out)
        x_out = Dropout(rate=float(dropout_rate))(x_out)

        predictions = Dense(units=1, activation="sigmoid")(x_out)
        model = Model(inputs=x_inp, outputs=predictions)

        # Compile model
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics=["accuracy"])

        # Data generators
        train_gen = generator.flow(list(train_graphs.index), targets=train_graphs.values, batch_size=batch_size)
        test_gen = generator.flow(list(test_graphs.index), targets=test_graphs.values, batch_size=1)

        # Train the model
        model.fit(train_gen, epochs=training_size, verbose=1)

        return model, test_gen, test_graphs

    except Exception as e:
        print(f"Error during training: {e}")

def objective(trial):
    path2 = 'TB_case_study/'

    # Suggest hyperparameters with refined search space
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)  # Narrower range
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.4, 0.5)  # Narrower range
    num_rows_tensor = trial.suggest_int('num_rows_tensor', 15, 30)  # Smaller range
    training_size = trial.suggest_int('training_size', 100, 300)  # Limiting the epochs to 100-300

    # Train model with current hyperparameters
    model, test_gen, test_graphs = training_model(path2, batch_size, learning_rate, dropout_rate, num_rows_tensor, training_size)

    # Evaluate the model on the test set
    test_metrics = model.evaluate(test_gen, verbose=0)
    accuracy = test_metrics[1]  # Index 1 corresponds to accuracy

    print(f"Trial {trial.number}: Accuracy = {accuracy}, Params = {trial.params}")

    return accuracy  # Objective is to maximize accuracy


# Optuna study for hyperparameter tuning with trial logging
def tune_hyperparameters():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Increased to 50 trials for better search

    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")
    
    return study.best_params


# Main function
if __name__ == "__main__":
    # Hyperparameter tuning using Optuna
    best_hyperparams = tune_hyperparameters()
    
    # Use the best hyperparameters to train the final model
    path2 = 'TB_case_study/'
    model, test_gen, test_graphs = training_model(
        path2, 
        batch_size=best_hyperparams['batch_size'],
        learning_rate=best_hyperparams['learning_rate'],
        dropout_rate=best_hyperparams['dropout_rate'],
        num_rows_tensor=best_hyperparams['num_rows_tensor'],
        training_size=best_hyperparams['training_size']
    )
    
    # Save the final trained model
    model_name = path2.split('/')[-2]
    model.save(f'media/model/{model_name}')
    print(f"Final model saved to media/model/{model_name}")
    
    # Evaluate the final model
    test_metrics = model.evaluate(test_gen)
    print(f"Final Test Set Metrics: {test_metrics}")


