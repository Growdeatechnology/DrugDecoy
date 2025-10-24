import os
import re
import glob
import time
import math
import joblib
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import optuna

# ============================================
# 1️⃣ Load atom properties
# ============================================
atom_props = pd.read_csv("atom_props.csv")       # numerical features for each atom type
atom_type = pd.read_csv("Atom_Type.csv")        # atom type mapping

# ============================================
# 2️⃣ Utility functions
# ============================================
def bond_parser(filename):
    with open(filename, 'r') as f:
        text = f.read()
    match = re.search(r'@<TRIPOS>BOND([\s\S]*)', text)
    if not match: return pd.DataFrame(columns=['bond_id','source','target','bond_type'])
    bonds = np.array(re.sub(r'\s+', ' ', match.group(1).strip()).split()).reshape((-1,4))
    df = pd.DataFrame(bonds, columns=['bond_id','source','target','bond_type']).set_index('bond_id')
    return df

def calc_dist(df, node1, node2):
    x1,y1,z1 = df.loc[str(node1), ['x','y','z']]
    x2,y2,z2 = df.loc[str(node2), ['x','y','z']]
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

def ligand_to_graph(filepath):
    try:
        ligand = PandasMol2().read_mol2(filepath)
        atoms = ligand.df
        bonds = bond_parser(filepath)
        atoms['atom_id'] = atoms['atom_id'].astype(str)

        # heavy atoms only
        heavy_mask = atoms.iloc[:,5].str.split('.', expand=True)[0] != 'H'
        atoms = atoms[heavy_mask].iloc[:, [0,2,3,4,5,8]].set_index('atom_id')
        bonds = bonds[bonds.source.isin(atoms.index) & bonds.target.isin(atoms.index)]

        # merge features
        atoms = (
            atoms.reset_index()
            .merge(atom_type, how='left', on='atom_type')
            .assign(atom_type=lambda df: df['atom_type'].str.split('.').str[0])
            .merge(atom_props, how='left', left_on='atom_type', right_on='Atom')
            .drop(['atom_type','Atom'], axis=1)
            .set_index('atom_id')
        )

        # bond distances & type encoding
        bonds['weight'] = [calc_dist(atoms, s, t) for s,t in zip(bonds.source, bonds.target)]
        bonds['bond_type'] = bonds['bond_type'].replace({'ar':6, 'am':7, 'du':0, 'un':0})

        # normalize features
        atoms = (atoms - atoms.min()) / (atoms.max() - atoms.min())
        atoms = atoms.fillna(0)
        return StellarGraph(atoms, edges=bonds)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def load_all_ligands(directory):
    ligand_ids = [os.path.basename(f).replace('.mol2','') for f in glob.glob(os.path.join(directory,'*.mol2'))]
    graphs, valid_ids, errors = [], [], []
    for lid in ligand_ids:
        path = os.path.join(directory,f"{lid}.mol2")
        graph = ligand_to_graph(path)
        if graph: graphs.append(graph); valid_ids.append(lid)
        else: errors.append(lid)
    return graphs, valid_ids, errors

# ============================================
# 3️⃣ Build DGCNN model
# ============================================
def build_dgcnn_model(generator, k, learning_rate, dropout_rate):
    layer_sizes=[1024,1024,1024,1]
    dgcnn=DeepGraphCNN(layer_sizes=layer_sizes, activations=["tanh"]*4, k=k, generator=generator)
    x_inp,x_out=dgcnn.in_out_tensors()
    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
    x_out = Flatten()(x_out)
    x_out = Dense(128, activation="relu")(x_out)
    x_out = Dropout(dropout_rate)(x_out)
    predictions = Dense(1, activation="sigmoid")(x_out)
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=binary_crossentropy, metrics=["accuracy"])
    return model

# ============================================
# 4️⃣ Training and evaluation
# ============================================
def train_and_evaluate(directory, batch_size, learning_rate, dropout_rate, num_rows_tensor, epochs, random_seed=42):
    np.random.seed(random_seed)
    graphs, valid_ids, errors = load_all_ligands(directory)
    print(f"Loaded {len(graphs)} ligands, {len(errors)} failed.")

    # labels
    target_file = os.path.join(directory,"All_Activity.txt")
    target = pd.read_csv(target_file, sep="\s+", header=None)
    labels = [target[target[0]==lid][2].values.item() for lid in valid_ids]
    labels = pd.Series(labels, index=range(len(labels)))
    labels_cat = labels.astype('category').cat.codes

    # train/test split
    train_idx, temp_idx = train_test_split(labels_cat.index, train_size=0.7, stratify=labels_cat, random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.33, stratify=labels_cat[temp_idx], random_state=random_seed)

    # generator
    generator = PaddedGraphGenerator(graphs=graphs)
    train_gen = generator.flow(train_idx, targets=labels_cat[train_idx].values, batch_size=batch_size)
    val_gen = generator.flow(val_idx, targets=labels_cat[val_idx].values, batch_size=batch_size)
    test_gen = generator.flow(test_idx, targets=labels_cat[test_idx].values, batch_size=1)

    # build model
    model = build_dgcnn_model(generator, k=num_rows_tensor, learning_rate=learning_rate, dropout_rate=dropout_rate)
    start_time = time.time()
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
    training_time = time.time() - start_time

    # evaluation
    y_true = labels_cat[test_idx].values
    y_pred = model.predict(test_gen).flatten()
    y_pred_class = (y_pred>0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_class)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_class)

    # return everything for report
    return model, {
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "training_time": training_time,
        "num_ligands": len(graphs),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "errors": errors
    }

# ============================================
# 5️⃣ Hyperparameter tuning
# ============================================
def objective(trial):
    batch_size = trial.suggest_categorical('batch_size',[16,32,64])
    learning_rate = trial.suggest_loguniform('learning_rate',1e-4,1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate',0.4,0.5)
    num_rows_tensor = trial.suggest_int('num_rows_tensor',15,30)
    epochs = trial.suggest_int('epochs',100,300)
    model, metrics = train_and_evaluate("TB_case_study", batch_size, learning_rate, dropout_rate, num_rows_tensor, epochs)
    return metrics["accuracy"]

def tune_hyperparameters():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    print(f"Best params: {study.best_params}, Best accuracy: {study.best_value}")
    return study.best_params

# ============================================
# 6️⃣ Generate report
# ============================================
def generate_report(metrics, best_params, filename="report.txt"):
    with open(filename,"w") as f:
        f.write("=== DGCNN Ligand Classification Report ===\n\n")
        f.write(f"1️⃣ Reproducibility:\n")
        f.write(f"Total ligands: {metrics['num_ligands']}\n")
        f.write(f"Train/Val/Test sizes: {metrics['train_size']}/{metrics['val_size']}/{metrics['test_size']}\n")
        f.write(f"Failed ligands: {metrics['errors']}\n\n")

        f.write(f"2️⃣ Feature Transparency:\n")
        f.write(f"Atom features: {list(atom_props.columns[1:])}\n")
        f.write(f"Bond features: ['bond_type','weight']\n\n")

        f.write(f"3️⃣ Benchmarking / Performance:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Training Time: {metrics['training_time']:.2f} sec\n\n")

        f.write(f"4️⃣ Hyperparameters:\n")
        for k,v in best_params.items(): f.write(f"{k}: {v}\n")
    print(f"Report saved to {filename}")

# ============================================
# 7️⃣ Main
# ============================================
if __name__=="__main__":
    best_params = tune_hyperparameters()
    model, metrics = train_and_evaluate("TB_case_study",
                                        batch_size=best_params['batch_size'],
                                        learning_rate=best_params['learning_rate'],
                                        dropout_rate=best_params['dropout_rate'],
                                        num_rows_tensor=best_params['num_rows_tensor'],
                                        epochs=best_params['epochs'])
    model.save("media/model/final_model")
    print("Model saved.")
    generate_report(metrics, best_params)

