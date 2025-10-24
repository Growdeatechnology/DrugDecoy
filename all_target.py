import os
import re
import glob
import math
import joblib
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

# ============================================
# 1️⃣ Load atom features
# ============================================
atom_props = pd.read_csv("atom_props.csv")
atom_type = pd.read_csv("Atom_Type.csv")

# ============================================
# 2️⃣ Utility functions
# ============================================
def bond_parser(filename):
    with open(filename, 'r') as f:
        text = f.read()
    match = re.search(r'@<TRIPOS>BOND([\s\S]*)', text)
    if not match:
        return pd.DataFrame(columns=['bond_id', 'source', 'target', 'bond_type'])
    bonds = np.array(re.sub(r'\s+', ' ', match.group(1).strip()).split()).reshape((-1, 4))
    df = pd.DataFrame(bonds, columns=['bond_id', 'source', 'target', 'bond_type']).set_index('bond_id')
    return df

def calc_dist(df, node1, node2):
    x1, y1, z1 = df.loc[str(node1), ['x', 'y', 'z']]
    x2, y2, z2 = df.loc[str(node2), ['x', 'y', 'z']]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def create_activity_file_if_missing(target_dir):
    """
    Create All_Activity.txt in the target_dir if not found.
    It will scan 'Uniques' folder and assign Inactive (0) or Active (1)
    based on ligand name containing 'active' or 'decoy'.
    """
    activity_path = os.path.join(target_dir, "All_Activity.txt")
    ligand_path = os.path.join(target_dir, "Uniques")

    # Skip if already exists
    if os.path.exists(activity_path):
        return

    if not os.path.exists(ligand_path):
        print(f"⚠️ No 'Uniques' folder found in {target_dir}, skipping activity file creation.")
        return

    # Collect all mol2 ligands
    ligand_files = glob.glob(os.path.join(ligand_path, "*.mol2"))
    if not ligand_files:
        print(f"⚠️ No ligands found in {ligand_path}, skipping activity file creation.")
        return

    # Build entries
    entries = []
    for f in ligand_files:
        name = os.path.splitext(os.path.basename(f))[0]
        if re.search(r"active", name, re.IGNORECASE):
            entries.append(f"{name}\tActive\t1")
        else:
            entries.append(f"{name}\tInactive\t0")

    # Write to All_Activity.txt
    with open(activity_path, "w") as f:
        f.write("\n".join(entries))

    print(f"📝 Created missing All_Activity.txt in {target_dir} ({len(entries)} ligands)")


def ligand_to_graph(filepath):
    try:
        ligand = PandasMol2().read_mol2(filepath)
        atoms = ligand.df
        bonds = bond_parser(filepath)
        atoms['atom_id'] = atoms['atom_id'].astype(str)

        # remove hydrogens
        heavy_mask = atoms.iloc[:, 5].str.split('.', expand=True)[0] != 'H'
        atoms = atoms[heavy_mask].iloc[:, [0, 2, 3, 4, 5, 8]].set_index('atom_id')
        bonds = bonds[bonds.source.isin(atoms.index) & bonds.target.isin(atoms.index)]

        atoms = (
            atoms.reset_index()
            .merge(atom_type, how='left', on='atom_type')
            .assign(atom_type=lambda df: df['atom_type'].str.split('.').str[0])
            .merge(atom_props, how='left', left_on='atom_type', right_on='Atom')
            .drop(['atom_type', 'Atom'], axis=1)
            .set_index('atom_id')
        )

        bonds['weight'] = [calc_dist(atoms, s, t) for s, t in zip(bonds.source, bonds.target)]
        bonds['bond_type'] = bonds['bond_type'].replace({'ar': 6, 'am': 7, 'du': 0, 'un': 0})

        atoms = (atoms - atoms.min()) / (atoms.max() - atoms.min())
        atoms = atoms.fillna(0)
        return StellarGraph(atoms, edges=bonds)
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
        return None

def load_all_ligands(directory):
    # 👇 Look inside Uniques folder for .mol2 files
    ligand_path = os.path.join(directory, "Uniques")
    ligand_files = glob.glob(os.path.join(ligand_path, "*.mol2"))
    ligand_ids = [os.path.splitext(os.path.basename(f))[0] for f in ligand_files]

    graphs, valid_ids, errors = [], [], []
    for lid in ligand_ids:
        path = os.path.join(ligand_path, f"{lid}.mol2")
        graph = ligand_to_graph(path)
        if graph:
            graphs.append(graph)
            valid_ids.append(lid)
        else:
            errors.append(lid)
    return graphs, valid_ids, errors


# ============================================
# 3️⃣ Build GCNN model
# ============================================
def build_dgcnn_model(generator, k=20, learning_rate=1e-3, dropout_rate=0.4):
    layer_sizes = [1024, 1024, 1024, 1]
    dgcnn = DeepGraphCNN(layer_sizes=layer_sizes, activations=["tanh"] * 4, k=k, generator=generator)
    x_inp, x_out = dgcnn.in_out_tensors()
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
# 4️⃣ EF 1% calculation
# ============================================
def calculate_ef1(y_true, y_pred, top_percent=0.01):
    n = len(y_true)
    top_k = max(1, int(n * top_percent))
    sorted_idx = np.argsort(y_pred)[::-1]
    top_idx = sorted_idx[:top_k]
    observed_actives = np.sum(y_true[top_idx] == 1)
    total_actives = np.sum(y_true == 1)
    expected_actives = total_actives * top_percent
    if expected_actives == 0:
        return 0
    return observed_actives / expected_actives

# ============================================
# 5️⃣ Run training and EF 1% for a target
# ============================================
def run_for_target(target_dir):
    create_activity_file_if_missing(target_dir)  # 👈 add this line here
    graphs, valid_ids, errors = load_all_ligands(target_dir)
    if len(graphs) == 0:
        print(f"❌ No ligands found for {target_dir}")
        return None

    target_file = os.path.join(target_dir, "All_Activity.txt")
    if not os.path.exists(target_file):
        print(f"❌ No activity file found in {target_dir}")
        return None

    target = pd.read_csv(target_file, sep="\s+", header=None)
    labels = []
    for lid in valid_ids:
        if lid in target[0].values:
            labels.append(target[target[0] == lid][2].values.item())
        else:
            labels.append(0)  # if not found, default inactive

    labels = np.array(labels)

    generator = PaddedGraphGenerator(graphs=graphs)
    gen = generator.flow(np.arange(len(labels)), targets=labels, batch_size=1)

    model = build_dgcnn_model(generator)
    model.fit(gen, epochs=5, verbose=0)

    y_pred = model.predict(gen).flatten()
    ef1 = calculate_ef1(labels, y_pred, top_percent=0.01)

    print(f"✅ {os.path.basename(target_dir)} → EF 1% = {ef1:.2f}")
    return ef1

# ============================================
# 6️⃣ Loop through multiple targets
# ============================================
if __name__ == "__main__":
    base_dir = "."
    # folders like ace, aldr, andr, esr1, esr2 ...
    targets = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d)) and d not in ["media", "__pycache__"]]

    ef_scores = []
    valid_targets = []

    for target_dir in targets:
        ef = run_for_target(target_dir)
        if ef is not None:
            ef_scores.append(ef)
            valid_targets.append(os.path.basename(target_dir))

    if ef_scores:
        avg_ef = np.mean(ef_scores)
        print("\n📊 EF 1% per target:")
        for t, ef in zip(valid_targets, ef_scores):
            print(f"  - {t}: {ef:.2f}")
        print(f"\n🌟 Average EF 1% across all targets = {avg_ef:.2f}")
    else:
        print("❌ No valid EF 1% scores computed.")

