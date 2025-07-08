#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
st.set_page_config(page_title="Drug Predictor", page_icon="🧪", layout="centered")
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
import numpy as np
import tensorflow as tf

st.title("🧪 AI-Powered Drug Bioactivity Predictor")

try:
    # Try loading the model
    model = tf.keras.models.load_model("C:/Users/Anshika/Documents/drug_dashboard_project/drug_model.keras")
except Exception as e:
    st.error(f"🚨 Failed to load model: {e}")
    st.stop()


# In[2]:


import pandas as pd
import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define target - EGFR (ChEMBL ID: CHEMBL203)
target_id = "CHEMBL203"
url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&limit=1000"

response = requests.get(url)
data = response.json()
activities = data['activities']

# Extract SMILES and bioactivity
smiles = []
bioactivities = []

for a in activities:
    if a['standard_type'] == 'IC50' and a['standard_value']:
        smiles.append(a['canonical_smiles'])
        bioactivities.append(float(a['standard_value']))

df = pd.DataFrame({'smiles': smiles, 'IC50': bioactivities})

# Remove invalid SMILES
def is_valid(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

df = df[df['smiles'].apply(is_valid)]

# Calculate molecular descriptors
def calc_mol_weight(s):
    mol = Chem.MolFromSmiles(s)
    return Descriptors.MolWt(mol)

df['mol_weight'] = df['smiles'].apply(calc_mol_weight)

# Label: Active (< 1000 nM) = 1, Inactive = 0
df['bioactivity_class'] = df['IC50'].apply(lambda x: 1 if x < 1000 else 0)

df.to_csv("chembl_egfr_dataset.csv", index=False)
df.head()


# In[3]:


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
df = pd.read_csv("chembl_egfr_dataset.csv")

# Convert SMILES to Morgan fingerprints
def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

X = np.array([smiles_to_fp(s) for s in df['smiles']])
y = df['bioactivity_class'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Deep Neural Network
model = models.Sequential([
    layers.Input(shape=(2048,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate
y_pred = model.predict(X_test).ravel()
y_pred_binary = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary))
print("AUC:", roc_auc_score(y_test, y_pred))


# In[4]:


#training graph


# In[5]:




import matplotlib.pyplot as plt

# Plot training & validation accuracy/loss
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[6]:


#Add a dashboard (Streamlit or plotly) for visualizing drug predictions.


# In[7]:


model.save('drug_model.keras')


# In[8]:


import os
print("drug_model.keras exists:", os.path.exists('drug_model.keras'))


# In[9]:


import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("C:/Users/Anshika/Documents/drug_dashboard_project/drug_model.keras")

# Input SMILES
smiles = st.text_input("Enter SMILES string:", "CCOC(=O)C1=CC=CC=C1")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.write("### Molecular Structure")
        st.image(Draw.MolToImage(mol), caption=smiles)

        # Calculate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        X = np.array(fp).reshape(1, -1)

        # Predict
        pred = model.predict(X)[0][0]
        result = "Active" if pred > 0.5 else "Inactive"
        st.markdown(f"### 🔍 Prediction: `{result}` (Confidence: {pred:.2f})")

        # Show descriptors
        st.write("### Molecular Descriptors")
        st.write({
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H-Bond Donors": Descriptors.NumHDonors(mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(mol)
        })
    else:
        st.error("Invalid SMILES")


# In[ ]:




