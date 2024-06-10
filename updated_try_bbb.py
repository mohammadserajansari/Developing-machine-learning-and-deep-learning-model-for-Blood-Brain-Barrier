
import pickle
import pandas as pd
import streamlit as st
from rdkit import Chem
from mordred import Calculator, descriptors
from rdkit.Chem import GetMolFrags
import os

# Load the model and selected_features from the pickle file
filename = "rf_acc_86_mc_68_model_try.pkl"

# Check if the file exists
if not os.path.exists(filename):
    st.error(f"Model file {filename} not found.")
    raise FileNotFoundError(f"Model file {filename} not found.")

try:
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    raise e

# Extract components
loaded_scaler = loaded_model['scaler']
loaded_selected_column_names = loaded_model['selected_column_names']
loaded_rf_model = loaded_model['rf_classifier']

# Function to convert SMILES to isomeric SMILES
def convert_to_isomeric_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        isomeric_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return isomeric_smiles
    else:
        return None

# Function to calculate descriptors from SMILES
def calculate_descriptors(smiles):
    isomeric_smiles = convert_to_isomeric_smiles(smiles)
    if isomeric_smiles is not None:
        data = {'SMILES': [smiles], 'Isomeric SMILES': [isomeric_smiles]}
        df = pd.DataFrame(data)

        # Initialize descriptor calculator
        calc = Calculator(descriptors, ignore_3D=False)

        # Read SMILES from the DataFrame
        molecules = df['Isomeric SMILES']

        # Convert valid SMILES to RDKit Mol objects and filter out molecules with disconnected fragments
        mols = []
        valid_molecules = []
        for smi in molecules:
            if smi is not None:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fragments = GetMolFrags(mol)
                    if len(fragments) == 1:  # Exclude molecules with multiple fragments
                        mols.append(mol)
                        valid_molecules.append(smi)

        # Generate descriptors
        descriptors_df = calc.pandas(mols)

        # Create a DataFrame with descriptors and valid SMILES
        df_des = pd.concat([descriptors_df, pd.Series(valid_molecules, name='Isomeric SMILES')], axis=1)

        for column in df_des.columns:
            if df_des[column].dtype == object:
                df_des[column] = pd.to_numeric(df_des[column], errors='coerce').fillna(0)
            elif pd.api.types.is_numeric_dtype(df_des[column]):
                df_des[column] = pd.to_numeric(df_des[column], errors='coerce').fillna(0)
            else:
                df_des[column] = 0.0

        # Drop the 'Isomeric SMILES' column
        df_des_only = df_des.drop("Isomeric SMILES", axis=1)

        return df_des_only
    else:
        return None

# Streamlit App
def main():
    st.title("Neurocare-eLAB")
    # Set the title and description of the app
    st.subheader("BBB Permeability Prediction Using AI")

    # Add a side option menu with additional information if needed
    st.sidebar.title("Menu")
    st.sidebar.write("BBB Permeability.")

    # User input for SMILES
    smiles_input = st.text_input("Enter the SMILES :")

    # Check if user provided a SMILES
    if smiles_input:
        # Calculate descriptors and make prediction
        descriptors_df = calculate_descriptors(smiles_input)
        if descriptors_df is not None:
            # Select only the features used during training from 'input_data' based on selected_features
            input_data_selected = descriptors_df[loaded_selected_column_names]
            sacaled_data = loaded_scaler.transform(input_data_selected)
            # Make predictions using the loaded model and the input data with selected features
            prediction = loaded_rf_model.predict(sacaled_data)

            # Display the prediction result
            if prediction == 0:
                st.write("Our assessment concludes that the molecule's BBB Permeability is negative")
            else:
                st.write("Our assessment concludes that the molecule's BBB Permeability is positive")
        else:
            st.write("Invalid SMILES input. Please enter a valid SMILES string.")
    else:
        st.write("Enter a SMILES string to predict its BBB Permeability.")
        
st.write("_This website belongs to NeuroeCare LAB and was developed by Mohammad Seraj (M.Tech, CB, IIIT-D) under the supervision of Dr. N. Arul Murugan._")
# Run the Streamlit app
if __name__ == "__main__":
    main()
