#  Imports

# Importing custom preprocessing module (deduplication, cleaning)
import preprocess

# Importing TF-IDF embedding generation function
from embeddings import get_tfidf_embd

# Importing model runner (chained multi-output logic)
from modelling.modelling import model_predict

# Importing data wrapper class
from modelling.data_model import Data

# Configuring contains constants like column names (e.g., y2, y3, etc.)
from Config import Config

# Standardising libraries
import random
import numpy as np


# Setting seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)


#  Loading merged and cleaned input CSVs
def load_data():
    return preprocess.get_input_data()


# Applying all preprocessing: deduplication + noise cleaning
def preprocess_data(df):
    df = preprocess.de_duplication(df)
    df = preprocess.noise_remover(df)
    return df


# Generating embeddings
def get_embeddings(df):
    X = get_tfidf_embd(df)
    return X, df


# Wrapping the features and dataframe into a Data class
def get_data_object(X, df):
    return Data(X, df)


# Running the chained multi-output classifier for one group
def perform_modelling(data, df, name):
    model_predict(data, df, name)
