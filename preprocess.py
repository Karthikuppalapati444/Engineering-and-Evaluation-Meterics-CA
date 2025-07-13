import pandas as pd
import re
from Config import *


def get_input_data() -> pd.DataFrame:
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df = pd.concat([df1, df2])

    # Dropping rows with missing values in any of the output labels
    df = df.dropna(subset=['y2', 'y3', 'y4']).reset_index(drop=True)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    df["y"] = df[Config.CLASS_COL]
    return df


def de_duplication(df: pd.DataFrame) -> pd.DataFrame:
    # Simplified stub for testing
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.replace(r'\s+', ' ', regex=True)
    return df


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal noise remover
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].str.lower().replace(r'\s+', ' ', regex=True).str.strip()
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower().replace(r'\s+', ' ', regex=True).str.strip()
    return df
