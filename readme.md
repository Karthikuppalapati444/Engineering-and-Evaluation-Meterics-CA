# Multistage Classification Pipeline for Customer Support Tickets

This project implements a three-stage classification pipeline using Random Forest models to analyze customer support ticket data. The goal is to predict:

- **y2**: Intent  
- **y3**: Tone  
- **y4**: Resolution  

based on combined textual information from `Ticket Summary` and `Interaction content`.

---

## Data Preparation

1. Place the following CSV files in the root directory:
   - `AppGallery.csv`
   - `Purchasing.csv`

2. Ensure that each CSV file contains the following columns:
   - `Ticket Summary`
   - `Interaction content`
   - `y2`, `y3`, and `y4`

---

##  Preprocessing & Embedding

1. **Concatenate** `Ticket Summary` and `Interaction content` for each record into a single string.

2. **Generate TF-IDF embeddings** from the concatenated text using:
   - `max_features=2000`
   - `min_df=4`
   - `max_df=0.90`

3. **Store** the resulting matrix for model input.

---

##  Data Initialization

1. Use the `Data` class to:
   - Filter labels (y2, y3, y4) to keep only those with **at least 3 samples**
   - Perform **stratified train-test splitting** for balanced class representation

---

##  Modeling Pipeline

###  Stage 1: Predict `y2` (Intent)
- Train a Random Forest classifier on the original embedding matrix.
- Generate predictions and store as `pred_y2`.
- Print a classification report.

###  Stage 2: Predict `y3` (Tone)
- Encode `pred_y2` using `LabelEncoder` and concatenate to embeddings.
- Train a new model on `[X + pred_y2]`.
- Store predictions as `pred_y3`.
- Print classification report.

###  Stage 3: Predict `y4` (Resolution)
- Encode `pred_y3` and concatenate to the matrix.
- Train the final model on `[X + pred_y2 + pred_y3]`.
- Save predictions as `pred_y4`.
- Print classification results.

---

## Output

- Final predictions will be saved in:
  - `predictions_AppGallery.csv`
  - `predictions_Purchasing.csv`

Each file will contain original data and predictions for `y2`, `y3`, and `y4`.

---

##  Configuration

- Modify `Config.py` to change:
  - Column names (e.g., `TICKET_SUMMARY`, `INTERACTION_CONTENT`)
  - Classification targets (e.g., `CLASS_COL`)

---

## Dependencies

Ensure the following Python libraries are installed:

```bash
pip install pandas numpy scikit-learn
