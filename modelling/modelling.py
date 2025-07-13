from model.randomforest import RandomForest
from sklearn.preprocessing import LabelEncoder
from Config import Config
import numpy as np

def model_predict(data, df, name):
    # Printing header for current y1 group (AppGallery & Games)
    print(f"\n=== {name.upper()} ===")

    # Dropping any records missing labels y2, y3, or y4
    df = df.dropna(subset=['y2', 'y3', 'y4']).reset_index(drop=True)

    # Getting raw TF-IDF embeddings from the Data object
    X_raw = data.get_embeddings()

    # Stage 1: Predict y2 (Intent)
    Config.CLASS_COL = 'y2'  # Set current classification target
    y2 = df['y2']            # Ground-truth for stage 1
    model_y2 = RandomForest("RandomForest_y2", X_raw, y2)  # Initialize model
    print(f"\n  {model_y2.model_name}")                    # Print model name
    model_y2.train(X_raw, y2)                              # Train on embeddings
    y2_pred = model_y2.predict(X_raw)                      # Predict y2
    df['pred_y2'] = y2_pred                                # Save predictions
    print("\nStage 1 - y2 (Intent) Results:")              # Display results
    model_y2.print_results(y2)

    # Stage 2: Predict y3 (Tone)
    Config.CLASS_COL = 'y3'
    le_y2 = LabelEncoder()                                 # Encode predicted y2
    y2_encoded = le_y2.fit_transform(df['pred_y2']).reshape(-1, 1)
    X_with_y2 = np.concatenate((X_raw, y2_encoded), axis=1)  # Add y2 as feature
    y3 = df['y3']
    model_y3 = RandomForest("RandomForest_y3", X_with_y2, y3)
    print(f"\n  {model_y3.model_name}")
    model_y3.train(X_with_y2, y3)
    y3_pred = model_y3.predict(X_with_y2)
    df['pred_y3'] = y3_pred
    print("\nStage 2 - y3 (Tone) Results:")
    model_y3.print_results(y3)

    # Stage 3: Predict y4 (Resolution)
    Config.CLASS_COL = 'y4'
    le_y3 = LabelEncoder()                                 # Encode predicted y3
    y3_encoded = le_y3.fit_transform(df['pred_y3']).reshape(-1, 1)
    X_with_y2_y3 = np.concatenate((X_with_y2, y3_encoded), axis=1)  # Add y3
    y4 = df['y4']
    model_y4 = RandomForest("RandomForest_y4", X_with_y2_y3, y4)
    print(f"\n  {model_y4.model_name}")
    model_y4.train(X_with_y2_y3, y4)
    y4_pred = model_y4.predict(X_with_y2_y3)
    df['pred_y4'] = y4_pred
    print("\nStage 3 - y4 (Resolution) Results:")
    model_y4.print_results(y4)

    # Saving final predictions to CSV
    df.to_csv(f"predictions_{name}.csv", index=False)
    print(f"\nSaved: predictions_{name}.csv")
