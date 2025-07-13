# Main program starts here
if __name__ == '__main__':
    # Loading raw data from both CSVs
    df = load_data()

    # Preprocessing data (clean & deduplicate)
    df = preprocess_data(df)

    # Ensuring text fields are in string format (defensive coding)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    # Groupping data by y1 (AppGallery, In-App Purchase)
    grouped_df = df.groupby(Config.GROUPED)

    # For each group (one product area), train & evaluate
    for name, group_df in grouped_df:
        print(f"\n{name}")  # Print group name
        X, group_df = get_embeddings(group_df)              # TF-IDF embeddings
        data = get_data_object(X, group_df)                 # Wrap into Data object
        perform_modelling(data, group_df, name)             # Train + predict + save results
