
from data.paths import CSV_DIR

import pandas as pd
from sklearn.model_selection import train_test_split


# Split the dataset into training and validation sets.
# Each country is split separately to preserve its representation in both sets.
def split_dataset():

    df = pd.read_csv(CSV_DIR / "training_full.csv")

    # Stop if the dataset is empty, since there is nothing to split.
    if len(df) == 0:
        raise RuntimeError("training_full.csv contains no images")

    train_parts = []
    val_parts = []

    #go through each country
    # Grouping by country allows us to split each country's images separately.
    # This prevents countries with few images from being completely
    # missing from the training set.
    for country, group in df.groupby("country"):

        #if a country has only one image, keep it in training
        # A single image cannot be divided between training and validation.
        # Keeping it in training ensures the model can learn that country.
        if len(group) == 1:
            train_parts.append(group)
            
            continue

        # Use 80% of each country's images for training and 20% for validation.
        # random_state=42 makes the split reproducible across runs.
        train_group, val_group = train_test_split( group,test_size=0.2, random_state=42)

        train_parts.append(train_group)
        val_parts.append(val_group)

    train_df = pd.concat(train_parts)

    # Some datasets may contain only countries with one image.
    # In that case, create an empty validation DataFrame with the same columns.
    if len(val_parts) > 0:
        val_df = pd.concat(val_parts)
    else:
        val_df = df.iloc[0:0].copy()

    # Shuffle the combined datasets so images from the same country
    # are not grouped together in the saved CSV files.
    train_df = train_df.sample(frac=1, random_state=42)

    val_df = val_df.sample(frac=1, random_state=42)

    train_df.to_csv(CSV_DIR / "training_split.csv", index=False)
    val_df.to_csv(CSV_DIR / "validation_split.csv", index=False)

    print("Training images:", len(train_df), flush=True)
    print("Validation images:", len(val_df), flush=True)
    print("Number of classes:", df["label"].nunique(), flush=True)


if __name__ == "__main__":
    split_dataset()
