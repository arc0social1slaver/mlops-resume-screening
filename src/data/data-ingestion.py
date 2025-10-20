import os
import pandas as pd
from src.util import load_data, save_data


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Get the largest category size (i.e., the category with the maximum number of entries)
        max_size = df["Category"].value_counts().max()
        # Perform oversampling
        balanced_df = (
            df.groupby("Category")
            .apply(lambda x: x.sample(max_size, replace=True))
            .reset_index(drop=True)
        )

        # Shuffle the dataset to avoid any order bias
        return balanced_df.sample(frac=1).reset_index(drop=True)
    except Exception as e:
        print(e)
        raise


if __name__ == "__main__":
    df = load_data(
        data_url=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../data/UpdatedResumeDataSet.csv",
        )
    )

    finalDf = balance_classes(df)

    save_data(
        finalDf,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw"),
        "resume.csv",
    )
