import pandas as pd
import random
from sampling import SamplingConfig

def process_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "text"})

    for col in ["type", "formality_label", "formality_explanation"]:
        if col not in df.columns:
            df[col] = ""

    df = df[["text", "type", "formality_label", "formality_explanation"]]
    return df


def merge_and_shuffle_datasets(
    config: SamplingConfig, input_files: list, output_csv: str
) -> None:
    dataframes = []

    for file in input_files:
        df = process_dataset(file)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    merged_df = merged_df[merged_df["formality_label"] != "error"]

    formal_mask = merged_df["formality_label"] == "formal"
    formal_df = merged_df[formal_mask]
    others_df = merged_df[~formal_mask]

    # The dataset would be unbalanced with too many formal samples, here a part of them is removed
    if len(formal_df) > 500:
        formal_to_remove = formal_df.sample(n=500, random_state=42).index
        formal_df = formal_df.drop(formal_to_remove)
    merged_df = pd.concat([formal_df, others_df], ignore_index=True)

    merged_df = merged_df.sample(frac=1, random_state=config.seed).reset_index(
        drop=True
    )

    merged_df.to_csv(output_csv, index=False, header=True, quoting=csv.QUOTE_ALL)
    print(f"Merged dataset saved as '{output_csv}'")


def main(config: SamplingConfig = SamplingConfig()):
    random.seed(config.seed)

    input_csv_files = [
        "govreport_formal_resampled_labeled.csv",
        "pubmed_formal_resampled_labeled.csv",
        "reddit_comments_sampled_labeled.csv",
        "neutral_sampled.csv",
    ]

    output_csv_file = "formality_dataset.csv"

    merge_and_shuffle_datasets(config, input_csv_files, output_csv_file)


if __name__ == "__main__":
    main()
