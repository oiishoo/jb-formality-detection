import random

import nltk

nltk.download("punkt_tab")
from nltk import sent_tokenize
from datasets import (
    load_dataset,
    Dataset,
    VerificationMode,
)
from pydantic import BaseModel
import csv
import pandas as pd


class SamplingConfig(BaseModel):
    seed: int = 64

# Cleanup of the reddit comments
def clean_text(example):
    example["body"] = (
        example["body"]
        .replace("\n", " ")
        .replace("\r", " ")
        .removeprefix('"')
        .removesuffix('"')
    )
    return example


def sample_dataset(
    config: SamplingConfig,
    dataset: Dataset,
    total_samples_desired,
    type,
    column,
    csv_name,
):

    sample = dataset.shuffle(seed=config.seed).select(range(total_samples_desired))

    sample = sample.map(lambda x: {"type": type})

    # Filtering out non-informative comments
    if type == "reddit_comment":
        sample = sample.filter(lambda x: x[column] not in ["[deleted]", "[removed]"])
        sample = sample.select(range(1000))

    sample = sample.select_columns([column, "type"])

    if type == "reddit_comment":
        sample = sample.map(clean_text)

    df = sample.to_pandas()
    df.to_csv(csv_name, index=False, quoting=csv.QUOTE_ALL)
    print(f"Sampled dataset saved as '{csv_name}'")


def sample_neutral(config: SamplingConfig, dataset: Dataset, type, column, csv_name):
    # Samples with scores within range [-0.5, 0.5] are considered neuteral without LLM label assignment
    ds_filtered = dataset.filter(lambda x: -0.5 <= x["avg_score"] <= 0.5)

    ds_mapped = ds_filtered.map(
        lambda x: {
            "text": x[column],
            "type": x[type],
            "formality_label": "neutral",
            "formality_explanation": "",
        }
    )

    ds_mapped = ds_mapped.select_columns(
        ["text", "type", "formality_label", "formality_explanation"]
    )

    ds_mapped.to_csv(csv_name, index=False, quoting=csv.QUOTE_ALL)
    print(f"Dataset saved as '{csv_name}'")


def split_sentences(text: str):
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]


def resample_text(text: str, input: str):
    sentences = split_sentences(text)
    n = len(sentences)

    if n == 1:
        return text

    if n < 2:
        return text

    # If the input csv is govreport_formal_sampled.csv, three random sentences are resampled
    # This is because this dataset contained very long text samples (more than 10 sentences each), it made sense to resample for more instances here
    if input == "govreport_formal_sampled.csv":
        chosen_indices = random.sample(range(n), 3)
        sample1 = sentences[chosen_indices[0]]
        sample2 = sentences[chosen_indices[1]]
        sample3 = sentences[chosen_indices[2]]
        return [sample1, sample2, sample3]

    # Otherwise two random sentences are resampled
    chosen_indices = random.sample(range(n), 2)
    sample1 = sentences[chosen_indices[0]]
    sample2 = sentences[chosen_indices[1]]
    return [sample1, sample2]


def resample_csv(input_csv: str, output_csv: str, text_column: str, type_column: str):
    df = pd.read_csv(input_csv)

    new_rows = []
    for _, row in df.iterrows():
        text = row[text_column]
        type_val = row[type_column]
        resampled = resample_text(text, input_csv)

        if isinstance(resampled, list):
            for new_text in resampled:
                new_rows.append({text_column: new_text, type_column: type_val})
        else:
            new_rows.append({text_column: resampled, type_column: type_val})

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(output_csv, index=False, header=False, quoting=csv.QUOTE_ALL)
    print(f"Resampled CSV saved as '{output_csv}'")


def main(config: SamplingConfig = SamplingConfig()):
    random.seed(config.seed)

    # 1. Sampling from datasets of different levels of formality

    # Informal -- reddit comments
    ds_reddit = load_dataset(
        "HuggingFaceGECLM/REDDIT_comments",
        verification_mode=VerificationMode.NO_CHECKS,
        data_files={"tifu": "data/tifu*"},
        split="tifu",
    )
    sample_dataset(config, ds_reddit, 1500, "reddit_comment", "body", "data/reddit_comments_sampled.csv")

    # Formal -- medical information
    ds_pubmed = load_dataset(
        "MedRAG/pubmed",
        verification_mode=VerificationMode.NO_CHECKS,
        # num_proc=3,
        split="train",
        data_files="chunk/pubmed23n000*.jsonl"
    )
    sample_dataset(config, ds_pubmed, 300, "pubmed_formal", "content", "data/pubmed_formal_sampled.csv")

    # Formal -- government documents
    ds_govreport = load_dataset(
        "ccdv/govreport-summarization",
        verification_mode=VerificationMode.NO_CHECKS,
        # num_proc=3,
        split="test",
    )
    sample_dataset(config, ds_govreport, 300, "govreport_formal", "report", "data/govreport_formal_sampled.csv")

    # Neutral -- online communication
    ds_neutral = load_dataset(
        "osyvokon/pavlick-formality-scores",
        verification_mode=VerificationMode.NO_CHECKS,
        # num_proc=3,
        split="test",
    )
    sample_neutral(config, ds_neutral, "domain", "sentence", "neutral_sampled.csv")

    # 2. Resampling of the data for the consecutively merged dataset to be more balanced
    resample_csv("data/pubmed_formal_sampled.csv", "data/pubmed_formal_resampled.csv", "content", "type")
    resample_csv("data/govreport_formal_sampled.csv", "data/govreport_formal_resampled.csv", "report", "type")


if __name__ == "__main__":
    main()
