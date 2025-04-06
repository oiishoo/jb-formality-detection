import concurrent.futures
import json
import argparse
import threading
from pathlib import Path
from string import Template

import openai
import pandas as pd
from tqdm import tqdm


def read_prompt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def classify_formality(text, src, system_prompt, user_prompt_template, client):
    user_prompt = user_prompt_template.substitute(text=text, src=src)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        result_text = (
            response.choices[0]
            .message.content.removeprefix("```json")
            .removesuffix("```")
        )

        try:
            result = json.loads(result_text)
            if "label" not in result or "explanation" not in result:
                return {
                    "label": "error",
                    "explanation": f"Missing required fields in response: {result_text}",
                }
            return result
        except json.JSONDecodeError:
            return {
                "label": "error",
                "explanation": f"Failed to parse response as JSON: {result_text[:100]}...",
            }

    except Exception as e:
        return {"label": "error", "explanation": f"API error: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(
        description="Classify text formality using OpenAI API"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input CSV file path"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--system_prompt", "-s", type=str, required=True, help="System prompt file path"
    )
    parser.add_argument(
        "--user_prompt",
        "-u",
        type=str,
        required=True,
        help="User prompt template file path",
    )
    parser.add_argument(
        "--api_key", "-k", type=str, required=True, help="OpenAI API key"
    )
    parser.add_argument(
        "--text_column", type=str, default="body", help="Column name for text"
    )
    parser.add_argument(
        "--type_column", type=str, default="type", help="Column name for source type"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of samples to process before saving progress",
    )
    args = parser.parse_args()

    client = openai.OpenAI(api_key=args.api_key, base_url="https://api.deepseek.com")

    system_prompt = read_prompt_file(args.system_prompt)
    user_prompt_template = Template(read_prompt_file(args.user_prompt))

    df = pd.read_csv(args.input)

    if "formality_label" not in df.columns:
        df["formality_label"] = None
    if "formality_explanation" not in df.columns:
        df["formality_explanation"] = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_to_process = df[df["formality_label"].isna()].index.tolist()

    print(f"Processing {len(rows_to_process)} out of {len(df)} rows...")


    def process_row(idx, df, args, system_prompt, user_prompt_template, client, lock):
        if pd.notna(df.at[idx, "formality_label"]):
            return None

        row = df.loc[idx]
        text = row[args.text_column]
        src = row[args.type_column]

        result = classify_formality(
            text, src, system_prompt, user_prompt_template, client
        )

        with lock:
            df.at[idx, "formality_label"] = result.get("label")
            df.at[idx, "formality_explanation"] = result.get("explanation")
            return idx

        # time.sleep(0.5)

    lock = threading.Lock()
    processed_count = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(
                process_row,
                idx,
                df,
                args,
                system_prompt,
                user_prompt_template,
                client,
                lock,
            ): idx
            for idx in rows_to_process
        }

        with tqdm(total=len(rows_to_process)) as pbar:
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    processed_idx = future.result()
                    if processed_idx is not None:
                        with lock:
                            processed_count += 1
                            if processed_count % args.batch_size == 0:
                                df.to_csv(args.output, index=False)
                                print(
                                    f"Progress saved: {processed_count}/{len(rows_to_process)} samples processed"
                                )
                except Exception as e:
                    print(f"Error processing idx {idx}: {e}")
                finally:
                    pbar.update(1)

    df.to_csv(args.output, index=False)
    print(f"Complete! {processed_count} samples processed and saved to {args.output}")


if __name__ == "__main__":
    main()
