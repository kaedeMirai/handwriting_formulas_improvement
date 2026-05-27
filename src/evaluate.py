import re
import json
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_PATH = "checkpoints/qwen3vl_2b_latex_ocr_math"
DATASET_ID = "linxy/LaTeX_OCR"
PROMPT = "Convert the formula in the image to LaTeX. Return only LaTeX."

MAX_SAMPLES = 70
MAX_NEW_TOKENS = 384

OUTPUT_JSON = "eval_results_qwen3vl_2b_latex_ocr_math.json"
ONE_SHOT = False


def levenshtein_distance(a: str, b: str) -> int:
    """
    Мера различия между двумя строками.
    """

    if a == b:
        return 0

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current_row = [i]

        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (ca != cb)

            current_row.append(min(insert_cost, delete_cost, replace_cost))

        previous_row = current_row

    return previous_row[-1]


def cer(prediction: str, target: str) -> float:
    """
    Character Error Rate.
    """
    if len(target) == 0:
        return 1.0 if prediction else 0.0

    return levenshtein_distance(prediction, target) / len(target)


def generate_prediction(model, processor, image, demo_image, demo_latex):
    """
    Генерирует LaTeX для одного изображения.
    """
    if ONE_SHOT:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Example:\n"
                            "This image corresponds to the following LaTeX:\n"
                            f"{demo_latex}\n\n"
                            "Now convert the next formula image to LaTeX."
                        ),
                    },
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if ONE_SHOT:
        inputs = processor(
            text=[text],
            images=[[demo_image, image]],
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.0,
            # no_repeat_ngram_size=3,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    input_token_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_token_len:]

    prediction = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return prediction


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID)

    if "test" in dataset:
        test_dataset = dataset["test"].select(
            range(min(MAX_SAMPLES, len(dataset["test"])))
        )
    else:
        test_dataset = dataset["train"].select(range(MAX_SAMPLES))

    train_dataset = dataset["train"]
    demo_example = train_dataset[0]
    demo_image = demo_example["image"].convert("RGB")
    demo_latex = demo_example["text"].strip()

    print(f"Eval samples: {len(test_dataset)}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    processor.tokenizer.padding_side = "right"

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )

    base_model.config.pad_token_id = processor.tokenizer.pad_token_id

    print("Loading LoRA adapter...")
    if ONE_SHOT:
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    rows = []

    total_cer = 0.0
    exact_matches = 0

    for idx, example in enumerate(tqdm(test_dataset)):
        image = example["image"].convert("RGB")
        target = example["text"].strip()

        prediction = generate_prediction(
            model, processor, image, demo_image, demo_latex
        )

        sample_cer = cer(prediction, target)

        exact_match = int(prediction == target)

        total_cer += sample_cer
        exact_matches += exact_match

        rows.append(
            {
                "idx": idx,
                "target": target,
                "prediction": prediction,
                "cer": sample_cer,
                "exact_match": exact_match,
            }
        )

    n = len(test_dataset)

    metrics = {
        "model": "Qwen/Qwen3-VL-2B-Instruct + LoRA masked SFT",
        "adapter_path": ADAPTER_PATH,
        "num_samples": n,
        "mean_cer": total_cer / n,
        "exact_match": exact_matches / n,
    }

    output = {
        "metrics": metrics,
        "samples": rows,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n Evaluation results: ")
    print(f"Samples:        {n}")
    print(f"Mean CER:       {metrics['mean_cer']:.4f}")
    print(f"Exact Match:    {metrics['exact_match']:.4f}")

    print(f"\nSaved detailed results to: {OUTPUT_JSON}")

    print("\nFew examples: ")
    for row in rows[:5]:
        print("\n---")
        print("IDX:", row["idx"])
        print("TARGET:")
        print(row["target"])
        print("PREDICTION:")
        print(row["prediction"])
        print("CER:", round(row["cer"], 4))


if __name__ == "__main__":
    main()
