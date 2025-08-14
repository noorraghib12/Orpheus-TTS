import argparse
import locale
import os
import torch
import random
import torchaudio.transforms as T
from snac import SNAC
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# ---- Argument Parser ----
parser = argparse.ArgumentParser(description="Tokenize dataset using SNAC and Orpheus format.")

parser.add_argument(
    "--original_dataset",
    type=str,
    default="noor-raghib-12/sust_banspeech-orpheus-formatted",
    help="Original Hugging Face dataset path"
)

parser.add_argument(
    "--output_dataset",
    type=str,
    default="noor-raghib-12/sust_banspeech-orpheus-tokenized",
    help="Name to push tokenized dataset to"
)

args = parser.parse_args()

my_original_dataset_name = args.original_dataset
name_to_push_dataset_to = args.output_dataset

# ---- Fix UTF-8 encoding for Hugging Face issues in some environments ----
locale.getpreferredencoding = lambda: "UTF-8"

# ---- Download snapshot ----
snapshot_download(
    repo_id=my_original_dataset_name,
    repo_type="dataset",
    revision="main",
    max_workers=64,
)

# ---- Load dataset ----
ds = load_dataset(my_original_dataset_name)
ds_sample_rate = ds['train'][0]["audio"]["sampling_rate"]

# ---- Load SNAC model ----
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

# ---- Tokenization Function ----
def tokenise_audio(waveform):
    waveform = waveform.unsqueeze(0).to(dtype=torch.float32)
    resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
    waveform = resample_transform(waveform).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item()+128266)
        all_codes.append(codes[1][0][2*i].item()+128266+4096)
        all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))

    return all_codes

# ---- Map Tokenization ----
def add_codes(example):
    codes_list = None
    try:
        answer_audio = example.get("audio")
        if answer_audio:
            audio_array = answer_audio._decoder
            codes_list = tokenise_audio(audio_array)
    except Exception as e:
        print(f"Skipping row due to error: {e}")
    example["codes_list"] = codes_list
    return example

ds = ds.map(add_codes, remove_columns=["audio"])

# ---- Tokenizer Setup ----
tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009

start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2

start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4

start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7
audio_tokens_start = tokeniser_length + 10

tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
num_proc = os.cpu_count() - 2

# ---- Filter problematic examples ----
ds = ds.filter(lambda x: x["codes_list"] is not None)
ds = ds.filter(lambda x: len(x["codes_list"]) > 0)

# ---- Remove Duplicate Frames ----
def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = vals[:7]
    for i in range(7, len(vals), 7):
        if vals[i] != result[-7]:
            result.extend(vals[i:i+7])

    example["codes_list"] = result
    return example

ds = ds.map(remove_duplicate_frames, num_proc=num_proc)

# ---- Display prompt info ----
print("*** Modify the text prompt here if needed. For example, use source-specific prompts.")

# ---- Create Input IDs ----
def create_input_ids(example):
    text_ids = tokenizer.encode(example["text"], add_special_tokens=True)
    text_ids.append(end_of_text)
    example["text_tokens"] = text_ids
    input_ids = (
        [start_of_human]
        + text_ids
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + example["codes_list"]
        + [end_of_speech]
        + [end_of_ai]
    )
    example["input_ids"] = input_ids
    example["labels"] = input_ids
    example["attention_mask"] = [1] * len(input_ids)
    return example

ds = ds.map(create_input_ids, num_proc=num_proc, remove_columns=["text", "codes_list"])

# ---- Final cleanup ----
columns_to_keep = ["input_ids", "labels", "attention_mask"]
columns_to_remove = [col for col in ds['train'].column_names if col not in columns_to_keep]
ds = ds.remove_columns(columns_to_remove)

# ---- Push to Hub ----
ds.push_to_hub(name_to_push_dataset_to)
