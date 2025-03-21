import json
import jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import ArgumentParser


def translate(texts: str) -> str:
    input_ids = tokenizer(texts, padding=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **input_ids,
        decoder_start_token_id=tokenizer.lang_code_to_id[lang_code_to_id],
        num_return_sequences=1,
        num_beams=5,
        # num_beams=1,
        early_stopping=True
    )
    translated_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return translated_texts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/eval_data/KTVIC/KTVIC_vi_eval.json") 
    parser.add_argument("--data_type", type=int, default=0, help="0 for retrieval, 1 for classification, 2 for training data")
    parser.add_argument("--task", type=int, default=0, help="0 for en2vi, 1 for vi2en")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--start_index", type=int, default=0, help="Relative index")
    parser.add_argument("--end_index", type=int, default=-1, help="Relative index")
    parser.add_argument("--mid_index", type=int, default=0, help="Relative index")
    args = parser.parse_args()

    data_path = args.data_path
    batch_size = args.batch_size
    data_type = args.data_type

    if args.task == 0:
        model_id = "vinai/vinai-translate-en2vi-v2"
        lang_code_to_id = "vi_VN"
        src_lang = "en_XX"
    else:
        model_id = "vinai/vinai-translate-vi2en-v2"
        lang_code_to_id = "en_XX"
        src_lang = "vi_VN"

    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    if args.data_path.endswith(".json"):
        with open(args.data_path) as f:
            data = json.load(f)
    else:
        data = {"image_indices": [], "img_paths": [], "captions": []}
        with jsonlines.open(args.data_path) as f:
            for item in f:
                data["image_indices"].append(item["image_index"])
                data["img_paths"].append(item["img_path"])
                data["captions"].append(item["caption"])

    start_index = args.start_index
    mid_index = args.mid_index
    end_index = len(data["image_indices"]) if args.end_index == -1 else args.end_index

    for k, v in data.items():
        data[k] = v[start_index: end_index]

    data_folder, data_name = data_path.split('/')[-2], data_path.split('/')[-1]
    if args.data_type != 2:
        if args.task == 0:
            data_file = data_name.replace("_en_eval.json", "_vi_eval.json")
        else:
            data_file = data_name.replace("_vi_eval.json", "_en_eval.json")
    else:
        data_file = f"{data_name.split('.jsonl')[0]}_{start_index}_{end_index}.jsonl"

    if args.data_type == 0:
        all_keys = ["captions"]
        output_file = f"data/eval_data/zeroshot_retrieval{data_folder}/{data_file}"
    elif args.data_type == 1:
        all_keys = ["classnames", "default_templates", "custom_templates"]
        output_file = f"data/eval_data/zeroshot_classification/{data_folder}/{data_file}"
    else:
        all_keys = ["captions"]
        output_file = f"output/tmp/{data_file}"

    for key in all_keys:
        original_data = data[key]
        if original_data != []:
            original_data_batches = [original_data[i: i + batch_size] for i in range(0, len(original_data), batch_size)]
            translated_data = []

            for batch_index, original_data_batch in tqdm(enumerate(original_data_batches), total=len(original_data_batches)):
                if batch_index*batch_size >= mid_index:
                    translated_data_batch = translate(original_data_batch)
                    translated_data.extend(translated_data_batch)

                    if args.data_type == 2:
                        translated_data_batch_reformat = []
                        for i in range(batch_size):
                            index = batch_index*batch_size + i
                            translated_data_batch_reformat.append({
                                "index": index,
                                "image_index": data["image_indices"][index],
                                "img_path": data["img_paths"][index],
                                "caption": translated_data_batch[i]
                            })

                        with jsonlines.open(output_file, 'a') as f:
                            f.write_all(translated_data_batch_reformat)
            
            data[key] = translated_data

    if args.data_type != 2:
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)