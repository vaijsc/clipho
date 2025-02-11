import os
import sys
import signal
import json
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import PngImagePlugin, Image
from datasets import load_from_disk, Dataset


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None


def get_completion_chatgpt(model, client, question, generation_cofig):
    if args.use_system_prompt:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    else:
        messages = [
            {"role": "user", "content": question}
        ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **generation_cofig
    )
    response = completion.choices[0].message.content

    return response

def generate_id(example, idx):
    example["image_index"] = idx
    return example

# Define a signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("\nJob interrupted! Saving error indices...", flush=True)
    with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
        json.dump(error_indices, f_error, ensure_ascii=False, indent=4)
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="gemini-1.5-flash") # gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro, gpt-4-0125-preview, gpt-3.5-turbo-0125
    parser.add_argument("--api_num", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="/lustre/scratch/client/vinai/users/thinhphp1/text2img/vidata/images_27M")
    parser.add_argument("--data_name", type=str, default="images_27M")
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--caption_type", type=int, default=0, help="0 for detailed caption only, 1 for concise caption only, 2 for both")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--decoding", type=str, default="sampling")
    parser.add_argument("--block_non_safety", action="store_true")
    parser.add_argument("--fix_error", action="store_true")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    SYSTEM_PROMPT = "Bạn là một chuyên gia mô tả hình ảnh và bạn có hiểu biết, kiến thức sâu rộng về  lịch sử, văn hóa, cảnh quan và con người Việt Nam."

    if "gemini" in args.model_id:
        keys = [line.strip() for line in open("data/tmp/gemini_api_keys.txt").readlines()]
        genai.configure(api_key=keys[args.api_num])
        if args.use_system_prompt:
            model = genai.GenerativeModel(args.model_id, system_instruction=SYSTEM_PROMPT)
        else:
            model = genai.GenerativeModel(args.model_id)
    else:
        keys = [line.strip() for line in open("data/tmp/openai_api_keys.txt").readlines()]
        client = OpenAI(api_key=keys[args.api_num])

    if args.caption_type == 0:  
        question = "Viết mô tả chi tiết và đầy đủ về hình ảnh."
    elif args.caption_type == 1:
        question = "Viết mô tả ngắn gọn và súc tích về hình ảnh."
    else:
        question = "Viết hai mô tả cho hình ảnh này: một mô tả chi tiết và một mô tả ngắn gọn.\n**Mô tả chi tiết:** (Một mô tả đầy đủ và chi tiết về hình ảnh, bao gồm nhiều thông tin nhất có thể, chẳng hạn như các đối tượng, hành động, cảm xúc, bối cảnh, màu sắc và chi tiết nền)\n**Mô tả ngắn gọn:** (Một mô tả ngắn gọn và trực tiếp về hình ảnh)"

    if args.sanity_check:
        paths = [f"data/images/{file}" for file in os.listdir("data/images/")]
        images = [Image.open(path) for path in paths]
        dataset_dict = {"image": images, "path": paths}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(generate_id, with_indices=True)
        end_index = len(dataset) if args.end_index == -1 else args.end_index
    else:
        dataset = load_from_disk(args.data_path)
        end_index = len(dataset) if args.end_index == -1 else args.end_index
        dataset = dataset.select(list(range(args.start_index, end_index)))
        dataset = dataset.map(generate_id, with_indices=True, batched=True, batch_size=4000, num_proc=4)

        if args.fix_error:
            with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json") as f_error:
                error_indices = json.load(f_error)
            if len(error_indices) == 0:
                print("### Done fixing error indices ###")
                dataset = []
            else:
                print("### Start fixing error indices ###")
                dataset = dataset.select(error_indices)

    genration_config = dict()
    genration_config["top_p"] = 0.8

    if "gemini" in args.model_id:
        genration_config["max_output_tokens"] = 512
        genration_config["top_k"] = 40
    else:
        genration_config["max_tokens"] = 512
        genration_config["seed"] = args.seed

    if args.decoding == "greedy":
        genration_config["temperature"] = 0
    elif args.decoding == "sampling":
        genration_config["temperature"] = 0.8
    else:
        raise Exception("### Invalid decoding method ###")

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    error_indices = []
    if dataset != []:
        try:
            for item in tqdm(dataset, total=len(dataset)):
                prompt = question
                image = item["image"]

                if "gemini" in args.model_id:
                    try:
                        if args.block_non_safety:
                            gemini_response = model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(**genration_config))
                            try:
                                response = gemini_response.text
                            except:
                                response = "NON SAFETY"
                        else:
                            gemini_response = model.generate_content([prompt, image], generation_config=genai.types.GenerationConfig(**genration_config), safety_settings=safety_settings)
                            try:
                                response = gemini_response.text
                            except:
                                response = "NON SAFETY"

                        output_data = {"image_index": item["image_index"], "img_path": item["path"], "caption": response}
                        with jsonlines.open(f"output/raw_captions/captions_{args.data_name}_{args.start_index}_{end_index}.jsonl", 'a') as f_caption:
                            f_caption.write(output_data)
                    except Exception as e:
                        error_indices.append(item["image_index"])
                        print(e, flush=True)
                else:
                    try:
                        response = get_completion_chatgpt(args.model_id, client, question, genration_config)
                        output_data = {"image_index": item["image_index"], "img_path": item["path"], "caption": response}
                        with jsonlines.open(f"output/raw_captions/captions_{args.data_name}_{args.start_index}_{end_index}.jsonl", 'a') as f_caption:
                            f_caption.write(output_data)
                    except Exception as e:
                        error_indices.append(item["image_index"])
                        print(e, flush=True)
        except KeyboardInterrupt:
            print("\nProcess interrupted! Saving error indices...", flush=True)
            with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
                json.dump(error_indices, f_error, ensure_ascii=False, indent=4)
            sys.exit(0)

        with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
            json.dump(error_indices, f_error, ensure_ascii=False, indent=4)