import os, gc
import json
import jsonlines
import contextlib
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from PIL import PngImagePlugin, Image
from datasets import load_from_disk, Dataset
from datasets import disable_caching
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None


def generate_id(example, idx):
    example["image_index"] = idx
    return example

def load_model_and_tokenizer(args):
    llm = LLM(args.model_id, trust_remote_code=True, tensor_parallel_size=args.n_gpus, dtype=args.dtype, gpu_memory_utilization=0.95, max_model_len=args.max_model_len, download_dir=download_dir)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

def unload_model():
    global llm
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--data_path", type=str, default="/lustre/scratch/client/vinai/users/thinhphp1/text2img/vidata/images_27M")
    parser.add_argument("--data_name", type=str, default="images_27M")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--caption_type", type=int, default=0, help="0 for detailed caption only, 1 for concise caption only, 2 for both")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--decoding", type=str, default="sampling")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--fix_error", action="store_true")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    download_dir = "../.cache/huggingface/hub/"

    llm, tokenizer = load_model_and_tokenizer(args)

    if args.caption_type == 0:  
        question = "Write comprehensive and detailed description of the image."
    elif args.caption_type == 1:
        question = "Write a brief and to-the-point description of the image."
    else:
        question = "Write two captions for this image: a detailed one and a concise one.\nYour captions should adhere to the specific format below:\n**Detailed caption:** (A comprehensive and detailed description of the image, including as much information as possible, such as objects, actions, emotions, context, colors and background details)\n**Concise caption:** (A brief and to-the-point description of the image)"

    if "InternVL" in args.model_id:
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        user_content = f"<image>\n{question}"
    elif "MiniCPM-V":
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        user_content = f"(<image>./</image>)\n{question}"
    else:
        raise Exception("Check for stop_tokens and prompt_format of model")
    
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    if args.decoding == "greedy":
        sampling_params = SamplingParams(best_of=1, temperature=0.0, max_tokens=512, stop_token_ids=stop_token_ids)
    else:
        sampling_params = SamplingParams(temperature=0.8, top_k=40, top_p=0.8, max_tokens=512, stop_token_ids=stop_token_ids)

    if args.use_system_prompt:
        system_prompt = "You are a skilled image captioning expert specializing in Vietnamese culture, landscapes, and people."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    else:
        messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        dataset = dataset.map(generate_id, with_indices=True, batched=True, batch_size=4000, num_proc=16)

        if args.fix_error:
            with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json") as f_error:
                error_indices = json.load(f_error)
            if len(error_indices) == 0:
                print("### Done fixing error indices ###")
                dataset = []
            else:
                print("### Start fixing error indices ###")
                dataset = dataset.select(error_indices)

    error_indices = []
    if dataset != []:
        for i in tqdm(range(0, len(dataset), args.batch_size), total=len(range(0, len(dataset), args.batch_size))):
            try:
                batch = dataset[i: i + args.batch_size]
            except Exception as e:
                print(e, flush=True)
                # TODO: Hanlding error from firts iteration
                error_indices.extend([index + args.batch_size for index in batch["image_index"]])
                with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
                    json.dump(error_indices, f_error, ensure_ascii=False, indent=4)
                continue

            try:
                inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in batch["image"]]
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                captions = [output.outputs[0].text for output in outputs]
                output_data = [{"image_index": batch["image_index"][j] + args.start_index, "img_path": batch["path"][j], "caption": captions[j]} for j in range(len(batch["path"]))]
                with jsonlines.open(f"output/raw_captions/captions_{args.data_name}_{args.start_index}_{end_index}.jsonl", 'a') as f_caption:
                    f_caption.write_all(output_data)
            except Exception as e:
                print(e, flush=True)
                error_indices.extend(batch["image_index"])
                with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
                    json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

                # Unload then reload model and tokenizer
                unload_model()
                llm, tokenizer = load_model_and_tokenizer(args)

        # Finish fixing error indices
        with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
            json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

