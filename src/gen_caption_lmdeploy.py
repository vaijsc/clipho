from lmdeploy import pipeline, PytorchEngineConfig, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig, VisionConfig
from PIL import PngImagePlugin
from datasets import load_from_disk
import jsonlines
import json
from tqdm import tqdm
from argparse import ArgumentParser


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def generate_id(example, idx):
    example["image_index"] = idx
    return example


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--data_path", type=str, default="/lustre/scratch/client/vinai/users/thinhphp1/text2img/vidata/images_27M")
    parser.add_argument("--data_name", type=str, default="images_27M")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--use_system_prompt", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--decoding", type=str, default="sampling")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--session_len", type=int, default=16384)
    parser.add_argument("--fix_error", action="store_true")
    parser.add_argument("--n_gpus", type=int, default=1)
    args = parser.parse_args()

    download_dir = "../.cache/huggingface/hub/"
    question = "Describe the image in detail."

    if args.use_system_prompt:
        system_prompt = "You are a skilled image captioning expert specializing in Vietnamese culture, landscapes, and people. Provide detailed and informative captions."
        chat_template_config = ChatTemplateConfig("internvl2-internlm2")
        chat_template_config.meta_instruction = system_prompt
    else:
        chat_template_config = None

    vision_config=VisionConfig(max_batch_size=16)
    backend_config = PytorchEngineConfig(tp=args.n_gpus, session_len=args.session_len, max_batch_size=args.batch_size, download_dir=download_dir)
    # TurbomindEngine offers much better performance
    # backend_config = TurbomindEngineConfig(tp=args.n_gpus, session_len=args.session_len, max_batch_size=args.batch_size, download_dir=download_dir)

    if args.decoding == "greedy":
        gen_config = GenerationConfig(top_k=1, temperature=0.0, max_new_tokens=args.max_new_tokens)
    else:
        gen_config = GenerationConfig(top_k=args.top_k, top_p=args.top_p, temperature=args.temperature, max_new_tokens=args.max_new_tokens)

    pipe = pipeline(args.model_id, chat_template_config=chat_template_config, backend_config=backend_config, vision_config=vision_config)

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
            batch = dataset[i: i + args.batch_size]
            try:
                prompts = [(question, image) for image in batch["image"]]
                responses = pipe(prompts, gen_config=gen_config)
                captions = [response.text for response in responses]
                output_data = [{"image_index": batch["image_index"][j] + args.start_index, "img_path": batch["path"][j], "caption": captions[j]} for j in range(len(batch["path"]))]
                with jsonlines.open(f"output/captions_{args.data_name}_{args.start_index}_{end_index}.jsonl", 'a') as f_caption:
                    f_caption.write_all(output_data)
            except Exception as e:
                print(e, flush=True)
                error_indices.extend(batch["image_index"])
                with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
                    json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

    # Finish fixing error indices
    with open(f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json", 'w', encoding="utf-8") as f_error:
        json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

