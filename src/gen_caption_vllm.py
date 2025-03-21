import os, gc
import json
import jsonlines
# import contextlib
from tqdm import tqdm
# import torch
from argparse import ArgumentParser
from PIL import PngImagePlugin, Image
from datasets import load_from_disk, Dataset
# from datasets import disable_caching
from vllm import LLM, SamplingParams
# from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None

# disable_caching()

def generate_id(example, idx):
    example["image_index"] = idx
    return example

def load_model_and_tokenizer(args):
    if "Qwen2" in args.model_id:
        llm = LLM(args.model_id, trust_remote_code=True, max_num_seqs=4, tensor_parallel_size=args.n_gpus, dtype=args.dtype, gpu_memory_utilization=0.95, max_model_len=args.max_model_len, download_dir=download_dir)
    else:
        llm = LLM(args.model_id, trust_remote_code=True, max_num_seqs=2, tensor_parallel_size=args.n_gpus, dtype=args.dtype, gpu_memory_utilization=0.95, max_model_len=args.max_model_len, download_dir=download_dir)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

# def unload_model():
#     global llm
#     destroy_model_parallel()
#     destroy_distributed_environment()
#     with contextlib.suppress(AssertionError):
#         torch.distributed.destroy_process_group()
#     del llm
#     gc.collect()
#     torch.cuda.empty_cache()


INSTRUCTION_8_TOKENS = """
You are a Concise Image Captioning Expert. Your task is to generate a brief image caption, precisely 8 tokens in length.

Caption Requirements:
1. Length: Approximately 8 tokens.
2. Focus: Main subject and primary action/state.
3. Style: Concise, present tense.
4. Content Priority:
    - Subject
    - Action/State
    - Context (if space allows)
5. Exclude: Articles (a, an, the) where possible.

Generate a caption that is around 8 tokens long, focusing on the most essential elements of the image.
""".strip()

INSTRUCTION_16_TOKENS = """
You are a Succinct Image Captioning Expert. Your task is to generate a succinct image caption, approximately 16 tokens in length.

Caption Requirements:
1. Length: Approximately 16 tokens.
2. Focus: Key subjects, actions, and essential context.
3. Style: Clear, descriptive, present tense.
4. Content Elements:
    - Main subject(s)
    - Significant actions/states
    - Relevant context/setting
5. Tone: Objective and factual.

Generate a caption that is around 16 tokens, capturing the key elements and context of the image in a clear and concise manner.
""".strip()

INSTRUCTION_32_TOKENS = """
You are a Descriptive Image Captioning Expert. Your task is to generate a descriptive image caption, approximately 32 tokens in length.

Caption Requirements:
1. Length: Approximately 32 tokens.
2. Focus: Main subject, salient details, and relevant context.
3. Style: Clear, objective, present tense.
4. Content Elements:
    - Main subject(s)
    - Key actions or states
    - Important context (setting, time of day, atmosphere)
    - Notable attributes of subjects
5. Avoid: Subjective interpretations or assumptions.

Generate a caption that is around 32 tokens, providing a descriptive overview of the image's key elements and context.
""".strip()

INSTRUCTION_64_TOKENS = """
You are a Detailed Image Captioning Expert. Your task is to generate a detailed image caption, approximately 64 tokens in length.

Caption Requirements:
1. Length: Approximately 64 tokens.
2. Focus: Comprehensive details about subjects, actions, setting, and visual attributes.
3. Style: Natural, flowing language, descriptive, present tense.
4. Content Elements:
    - Main subjects/objects and their attributes
    - Actions or interactions
    - Setting/environment and spatial relationships
    - Notable colors, textures, patterns
    - Lighting and atmosphere
5. Tone: Neutral and descriptive.

Generate a caption that is around 64 tokens, offering a detailed description of the image, including subjects, actions, setting, and key visual details.
""".strip()

INSTRUCTION_128_TOKENS = """
You are an In-depth Image Captioning Expert. Your task is to generate an in-depth image caption, approximately 128 tokens in length.

Caption Requirements:
1. Length: Approximately 128 tokens.
2. Focus: Extensive details about the scene, subjects, context, and technical aspects if applicable.
3. Style: Clear, concise yet detailed, present tense.
4. Content Elements:
    - High-level description of the scene/subject
    - Main subjects and detailed attributes (appearance, actions, expressions)
    - Setting and environment with background elements
    - Lighting and atmosphere in detail
    - Relevant context and background information
    - Photography style, composition (if discernible and relevant)
5. Tone: Objective and informative.

Generate a caption that is around 128 tokens, providing an in-depth and informative description of the image, covering content, context, and relevant technical aspects.
""".strip()

INSTRUCTION_256_TOKENS = """
You are an Exhaustive Image Captioning Expert. Your task is to generate a comprehensive image caption, approximately 256 tokens in length.

Caption Requirements:
1. Length: Approximately 256 tokens.
2. Focus: Exhaustive details covering all discernible aspects of the image, including minute details and subtle nuances.
3. Style: Highly detailed, descriptive, articulate, present tense.
4. Content Elements:
    - Comprehensive Scene Overview: Begin with an encompassing description of the entire scene.
    - Primary Subjects: In-depth description of main subjects, including appearance, actions, expressions, posture, gestures, and interactions.
    - Detailed Setting/Environment: Describe the environment exhaustively, including specific objects, ambient conditions (weather, time of day, sounds if suggested), and spatial relationships between elements.
    - Lighting and Atmosphere: Capture the nuances of lighting, shadows, and overall atmosphere, including mood and emotional tone.
    - Visual Attributes: Detail colors, textures, patterns, materials, surfaces, reflections, and any visual effects or artistic styles.
    - Contextual Information: Include relevant background context, historical or cultural references if discernible, and any implied narrative.
    - Technical Details (if applicable and discernible):  Mention photography style (e.g., macro, panoramic), composition techniques, camera viewpoint, lens effects, or post-processing effects if they significantly contribute to the image's interpretation.
5. Tone: Objective, informative, and thoroughly descriptive.

Generate a caption that is around 256 tokens, providing an exhaustive and comprehensive description of the image, covering all discernible details and nuances.
""".strip()

INSTRUCTIONS = {
    8: INSTRUCTION_8_TOKENS,
    16: INSTRUCTION_16_TOKENS,
    32: INSTRUCTION_32_TOKENS,
    64: INSTRUCTION_64_TOKENS,
    128: INSTRUCTION_128_TOKENS,
    256: INSTRUCTION_256_TOKENS
}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str, default="/lustre/scratch/client/movian/asset/datasets/thinhphp1/text2img/vidata/images_27M")
    parser.add_argument("--data_name", type=str, default="images_27M_refined_prompts")
    parser.add_argument("--start_index", type=int, default=0, help="Absolute index")
    parser.add_argument("--end_index", type=int, default=-1, help="Absolute index")
    parser.add_argument("--mid_index", type=int, default=None, help="Absolute index, don't use togher with --fix_error")
    parser.add_argument("--caption_length", type=int, default=8, help="8, 16, 32, 64, 128, 256 tokens")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--decoding", type=str, default="sampling")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--fix_error", action="store_true", help="Fix relative error indices")
    parser.add_argument("--sanity_check", action="store_true")
    args = parser.parse_args()

    download_dir = "/lustre/scratch/client/movian/applied/users/dungnn28/.cache/huggingface/hub/"

    llm, tokenizer = load_model_and_tokenizer(args)

    question = INSTRUCTIONS[args.caption_length]

    if "InternVL2" in args.model_id:
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        user_content = f"<image>\n{question}"
    elif "MiniCPM" in args.model_id:
        stop_tokens = ['<|im_end|>', '<|endoftext|>']
        user_content = f"(<image>./</image>)\n{question}"
    elif "Qwen2" in args.model_id:
        user_content = f"<|vision_start|><|image_pad|><|vision_end|>{question}"
        stop_tokens = []
    else:
        raise Exception("Check for stop_tokens and prompt_format of model")
    
    if stop_tokens:
        stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    else:
        stop_token_ids = None

    if args.decoding == "greedy":
        sampling_params = SamplingParams(best_of=1, temperature=0.0, max_tokens=200, stop_token_ids=stop_token_ids)
    else:
        sampling_params = SamplingParams(temperature=0.8, top_k=40, top_p=0.8, max_tokens=200, stop_token_ids=stop_token_ids)

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.sanity_check:
        paths = [f"data/images_27M/images/{file}" for file in os.listdir("data/images_27M/images")]
        images = [Image.open(path) for path in paths]
        dataset_dict = {"image": images, "path": paths}
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(generate_id, with_indices=True)
        end_index = len(dataset) if args.end_index == -1 else args.end_index
        error_indices = []
    else:
        dataset = load_from_disk(args.data_path)
        end_index = len(dataset) if args.end_index == -1 else args.end_index
        dataset = dataset.select(list(range(args.start_index, end_index)))
        dataset = dataset.map(generate_id, with_indices=True, batched=True, batch_size=4000, num_proc=16)

        error_indices_path = f"output/error_indices/error_indices_{args.data_name}_{args.start_index}_{end_index}.json"
        captions_path = f"output/raw_captions/captions_{args.data_name}_{args.start_index}_{end_index}.jsonl"

        if args.mid_index:
            dataset = dataset.select(list(range(args.mid_index - args.start_index, end_index - args.start_index)))
            if os.path.exists(error_indices_path):
                with open(error_indices_path) as f_error:
                    error_indices = json.load(f_error)
            else:
                error_indices = []
        elif args.fix_error:
            with open(error_indices_path) as f_error:
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
                with open(error_indices_path, 'w', encoding="utf-8") as f_error:
                    json.dump(error_indices, f_error, ensure_ascii=False, indent=4)
                continue
            try:
                inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} for image in batch["image"]]
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                captions = [output.outputs[0].text.strip('"').strip('\\') for output in outputs]
                output_data = [{"image_index": batch["image_index"][j] + args.start_index, "img_path": batch["path"][j], "caption": captions[j]} for j in range(len(batch["path"]))]
                with jsonlines.open(captions_path, 'a') as f_caption:
                    f_caption.write_all(output_data)
            except Exception as e:
                print(e, flush=True)
                error_indices.extend(batch["image_index"])
                with open(error_indices_path, 'w', encoding="utf-8") as f_error:
                    json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

                # # Unload then reload model
                # unload_model()
                # llm, tokenizer = load_model_and_tokenizer(args)

        # Finish fixing error indices
        with open(error_indices_path, 'w', encoding="utf-8") as f_error:
            json.dump(error_indices, f_error, ensure_ascii=False, indent=4)

