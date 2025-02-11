import os
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="saves/openai_clip-vit-base-patch16_phobert_syllable_stage-2_no_gemini")
    parser.add_argument("--task", type=str, default="retrieval", help="retrieval / classification")
    parser.add_argument("--data_path", type=str, default="data/eval_data/crossmodal3600/crossmodal3600_vi_eval.json")
    parser.add_argument("--template_type", type=int, default=0, help="0 for default templates, 1 for custom templates")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    checkpoints = [checkpoint for checkpoint in os.listdir(args.checkpoint_dir) if "checkpoint-" in checkpoint]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
    checkpoints.append("") # The final checkpoint

    for checkpoint in tqdm(checkpoints, total=len(checkpoints)):
        checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint)
        print("### Checkpoint path ###")
        print(checkpoint_path)

        if args.task == "retrieval":
            eval_command = "python src/zeroshot_retrieval.py "
        else:
            eval_command = f"python src/zeroshot_classification.py --template_type {args.template_type} "

        eval_command += f"--model_path {checkpoint_path} --data_path {args.data_path} --batch_size {args.batch_size} --device {args.device}"
        
        os.system(eval_command)