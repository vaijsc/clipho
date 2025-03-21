import os
import json
from tqdm import tqdm
from PIL import PngImagePlugin, Image
import numpy as np
import torch
import open_clip
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from argparse import ArgumentParser


LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)
Image.MAX_IMAGE_PIXELS = None


class UniCLIP():
    def __init__(self, model_path: str, pretrained: str="", device: str="cpu"):
        self.device = device

        if pretrained == "":
            self.hf_style = True
            self.image_processor = AutoImageProcessor.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        else:
            self.hf_style = False
            self.model, _, self.image_processor = open_clip.create_model_and_transforms(model_path, pretrained=pretrained)
            self.model.eval().to(device)
            self.tokenizer = open_clip.get_tokenizer(model_path)

    def get_image_features(self, images):
        if self.hf_style:
            image_inputs = self.image_processor(images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)
        else:
            image_inputs = torch.cat([self.image_processor(image).unsqueeze(0) for image in images]).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_inputs)
        return image_features

    def get_text_features(self, texts):
        if self.hf_style:
            # text_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
            text_inputs = self.tokenizer(texts, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
        else:
            text_inputs = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
        return text_features


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saves/google_siglip-base-patch16-384_phobert_syllable_base_512_image_27M_all_stage-1_shuffle_no_gemini_fix") 
    parser.add_argument("--pretrained", type=str, default="", help="pretrained for open_clip models")
    parser.add_argument("--data_path", type=str, default="data/eval_data/crossmodal3600/crossmodal3600_captions-vi_eval.json")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for compute logits when infer for each image / caption")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10], help="top_k for retrieval")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model_path.rstrip('/')
    pretrained = args.pretrained
    data_path = args.data_path
    batch_size = args.batch_size
    recall_k_list = args.top_k
    device = f"cuda:{args.device}"
    
    if pretrained == "":
        model_type = "hf"
        eval_output_file = f"tmp/evaluation/retrieval_results_{data_path.split('/')[-1].replace('.json', '')}_{model_path.split('/')[-2]}_{model_path.split('/')[-1]}_{model_type}.json"
    else:
        model_type = "open_clip" 
        eval_output_file = f"tmp/evaluation/retrieval_results_{data_path.split('/')[-1].replace('.json', '')}_{model_path}_{pretrained}_{model_type}.json"
    
    if os.path.exists(eval_output_file):
        print(f"Eval output: {eval_output_file}")
    else:
        with open(data_path) as f:
            data = json.load(f)

        uni_clip = UniCLIP(model_path, pretrained, device)

        # image_embeddings_file = f"tmp/embeddings/image_embeddings_{data_path.split('/')[-1].replace('.json', '')}_{model_path.split('/')[-2]}_{model_path.split('/')[-1]}_{model_type}.npy"
        # if os.path.exists(image_embeddings_file):
        #     print("### Loading image embeddings ###")
        #     with open(image_embeddings_file, 'rb') as f:
        #         image_embeddings = torch.from_numpy(np.load(f)).to(device)
        # else:
        images = [Image.open(image_path).convert("RGB") for image_path in data["image_paths"]]
        image_embeddings = []

        print("### Calculating image embeddings ###")
        for batch_index in tqdm(range(0, len(images), batch_size), total=len(range(0, len(images), batch_size))):
            batch_images = images[batch_index: batch_index + batch_size]
            batch_image_embeddings = uni_clip.get_image_features(batch_images)
            image_embeddings.append(batch_image_embeddings)

        image_embeddings = torch.cat(image_embeddings)
            # with open(image_embeddings_file, 'wb') as f:
            #     np.save(f, image_embeddings.to("cpu").numpy())

        # caption_embeddings_file = f"tmp/embeddings/caption_embeddings_{data_path.split('/')[-1].replace('.json', '')}_{model_path.split('/')[-2]}_{model_path.split('/')[-1]}_{model_type}.npy"
        # if os.path.exists(caption_embeddings_file):
        #     print("### Loading caption embeddings ###")
        #     with open(caption_embeddings_file, 'rb') as f:
        #         caption_embeddings = torch.from_numpy(np.load(f)).to(device)
        # else:
        captions = data["captions"]
        caption_embeddings = []

        print("### Calculating caption embeddings ###")
        for batch_index in tqdm(range(0, len(captions), batch_size), total=len(range(0, len(captions), batch_size))):
            batch_captions = captions[batch_index: batch_index + batch_size]
            batch_caption_embeddings = uni_clip.get_text_features(batch_captions)
            caption_embeddings.append(batch_caption_embeddings)

        caption_embeddings = torch.cat(caption_embeddings)
            # with open(caption_embeddings_file, 'wb') as f:
            #     np.save(f, caption_embeddings.to("cpu").numpy())

        # normalized features
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        caption_embeddings = caption_embeddings / caption_embeddings.norm(dim=-1, keepdim=True)

        # Calculate sim scores
        scores = caption_embeddings @ image_embeddings.t()

        # Store the index of image for each text
        texts_image_index = [None for _ in range(len(caption_embeddings))]
        for item in data["image_caption_pairs"]:
            for caption_index in item["caption_indicies"]:
                texts_image_index[caption_index] = item["image_index"]
        
        # Construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True

        metrics = {}
        for recall_k in recall_k_list:
            metrics[f"text2image_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k) > 0).float().mean().item()
            metrics[f"image2text_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k) > 0).float().mean().item()

        print("### Eval Metrics")
        for k, v in metrics.items():
            print(f"{k} = {v}")

        with open(eval_output_file, 'w', encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)