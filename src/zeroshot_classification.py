import os
import json
from tqdm import tqdm
from PIL import PngImagePlugin, Image
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, balanced_accuracy_score
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
    

def zero_shot_classifier(model, classnames, templates, device):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.

    model: 
        UniCLIP model
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    # autocast = torch.cuda.amp.autocast if amp else suppress
    # with torch.no_grad(), autocast():
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            class_embeddings = model.get_text_features(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def run_classification(model, classifier, data, batch_size, device, amp=True):
    """
    Run zero-shot classifcation

    model: 
        CLIP-like model
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    # autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []

    with torch.no_grad():
        for batch_index in tqdm(range(0, len(data["image_paths"]), batch_size), total=len(range(0, len(data["image_paths"]), batch_size))):
            image_paths = data["image_paths"][batch_index: batch_index + batch_size]
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            image_features = model.get_image_features(images)
            image_features = F.normalize(image_features, dim=-1)
            
            targets = torch.tensor(data["targets"][batch_index: batch_index + batch_size]).to(device)
            logits = 100. * image_features @ classifier
            
            true.append(targets.cpu())
            pred.append(logits.float().cpu())

    pred = torch.cat(pred)
    true = torch.cat(true)

    return pred, true

def average_precision_per_class(scores, targets):
    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    
    return ap

def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]

def evaluate(model, data, batch_size, classnames, templates, device, verbose=False):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: 
        UniCLIP model

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    classifier = zero_shot_classifier(model, classnames, templates, device)
    logits, target = run_classification(model, classifier, data, batch_size, device)
    is_multilabel = (len(target.shape) == 2)

    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class = average_precision_per_class(logits, target)
        if verbose:
            for class_name, ap in zip(data["classnames"], ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
        return {"mean_average_precision": ap_per_class.mean().item()}
    else:
        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall
        pred = logits.argmax(axis=1)
        # measure accuracy
        if len(data["classnames"]) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan")
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))

        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall}
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="saves/google_siglip-base-patch16-384_phobert_syllable_base_512_image_27M_all_stage-1_shuffle_no_gemini_fix") 
    parser.add_argument("--pretrained", type=str, default="", help="pretrained for open_clip models")
    parser.add_argument("--data_path", type=str, default="data/eval_data/crossmodal3600/crossmodal3600_captions-vi_eval.json")
    parser.add_argument("--template_type", type=int, default=0, help="0 for default templates, 1 for custom templates")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for compute logits when infer for each image / caption")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    model_path = args.model_path.rstrip('/')
    pretrained = args.pretrained
    data_path = args.data_path
    template_type = args.template_type
    batch_size = args.batch_size
    device = f"cuda:{args.device}"
    
    if pretrained == "":
        model_type = "hf"
        eval_output_file = f"tmp/evaluation/classification_results_{data_path.split('/')[-1].replace('.json', '')}_{model_path.split('/')[-2]}_{model_path.split('/')[-1]}_{model_type}.json"
    else:
        model_type = "open_clip" 
        eval_output_file = f"tmp/evaluation/classification_results_{data_path.split('/')[-1].replace('.json', '')}_{model_path}_{pretrained}_{model_type}.json"
    
    if os.path.exists(eval_output_file):
        print(f"Skip evaluate {model_path}-{model_type} on {data_path.split('/')[-1].replace('.json', '')}")
        print(f"Eval output file: {eval_output_file}")
    else:
        with open(data_path) as f:
            data = json.load(f)

        classnames = data["classnames"]
        if args.template_type == 0:
            templates = data["default_templates"]
        else:
            templates = data["custom_templates"]
        uni_clip = UniCLIP(model_path, pretrained, device)
        metrics = evaluate(uni_clip, data, batch_size, classnames, templates, device)

        print("### Eval Metrics")
        for k, v in metrics.items():
            print(f"{k} = {v}")

        with open(eval_output_file, 'w', encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)