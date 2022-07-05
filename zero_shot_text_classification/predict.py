import os
import pathlib
import zipfile
from transformers import pipeline


def unzip_model(zip_model_path):
    p = pathlib.Path(zip_model_path)
    root = str(p.parents[0])
    print(f'start unzip: {root}')
    model_path = os.path.join(root, os.path.splitext(os.path.basename(zip_model_path))[0] + '/')
    if not os.path.isdir(model_path):
        with zipfile.ZipFile(zip_model_path, "r") as zip_ref:
            zip_ref.extractall(root)
        
    print(f"finish zipping model to {model_path}")
    return model_path

def load_model(model_path):
    classifier = pipeline("zero-shot-classification",
                        model=model_path)
    return classifier

def predict_labels(text, candidate_labels, classifier, is_multi_labels):
    candidate_labels = [i.strip() for i in candidate_labels.split(",")]
    if is_multi_labels == "false":
        res = classifier(text, candidate_labels)
    else:
        res = classifier(text, candidate_labels, multi_class=True)

    return res