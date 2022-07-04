import json
import numpy as np


config = {
    'static_folder': "static",
    'UPLOAD_FOLDER': 'static/images/',
    'secret_key': "546421349874624",
    'debug': False,
    'json_config_path': 'config/',
    'onnx_model_path': 'model/classifier.onnx',
    'threshold': 0.35
}


def import_json_model_config(json_config_path: str):
    # Opening model JSON config
    with open(''.join([json_config_path, 'model.json']), 'r') as f:
        model_conf = json.load(f)
    # Opening ingredients JSON config
    with open(''.join([json_config_path, 'ingredients.json']), 'r') as f:
        ingedients_config = json.load(f)
    class_labels = ingedients_config["idx"]
    id2rus_genitive = ingedients_config["id2rus_genitive"]
    id2rus_nominative = ingedients_config["id2rus_nominative"]
    class_labels_ru = np.array([id2rus_genitive[idx] for idx in class_labels])
    ingredients_text = "\n".join([id2rus_nominative[idx].capitalize() for idx in class_labels])
    return model_conf, class_labels_ru, ingredients_text
