import torch
import os
import argparse

def merge_models(folder_path, lang_list, model_name):
    models = []
    for lang in lang_list:
        
        model_path = os.path.join(folder_path, f"{lang}", f"{model_name}", "bert-fix_ctx-shared-bs64/pytorch_model.bin")
        print(model_path)
        model = torch.load(model_path, map_location="cpu")
        models.append(model)
        print(f"Model for {lang} loaded successfully")

    
    merged_model = {}
    for name, params in models[0].items():
        if "position_ids" not in name:
            new_param = sum(model[name] for model in models) / len(models)
            merged_model[name] = new_param
        else:
            merged_model[name] = params

    save_path = os.path.join(folder_path, "merged", f"{model_name}", "pytorch_model.bin")

    save_folder = os.path.dirname(save_path)

    print(save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    torch.save(merged_model, save_path)
    print("Merged Model saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge models for different languages")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the models")
    parser.add_argument("--model", type=str, required=True, help="Model Name")
    parser.add_argument("--lang_list", type=str, required=True, help="Comma-separated list of languages")

    args = parser.parse_args()
    folder_path = args.folder
    lang_list = args.lang_list.split(',')
    print(lang_list)
    model_name = args.model

    merge_models(folder_path, lang_list, model_name)





