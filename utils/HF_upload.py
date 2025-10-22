from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="/home/kevingenghaopeng/MDD/IF-MDD/pretrained_models/CTC_pretrained_IF_MDD",
    repo_id="Haopeng/CTC_for_IF-MDD",
    repo_type="model",
)