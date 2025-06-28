from huggingface_hub import HfApi
import os

api = HfApi()



api.upload_folder(
    folder_path="feats",
    path_in_repo="data", # Upload to a specific folder
    repo_id="BayanDuygu/audio-feats",
    repo_type="dataset",
)

