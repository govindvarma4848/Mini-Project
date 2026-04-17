from huggingface_hub import HfApi, login
import sys

# Replace this with your Hugging Face space repo ID (e.g., "username/space-name")
REPO_ID = "Adapa-Govind-Varma/legal-rag-backend"

def upload_folder_to_space(folder_path, path_in_repo):
    api = HfApi()
    print(f"Uploading {folder_path} to {path_in_repo} in {REPO_ID}...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="space"
        )
        print(f"Successfully uploaded {folder_path}!")
    except Exception as e:
        print(f"Error uploading {folder_path}: {e}")

if __name__ == "__main__":
    print("Welcome to the Hugging Face Uploader!")
    
    # 1. ADD YOUR TOKEN HERE (Get it from huggingface.co -> Settings -> Access Tokens)
    HF_TOKEN = "paste_your_token_here_starting_with_hf_" 
    
    if REPO_ID == "YOUR_USERNAME/YOUR_SPACE_NAME" or HF_TOKEN == "paste_your_token_here_starting_with_hf_":
        print("ERROR: Please edit this script first, set REPO_ID, and paste your HF_TOKEN.")
        sys.exit(1)

    # Login programmatically
    login(token=HF_TOKEN, add_to_git_credential=True)

    # Uploading the datasets and model folders
    upload_folder_to_space("./dataset", "dataset")
    
    # If your model is in the 'model' folder instead of 'fine_tuned_lora_model', we upload it:
    import os
    if os.path.exists("./model"):
        upload_folder_to_space("./model", "model")
    if os.path.exists("./fine_tuned_lora_model"):
        upload_folder_to_space("./fine_tuned_lora_model", "fine_tuned_lora_model")
    
    print("\nAll done! Your nested folders are now in your Hugging Face Space.")
