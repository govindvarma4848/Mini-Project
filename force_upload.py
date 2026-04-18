from huggingface_hub import HfApi, login
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your repo details
REPO_ID = os.getenv("REPO_ID", "Adapa-Govind-Varma/legal-rag-backend")
HF_TOKEN = os.getenv("HF_TOKEN")

def force_upload():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not found in environment variables. Please check your .env file.")
        return

    print("Starting FORCE upload...")
    api = HfApi()
    
    # Login
    login(token=HF_TOKEN)
    
    # 1. Force upload critical modified files individually
    critical_files = [
        ("backend/api.py", "backend/api.py"),
        ("backend/rag_pipeline.py", "backend/rag_pipeline.py"),
        ("datasets/legal_dataset_small.csv", "datasets/legal_dataset_small.csv"),
        ("requirements.txt", "requirements.txt"),
    ]
    
    for local_path, repo_path in critical_files:
        if os.path.exists(local_path):
            print(f"Force uploading {local_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=REPO_ID,
                    repo_type="space"
                )
            except Exception as e:
                print(f"Error uploading {local_path}: {e}")
    
    # 2. Sync the rest of the folders
    print("Syncing backend folder...")
    api.upload_folder(
        folder_path="./backend",
        path_in_repo="backend",
        repo_id=REPO_ID,
        repo_type="space"
    )
    
    print("Syncing datasets folder...")
    api.upload_folder(
        folder_path="./datasets",
        path_in_repo="datasets",
        repo_id=REPO_ID,
        repo_type="space"
    )

    print("\nFORCE Sync complete! Check your Space for the 'v1.2-dataset-fix' version.")

if __name__ == "__main__":
    force_upload()
