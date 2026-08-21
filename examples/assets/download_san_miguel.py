import os
import urllib.request
import zipfile
import sys

URL = "https://casual-effects.com/g3d/data10/research/model/San_Miguel/San_Miguel.zip"
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_FILE = os.path.join(ASSETS_DIR, "San_Miguel.zip")
TARGET_MODEL_DIR = os.path.join(ASSETS_DIR, "San_Miguel")

def download_progress(count, block_size, total_size):
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

def main():
    print("=============================================")
    print("   San Miguel Model Downloader for Bazalt    ")
    print("=============================================")
    
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)

    if os.path.exists(os.path.join(TARGET_MODEL_DIR, "san-miguel.obj")):
        print(f"Model is already downloaded and extracted in:\n{TARGET_MODEL_DIR}")
        return

    print(f"Downloading from:\n{URL}")
    print("This file is ~500MB, it may take a while depending on your connection.")
    
    try:
        urllib.request.urlretrieve(URL, ZIP_FILE, reporthook=download_progress)
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return

    print("Extracting files...")
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            # Check if it contains a top-level directory "San_Miguel"
            top_level_items = set(item.split('/')[0] for item in zip_ref.namelist())
            
            if len(top_level_items) == 1 and list(top_level_items)[0].lower() == 'san_miguel':
                # Zip already contains the folder, extract to ASSETS_DIR
                extract_target = ASSETS_DIR
            else:
                # Zip contains files directly or something else, create San_Miguel folder
                extract_target = TARGET_MODEL_DIR
                if not os.path.exists(extract_target):
                    os.makedirs(extract_target)
                    
            zip_ref.extractall(extract_target)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip file: {e}")
        return
    finally:
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
            print("Cleaned up temporary zip file.")
            
    print("\nDone! You can now run the 07_model_loading example.")

if __name__ == "__main__":
    main()
