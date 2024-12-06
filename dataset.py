# Modified script from: https://stackoverflow.com/a/39225272
import os
import sys
import zipfile

import requests
from tqdm import tqdm


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")


def extract_zip(zip_path, extract_path):
    print(f"Extract {zip_path} to {extract_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def download_file_from_google_drive(file_id, destination):
    print(f"Download {file_id} to {destination}")

    URL = "https://drive.usercontent.google.com/download"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id, "confirm": "t"}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc="Downloading",
                          total=(int(response.headers.get("Content-Length")) // 32768 + 1)):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    file_id = "1_RjOMwaW8eFrm1dBNtKeuDUAZ9ZG9-Sh"
    destination = sys.path[0] + "/data.zip"
    download_file_from_google_drive(file_id, destination)

    extract_path = sys.path[0]
    extract_zip(destination, extract_path)

    delete_file(destination)
