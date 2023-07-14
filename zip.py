import os
import zipfile

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), arcname=file)

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipObj:
        zipdir(folder_path, zipObj)  
