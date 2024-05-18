import os
import pandas as pd

# ========================== WikiDPR Download ==========================
# for i in range(157):
#     os.system(f"huggingface-cli download wiki_dpr data/psgs_w100/nq/train-{str(i).zfill(5)}-of-00157.parquet --local-dir=\"D:/Boody/GP/wikidpr\" --repo-type dataset")
#     print(f"Downloaded {i} out of 157")
#     print("----------------------------------------------------------")
    

# ========================== WitBase Download ==========================
# cnt = 4141523
# for i in range(211, 330):
#     os.system(f"huggingface-cli download wikimedia/wit_base data/train-{str(i).zfill(5)}-of-00330.parquet --local-dir=\"/home/azureuser/RagNRoll\" --repo-type dataset")
#     print(f"Downloaded {i} out of 330")
#     print("----------------------------------------------------------")

#     df = pd.read_parquet(f"/home/azureuser/RagNRoll/data/train-{str(i).zfill(5)}-of-00330.parquet")
#     df = df[["image_url", "embedding", "caption_attribution_description"]]
#     # add id column 
#     df["id"] = range(cnt, cnt + len(df))
#     cnt += len(df)

#     # save the dataframe to a parquet file
#     df.to_parquet(f"/home/azureuser/RagNRoll/dataClean/train-{str(i).zfill(5)}-of-00330.parquet")
#     os.system(f"rm data/train-{str(i).zfill(5)}-of-00330.parquet")


# ========================== WitBase GDrive ==========================
import io
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# Set up your service account credentials (replace with your own)
credential_json = {
    # Your service account credentials here
    "type": "service_account",
    "project_id": "adbindex",
    "private_key_id": "518683087cafb9121ec58e7e08ab09c05a543763",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC9VNWbLkZvJGSN\nsK+k3nXTYTep5K2AZfVxiSrwXlztVJqnRpast8kKSuI2Y49eN3SHza1256dOAoUQ\nMUOacu00varTbv9kiYUEY5/IzGuu2W5f5qfNLoXM0AYHYHtzTo8OJSlAbmacTv98\n7hoK5Xg1qzWguQ138wagCFfXWTN1qM4mPqPM2x8aY3sccACfTRIvPZFu7nAK+3jJ\n5DhmlmoROjj7n/3wxwM/wvKEaBf0pUzHCmTY/ZU+DGE8HCwXJQx3bbV+sUu9DWh7\neFRLiN2y10aoDAnCzcT4i3pKsP6Gt9ZXC5RNnQiYB3FN1LrIfXqMmBaj7jWPf/IR\ngfSHRE0rAgMBAAECggEAH2+SBfGRhUusHHCNQVkkhCPcq6tJ+Ys0TRUi2YU5dbh5\n8zM/uk7LpRdY5wHx4IlholjOp/L/ru6xLEaZvG+/neWuEuwXoJhKyzdFZQA4EvCM\nKIUyMFm8ooyZq6jj5nnRO96EsmuCQwrE1ffswAlow7R5M4X2TAfAflrURu/LIQpM\nE4ldhp+SdrQluaw7OPWsDVaijyNTZCEB8iaB9LefWX2tzx3rxwUbtsemifR15k5G\nUHV/jTAHGID7ECY7vHf9smvrP3WPz+UDOx/gxILPql02uqGrmc8oBhx+GoERA1aI\nGLJB8PvjkpOQO1mLegmvtG0wBXjlMAighpMvwEPMJQKBgQD3GMimA5AJmeaUUYJd\n3qWx5qfe1KWsIKrpTBsw3LI6/jhdGEPX9F22b+wzU/rJG5/fsMP+p0nT7uiYBD3O\np8Wk8VAhaaWdb8Ff8VRQirEnbxUdjuvt3ADjpkLvrghYJ9yxkD4OgRW7BzLB42AL\nd5/h0bBGrIcRqnodTDzmF1HxXwKBgQDEJzkywXfyQ9W26M65joKgwbp1r99I9/ZV\n/VtlHtVrbY8OUj7BQpSwRXkO4a9WbbzeXCn0sBI0vVKjBTu7NKJeI5XRE6B0N4QR\n2k20oCm1UcttewgtatK39MQ2papR4Oha6qwBoo/UdC867ZWLD2Nxakuuo0l7Uczl\ney5MhiR7tQKBgQDDDgdmNa3ARoLkViuJGjbGZoPhvQ9C/06rd26HddAkDYZExuDX\nWomXeGaGZia0FmBv3kP5g+kdqZxmVALOXVYBzJQqrBEWZJ/Lst+R7MnHjUKn4KiR\nTXhOPHLebQ0dOepXKLw0CuYyi44fy/OHWdkWE8cJIyxEX8Sh2ArCv2nqFQKBgB/F\nVp9g9MOZtyjsJmeprIDLQB9FwD26Y/zjj4UebGT9Ftmz+pQk655tckE1zseJ+Lhv\nZyBJ0HkYXSUoeNdGnDHxQ5fcvPV19H9Lw6BI/NhwiimObvGkRsMi8xEC3kZqzlfD\ngN627OL1epzp1Hn0oR/CnsWHjyRZSH226PXeGFStAoGASwvizX6NPLK4jI+MmN/z\nRQePvbmQCSPtDHzP03YY3qVVHUQNDn5bZsW+0vkWI1Vg8PhhWwJl0iIa/vsLinkv\n3JoYg20QuHxBPwOWHw4iiQ7NgMyb5HGUw7QxQWWpCl+WG+z0muqdOTPUr+9ni4gf\nf1AJDVjGdN8apHDjif8xcZs=\n-----END PRIVATE KEY-----\n",
    "client_email": "adbindex@adbindex.iam.gserviceaccount.com",
    "client_id": "115766601135290386187",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/adbindex%40adbindex.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Create a service account and use its JSON content
credentials = service_account.Credentials.from_service_account_info(credential_json)

# Initialize the Drive API
drive_service = build('drive', 'v3', credentials=credentials)

# Specify the folder ID (replace with your folder ID)
folder_id = '1CiB8DMxBYQxBod5PSjtPJz4zcHatbOvV'

# List files in the folder
results = drive_service.files().list(q=f"'{folder_id}' in parents").execute()
files = results.get('files', [])
print(files)
# Download each file
# make an array if strings containing all the filenames from "train-00292-of-00330.parquet" to "train-00329-of-00330.parquet"
doneFiles = [f"train-{str(i).zfill(5)}-of-00330.parquet" for i in range(311, 330)]
for file in files:
    # check if file['name'] is in doneFiles then skip
    if file['name'] in doneFiles:
        print(file['name'], "already downloaded")
        continue
    request = drive_service.files().get_media(fileId=file['id'])
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()

    # Save the file locally (you can customize this part)
    with open("D:/Boody/GP/witbase/"+file['name'], 'wb') as local_file:
        local_file.write(fh.getvalue())
    print(f"Downloaded {file['name']}")
