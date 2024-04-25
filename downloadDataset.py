# run this command huggingface-cli download wiki_dpr data/psgs_w100/nq/train-00002-of-00157.parquet --local-dir="D:/Boody/GP/wikidpr" --repo-type dataset and make it loop from 00003 to 00157

import os

for i in range(99, 157):
    os.system(f"huggingface-cli download wiki_dpr data/psgs_w100/nq/train-{str(i).zfill(5)}-of-00157.parquet --local-dir=\"D:/Boody/GP/wikidpr\" --repo-type dataset")
    print(f"Downloaded {i} out of 157")
    print("----------------------------------------------------------")