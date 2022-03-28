import shutil
import os


res_dir = "code_test"
os.makedirs(res_dir, exist_ok=True)

shutil.copytree('./', res_dir, ignore=shutil.ignore_patterns('results'), dirs_exist_ok=True)


