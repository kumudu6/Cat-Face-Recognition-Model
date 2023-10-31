import os
import random
import glob
import shutil

os.chdir(r'C:\Cat Face Recognition\Data Set')
if os.path.isdir('train/dogs') is False:
    os.makedirs('train/dogs')
    os.makedirs('train/cats')
    os.makedirs('valid/dogs')
    os.makedirs('valid/cats')
    os.makedirs('test/dogs')
    os.makedirs('test/cats')

    for c in random.sample(glob.glob('dog*'),500):
        shutil.move(c,'train/dogs')
    for c in random.sample(glob.glob('cat*'),500):
        shutil.move(c,'train/cats')
    for c in random.sample(glob.glob('dog*'), 100):
        shutil.move(c, 'valid/dogs')
    for c in random.sample(glob.glob('cat*'),100):
        shutil.move(c,'valid/cats')
    for c in random.sample(glob.glob('dog*'),50):
        shutil.move(c,'test/dogs')
    for c in random.sample(glob.glob('cat*'),50):
        shutil.move(c,'test/cats')