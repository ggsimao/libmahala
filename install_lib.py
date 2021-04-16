import sys
import shutil

for dir in sys.path:
    if os.path.isfile(dir+"/libmahalapy.so"):
        print("-- Up-to-date: "+dir+"/libmahalapy.so")
    else:
        shutil.copy('./libmahalapy.so', dir)
        print("-- Installing: "+dir+"/libmahalapy.so")