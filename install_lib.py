import sys
import shutil

for dir in sys.path:
    shutil.copy('./libmahalapy.so', dir)