import sys
import os

for dir in sys.path:
    os.remove(dir + "/libmahalapy.so")