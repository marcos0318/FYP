import os 
import json
import sys

file_list = sys.argv[1:]

for file_nama in file_list:
    with open(file_nama, "r") as fin:
        js = json.load(fin)
    with open(file_nama +".csv", "w") as fout:
        fout.write("id, x, y \n")
        for key, value in js.items():
            fout.write(key + ", %.3f, %.3f \n" % ( value["x"], value["y"]))



