import json
import numpy as np


class Data:
    def __init__(self, path):
        with open(path, "r") as fin:
            self.rawDict = json.load(fin)

        print(self.rawDict)

if __name__ == "__main__":
    data = Data("2016-06-01-2017-06-01all-factors.json")