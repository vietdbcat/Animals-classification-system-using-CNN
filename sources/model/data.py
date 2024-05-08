import pandas as pd
import os

class Data:
    def __init__(self, image_dir):
        filenames = os.listdir(image_dir)
        labels = [x.split(".")[0] for x in filenames]
        self.data = pd.DataFrame({"filename": filenames, "label": labels})
        self.data = self.data[self.data["filename"].str.endswith("jpg")]
        
# data = Data("data/train/")       
# print(data.data)