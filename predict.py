from sources.model import cnn, data, utils
import pandas as pd
import os

test_dir = "data/test/"
filenames = os.listdir(test_dir)
df = pd.DataFrame({"filename": filenames})
df['label'] = 'unknown'

gen = utils.DataGenerator(image_size=128, batch_size=32)
test_generator = gen.generating(df, "data/test/")

model = cnn.CNNModel(image_size=128, image_channel=3)
model.load_weights("w.keras")

classifior = model.predict(test_generator, verbose = 0)
print(classifior)