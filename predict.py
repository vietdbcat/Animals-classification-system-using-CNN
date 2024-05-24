from sources.model import cnn, data, utils
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

test_dir = "data/test/"
filenames = os.listdir(test_dir)
df = pd.DataFrame({"filename": filenames})
df['label'] = 'unknown'

gen = utils.DataGenerator(image_size=128, batch_size=32)
test_generator = gen.test_generating(df, "data/test/")

model = cnn.CNNModel(image_size=128, image_channel=3)
model.load_weights("new_w.keras")

classifior = model.predict(test_generator, verbose = 0)
print(classifior)

label_0_1 = np.argmax(classifior, axis=1)

y_test_pred = label_0_1

df['label'] = y_test_pred

# mapping
label_mapping = {0: 'cat', 1: 'dog'}
df['label'] = df['label'].map(label_mapping)
df.head()

fig, axes = plt.subplots(1, 10, figsize=(20, 4))
for idx in range(10):
    image_path = os.path.join(test_dir, df.iloc[idx]['filename'])
    image = Image.open(image_path)
    axes[idx].imshow(image)
    axes[idx].set_title("Label: " + df.iloc[idx]['label'])
    axes[idx].axis('off')
plt.show()