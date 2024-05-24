from sources.model import cnn, data, utils
from sklearn.model_selection import train_test_split
import json

training_dir = "data/train/"
dt = data.Data(training_dir)
df = dt.data

labels = df['label']
X_train, X_temp = train_test_split(df, test_size=0.2, stratify=labels, random_state = 42)
# label_test_val = X_temp['label']
# X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 42)

gen = utils.DataGenerator(image_size=128, batch_size=32)
train_generator = gen.train_generating(X_train, "data/train/")
# val_generator = gen.test_generating(X_val, "data/train/")
val_generator = gen.test_generating(X_temp, "data/train/")

model = cnn.CNNModel(image_size=128, image_channel=3)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_generator,
                    validation_data = val_generator, 
                    callbacks=[model.early_stoping,model.learning_rate_reduction],
                    epochs = 30
                   )

with open("new_history.json", "w") as j:
    json.dump(model.history.history, j)

model.save("new_w.keras")
model.save("w.h5")