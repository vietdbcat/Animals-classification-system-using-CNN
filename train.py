from sources.model import cnn, data, utils
from sklearn.model_selection import train_test_split

training_dir = "data/train/"
dt = data.Data(training_dir)
df = dt.data

labels = df['label']
X_train, X_temp = train_test_split(df, test_size=0.2, stratify=labels, random_state = 42)
label_test_val = X_temp['label']
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 42)

gen = utils.DataGenerator(image_size=128, batch_size=32)
train_generator = gen.generating(X_train, "data/train/")
val_generator = gen.generating(X_val, "data/train/")
test_generator = gen.generating(X_test, "data/train/")

model = cnn.CNNModel(image_size=128, image_channel=3)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data = val_generator, 
                    callbacks=[model.early_stoping,model.learning_rate_reduction],
                    epochs = 30
                   )

model.save("model.h5")