from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator:
    def __init__(self, image_size, batch_size):
        self.image_size = image_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.1,
            fill_mode='reflect',
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
    def generating(self, data, directory):
        return self.datagen.flow_from_dataframe(
            dataframe=data,
            directory=directory,
            x_col='filename',
            y_col='label',
            batch_size=self.batch_size,
            target_size=(self.image_size, self.image_size)
        )

# from data import Data
# data = Data("data/train/")       
        
# gen = DataGenerator(128, 32)
# train = gen.generating(data=data.data, directory="data/train/")
# print(train)