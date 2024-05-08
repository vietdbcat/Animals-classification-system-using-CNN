import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

class CNNModel(tf.keras.Model):
    def __init__(self, image_size, image_channel):
        super(CNNModel, self).__init__()
        
        """
            Layers
            
        """
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, image_channel))
        self.batchnorm1 = BatchNormalization()
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.2)
        
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.batchnorm2 = BatchNormalization()
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = Dropout(0.2)
        
        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.batchnorm3 = BatchNormalization()
        self.maxpool3 = MaxPooling2D(pool_size=(2, 2))
        self.dropout3 = Dropout(0.2)
        
        self.conv4 = Conv2D(256, (3, 3), activation='relu')
        self.batchnorm4 = BatchNormalization()
        self.maxpool4 = MaxPooling2D(pool_size=(2, 2))
        self.dropout4 = Dropout(0.2)
        
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.batchnorm5 = BatchNormalization()
        self.dropout5 = Dropout(0.2)
        
        self.output_layer = Dense(2, activation='softmax')
        
        
        """
            Callbacks
            
        """
        self.learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=2,
                                            factor=0.5,
                                            min_lr = 0.00001,
                                            verbose = 1)

        self.early_stoping = EarlyStopping(monitor='val_loss',patience= 3,restore_best_weights=True,verbose=0)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm5(x)
        x = self.dropout5(x)
        
        output = self.output_layer(x)
        return output
