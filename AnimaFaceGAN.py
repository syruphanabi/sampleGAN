from glob import glob
import PIL

import matplotlib.pyplot as plt
import numpy as np

import os
from time import time

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import keras.backend as K


class AnimaFaceGAN:
    def __init__(self):
        
        self.img_rows = 96
        self.img_cols = 96
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 200
        
        self.optimizer = Adam(0.0002, 0.5)
        self.discriminator = None
        self.generator = None
        self.combined = None

        self.start = None

        self.data_iter = None

        if not os.path.exists('af_weight'):
            os.makedirs('af_weight')

        if not os.path.exists('af_result'):
            os.makedirs('af_result')

        self.current_epoch = 0

    def build_model(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=self.optimizer,
        )

    def build_generator(self):

        model = Sequential()
        
        # model.add(Dense(128 * 7 * 7, activation='relu', input_dim=self.latent_dim))
        # model.add(Reshape((7, 7, 128)))
        #
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=3, padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))
        #
        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding='same'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Activation('relu'))

        model.add(Dense(128 * 6 * 6, activation='relu', input_dim=self.latent_dim))
        model.add(Reshape((6, 6, 128)))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=5, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        model.add(UpSampling2D())
        model.add(Conv2D(16, kernel_size=3, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        
        model.add(Conv2D(self.channels, kernel_size=1, padding='same'))
        model.add(Activation('tanh'))
        
        model.summary()
        
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)

    def build_discriminator(self):
        
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
        # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        
        model.summary()
        
        img = Input(shape=self.img_shape)
        validity = model(img)
        
        return Model(img, validity)
    
    def train(self, epochs, batch_size=128, save_interval=50):

        self.start = time()

        # (X_train, _), _ = mnist.load_data()
        #
        # X_train = X_train / 127.5 - 1
        # X_train = np.expand_dims(X_train, axis=3)

        self.prepare_data()
        X_train = self.data_iter.next()
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(self.current_epoch, epochs):
            
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # imgs = self.data_iter.next()
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            g_loss = self.combined.train_on_batch(noise, valid)

            if (epoch + 1) % save_interval == 0:
                print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.save_imgs(epoch)
                self.generator.save_weights('af_weight/g_%d.h5' % epoch)
                self.discriminator.save_weights('af_weight/d_%d.h5' % epoch)
                self.combined.save_weights('af_weight/c_%d.h5' % epoch)

            # if epoch == 900 or epoch == 1300 or epoch == 1700:
            #     lr = K.get_value(self.optimizer.lr)
            #     K.set_value(self.optimizer.lr, lr * 0.3)
            #     print("lr changed to {}".format(K.get_value(self.optimizer.lr)))

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :], cmap='brg')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig('af_result/af_%d.png' % epoch)
        plt.close()

        print((time() - self.start) / (epoch - self.current_epoch + 1))

    def prepare_data(self):
        file_list = glob('faces/*.jpg')
        num_data = len(file_list)
        # print(num_data)
        data = np.zeros((num_data, 96, 96, 3))

        for i in range(num_data):
            x = load_img(file_list[i])
            data[i, :, :, :] = img_to_array(x, data_format='channels_last')

        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            horizontal_flip=True
        )

        # data = data.reshape(data.shape[0],1,96,96,3)
        # print(data.shape)
        datagen.fit(data)

        self.data_iter = datagen.flow(data, batch_size=num_data)

        # x_batch = self.data_iter.next()

        # plt.figure(num=2, figsize=(3, 3), dpi=100)
        # plt.subplot(1, 2, 1)
        # plt.imshow(x_batch[0], cmap='brg')
        # print(np.mean(x_batch), np.max(x_batch), np.min(x_batch))
        # plt.subplot(1, 2, 2)
        # plt.imshow(data[1]/255., cmap='brg')
        # plt.show()
        # plt.close()
        # print(x_batch.shape)

    def restore(self, num):
        if os.path.exists('weight_tmp/c_' + str(num) + '.h5'):
            if os.path.exists('weight_tmp/d_' + str(num) + '.h5'):
                if os.path.exists('weight_tmp/g_' + str(num) + '.h5'):
                    self.generator.load_weights('weight_tmp/g_' + str(num) + '.h5')
                    self.discriminator.load_weights('weight_tmp/d_' + str(num) + '.h5')
                    # self.combined.load_weights('weight_tmp/c_' + str(num) + '.h5')
                    self.current_epoch = num + 1
                    print('weight after ' + str(num) + ' epoch is loaded.')


if __name__ == '__main__':
    dcgan = AnimaFaceGAN()
    # dcgan.build_generator()
    # dcgan.build_discriminator()
    # dcgan.prepare_data(8)

    dcgan.build_model()
    dcgan.restore(599)
    lr = K.get_value(dcgan.optimizer.lr)
    K.set_value(dcgan.optimizer.lr, lr * 0.1)
    print("lr changed to {}".format(K.get_value(dcgan.optimizer.lr)))
    dcgan.train(epochs=900, batch_size=32, save_interval=50)
