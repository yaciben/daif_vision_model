import keras.optimizers
import tensorflow as tf

from keras import datasets
from keras import layers, models, losses, utils
from keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Dropout, Reshape, Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

import ssl

from tensorflow.python.keras.backend import arange

ssl._create_default_https_context = ssl._create_unverified_context

out_folder = './img_cnn_fountas_et_al'

def display(array1, array2, image_size=28, color_channels=1, save=True, epochs=0, fig_base_name=None, show=True):
    """Displays ten random images from each array.
    :param save:
    """
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    fig = plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(image_size, image_size, color_channels))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(image_size, image_size, color_channels))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig_name = f'{fig_base_name}-ep={epochs}'
    plt.title(fig_name)
    if save:
        plt.savefig(f'{out_folder}/{fig_name}.png', format='png')
    if show:
        plt.show()
    plt.close(fig=fig)

def plot_history(history, fig_base_name=None, epochs=None, show=True):
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0., 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.semilogy(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.title(fig_base_name)
    plt.savefig(f'./img_cnn_fountas_et_al/{fig_base_name}-history-ep={epochs}.png')
    if show:
        plt.show()
    plt.close(fig=fig)

def plot_images(images=None, class_names=None, train_labels=None, nb_images=25):
    colour_channels = images.shape[-1]
    plt.figure(figsize=(8, 8))
    for i in range(nb_images):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        if colour_channels == 1:
            plt.gray()
        if class_names is not None and train_labels is not None:
            plt.xlabel(class_names[np.argmax(train_labels[i])])
    plt.show()

def get_images(dataset_name, plot_images_after_load=False):
    is_image_normalized = False
    if dataset_name == 'cifar':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five',
                       'six', 'seven', 'eight', 'nine']
        # Pad to 32x32 as mnist images' size is 28x28
        train_images = tf.pad(tensor=train_images, paddings=[[0, 0], [2,2], [2,2]])
        test_images = tf.pad(tensor=test_images, paddings=[[0, 0], [2,2], [2,2]])
        assert train_images.shape[1] == 32
        assert test_images.shape[1] == 32
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)
    elif dataset_name == 'dSprites':
        dataset_zip = np.load('../../../../../../resources/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True,
                              encoding='latin1')
        imgs = dataset_zip['imgs']
        metadata = dataset_zip['metadata']
        # # Define number of values per latents and functions to convert to indices
        # latents_sizes = metadata[()]['latents_sizes']
        # latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
        #                                 np.array([1,])))
        #
        # def latent_to_index(latents):
        #     return np.dot(latents, latents_bases).astype(int)
        #
        # def select_latents(size=1):
        #     samples = np.zeros((size, latents_sizes.size))
        #     for lat_i, lat_size in enumerate(latents_sizes):
        #         samples[:, lat_i] = np.random.randint(lat_size, size=size)
        #
        #     return samples
        #
        # def filter_latents(filters=None):
        #     samples = np.zeros((0, latents_sizes.size))
        #     for lat_i, lat_size in enumerate(latents_sizes):
        #         # samples[:, lat_i] = np.random.randint(lat_size, size=size)
        #         arr = np.array(arange(lat_size))
        #         samples = np.append(samples, arr)
        #
        #     return samples
        #
        # # select images, latents: [color, shape, scale, orientation, posX, posY], latents_sizes: [1,  3,  6, 40, 32, 32]
        # selected_latents = select_latents(size=10) #  (filter={2:5, 3:0})
        # # selected_latents = filter_latents() #  (filter={2:5, 3:0})
        # selected_images_indices = latent_to_index(selected_latents)
        # selected_images = imgs[selected_images_indices]

        imgs = np.expand_dims(imgs, axis=-1)


        rng = np.random.default_rng(seed=42)
        rng.shuffle(imgs, axis=0)

        # truncate images for DEBUG
        # imgs = imgs[:5000]

        split_index = int(len(imgs) * 0.8)
        train_images, test_images = imgs[:split_index], imgs[split_index:]
        train_labels = None
        test_labels = None
        class_names = None
        #        train_images, test_images = train_images.astype('float32'), test_images.astype('float32')
        is_image_normalized=True

    print('train_images.shape', train_images.shape)
    print('test_images.shape', test_images.shape)
    # Normalize pixel values to be between 0 and 1
    if not is_image_normalized:
        train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0
    # train_images = (train_images - np.mean(train_images.astype('float32'))) / np.std(train_images.astype('float32'))
    # test_images = (test_images - np.mean(test_images.astype('float32'))) / np.std(test_images.astype('float32'))

    if train_labels is not None:
        train_labels = utils.to_categorical(train_labels, num_classes=10)
    if test_labels is not None:
        test_labels = utils.to_categorical(test_labels, num_classes=10)

    if plot_images_after_load:
        plot_images(images=train_images, class_names=class_names, train_labels=train_labels, nb_images=25)

    return train_images, test_images

def build_vision_model(colour_channels, image_size):
    # inner_activation = 'relu'
    inner_activation = 'gelu'
    outer_activation = 'sigmoid'

    assert image_size==32 or image_size==64

    # Fountas et al. qs_net (perception)
    last_strides = 2 if image_size==64 else 1
    padding='same'

    input = Input(shape=(image_size, image_size, colour_channels))
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(input)
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    encoded = Flatten()(x)

    x = Reshape((2*last_strides, 2*last_strides, -1))(encoded)
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding=padding, activation=inner_activation)(x)
    x = Conv2D(filters=colour_channels, kernel_size=3, padding=padding, activation=outer_activation)(x)

    model = Model(inputs=input, outputs=x)
    encoder = Model(inputs=input, outputs=encoded)
    return model, encoder


dataset_name = 'dSprites'
# dataset_name = 'cifar'
# dataset_name = 'mnist'
train_images, test_images = get_images(dataset_name=dataset_name, plot_images_after_load=False)

colour_channels = train_images.shape[-1]
image_size = train_images.shape[1]

model, encoder = build_vision_model(colour_channels=colour_channels, image_size=image_size)

model.summary()

encoder.summary()

# exit()

opt = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              #              loss='mse',
              metrics=['accuracy'],
              )

epochs = 2
model_type = f'{dataset_name}-train_std_conv_ae'
history = model.fit(train_images, train_images, epochs=epochs,
                    batch_size=64,
                    shuffle=True,
                    validation_split=0.2,
                    )


plot_history(history=history, fig_base_name=model_type, epochs=epochs, show=False)

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# print(test_acc)

model_weights_path = f'{out_folder}/{model_type}-ep={epochs}_weights.h5'
model.save_weights(model_weights_path)

model.load_weights(model_weights_path)

test_predictions = model.predict(test_images, batch_size=64)
display(test_images, test_predictions, image_size=image_size, color_channels=colour_channels, save=True,
        epochs=epochs,
        fig_base_name=f'test_{model_type}',
        show=False
        )

# Split in upstream batches due to keras GPU plugin bug
train_images_batches = np.array_split(train_images, indices_or_sections=4, axis=0)
predictions_batches = []
for tr in train_images_batches:
    print(tr.shape)
    predictions_batches.append(model.predict(tr, batch_size=64))

predictions = np.concatenate(predictions_batches)

display(train_images, predictions, image_size=image_size, color_channels=colour_channels, save=True,
        epochs=epochs,
        fig_base_name=f'train_{model_type}',
        show=False
        )
