from tensorflow import keras
from keras import layers, models

def conv(input, filters):
    x = layers.Conv2D(filters, kernel_size = 3, padding = 'same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def conv_block(input, filters):
    x = conv(input, filters)
    x = conv(x, filters)

    return x

def down_block(input, filters):
    skip = conv_block(input, filters)
    down = layers.MaxPooling2D((2, 2))(skip)

    return down, skip

def up_block(input, skip, filters):
    x = layers.Conv2DTranspose(filters, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    ## Skip Connection
    x = layers.Concatenate()([x, skip]) ## INVERTER AQUI A ORDEM?

    x = conv_block(x, filters)

    return x

def contracting_path(input):
    down1, skip1 = down_block(input, 64)
    down2, skip2 = down_block(down1, 128)
    down3, skip3 = down_block(down2, 256)
    down4, skip4 = down_block(down3, 512)

    return down4, skip1, skip2, skip3, skip4

def bottleneck(input):
    x = conv_block(input, 1024)

    return x

def expansive_path(input, skip1, skip2, skip3, skip4):
    input = up_block(input, skip4, 512)
    input = up_block(input, skip3, 256)
    input = up_block(input, skip2, 128)
    input = up_block(input, skip1, 64)

    return input

def fbpconvnet_model(input_shape = (512, 512, 1)) -> models.Model:
    input = layers.Input(input_shape)

    x = conv_block(input, 64)

    x, skip1, skip2, skip3, skip4 = contracting_path(x)

    x = bottleneck(x)

    x = expansive_path(x, skip1, skip2, skip3, skip4)

    output = layers.Conv2D(1, 1, activation = 'sigmoid')(x)

    output = layers.Add()([output, input]) ## FINAL SKIP CONNECTION

    # output = layers.Activation("sigmoid")(output) ## THIS ONE
    # output = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(output) ## OR THIS ONE

    model = models.Model(input, output, name = 'FBPConvNet')
    return model