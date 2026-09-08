import tensorflow as tf
import numpy as np

# Function that creates the model in the paper. This is the baseline used for all the metrics.
def separable_resnet(input_shape, num_classes, bias = False, y_train = [], reg_drop = False, lstm = False, PSD = False, blocks = 3, width_mult = 1):
    # Input tensor shape
    inputs = tf.keras.layers.Input(shape=input_shape)
    kernel_sizes = [25,25,25, 25]
    # Initial convolution layer
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 15), strides=(1, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Max pooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    filters = [32, 64]
    filters = [int(f * width_mult) for f in filters]

    # Residual blocks
    for i in range(blocks): 
        # Separable convolution layer 1
        # filters is dimension of output space, strides=convolution length
        residual = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)    #orig value = 32
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, kernel_sizes[i]), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)

        # modification of x
        # Shortcut connection / conv block
        if i == 0:
            x = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)

        # rest of the identity blocks
        x = tf.keras.layers.Add()([x, residual])
        
        x = tf.keras.layers.Activation('relu')(x)
        if reg_drop:
            x = tf.keras.layers.Dropout(0.1)(x)

    # Global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    if lstm:
        x = tf.keras.layers.Reshape((64,1))(x)
        x =  tf.keras.layers.LSTM(32)(x)
    if PSD:
        inputs2 = tf.keras.layers.Input(129)
        x2 = tf.keras.layers.Dense(32, 'relu')(inputs2)
        x = tf.keras.layers.concatenate([x2,x])
    x = tf.keras.layers.Dense(32, 'sigmoid')(x) # change to sigmoid
    # Fully connected layer
    if bias:
        x = tf.keras.layers.Dense(units=num_classes, activation='softmax', bias_initializer = tf.keras.initializers.Constant(init_bias(y_train)))(x)
    else:
        x = tf.keras.layers.Dense(units=num_classes, activation='softmax')(x)

    # Define the model
    if PSD:
        model = tf.keras.Model(inputs=[inputs,inputs2], outputs=x)
    else:
        model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

def physio_separable_resnet(input_shape, num_classes, bias = False, y_train = [], reg_drop = False, lstm = False, PSD = False, blocks = 3, width_mult = 1):
    # original paper: We addressed this by not quantizing the start block and the identity block, and this improved performance (indicated by **).
     # Input tensor shape
    inputs = tf.keras.layers.Input(shape=input_shape)
    kernel_sizes = [25,25,25]

    # Initial convolution layer
    # start block
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 15), strides=(1, 2), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Max pooling layer
    x = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    filters = [32, 64]
    filters = [int(f * width_mult) for f in filters]

    # first identity block not quantized
    # Separable convolution layer 1
    residual = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)    #orig value = 32
    residual = tf.keras.layers.BatchNormalization()(residual)
    residual = tf.keras.layers.Activation('relu')(residual)
    
    residual = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, kernel_sizes[0]), strides=(1, 1), padding='same', use_bias=False)(residual)
    residual = tf.keras.layers.BatchNormalization()(residual)
    residual = tf.keras.layers.Activation('relu')(residual)
    
    residual = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
    residual = tf.keras.layers.BatchNormalization()(residual)
    
   
    shortcut = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
    x = tf.keras.layers.Add()([shortcut, residual])
    
    x = tf.keras.layers.Activation('relu', name='fp32_output')(x)
    if reg_drop:
        x = tf.keras.layers.Dropout(0.1)(x)
    fp32_output_shape = x.shape[1:] 
    fp32_submodel = tf.keras.Model(inputs=inputs, outputs=x, name='fp32_submodel')

    # quantized parts, other blocks
    quant_input = tf.keras.layers.Input(shape=fp32_output_shape)
    y = quant_input
    for i in range(1, blocks):
        residual = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(y)    #orig value = 32
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, kernel_sizes[i]), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        residual = tf.keras.layers.Activation('relu')(residual)
        
        residual = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False)(residual)
        residual = tf.keras.layers.BatchNormalization()(residual)
        
        y = tf.keras.layers.Add()([y, residual])
        y = tf.keras.layers.Activation('relu')(y)
        if reg_drop:
            y = tf.keras.layers.Dropout(0.1)(y)
    
    # Global average pooling layer
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    if lstm:
        y = tf.keras.layers.Reshape((64,1))(y)
        y =  tf.keras.layers.LSTM(32)(y)
    if PSD:
        inputs2 = tf.keras.layers.Input(129)
        y2 = tf.keras.layers.Dense(32, 'relu')(inputs2)
        y = tf.keras.layers.concatenate([y2, y])

    y = tf.keras.layers.Dense(32, 'sigmoid')(y) # change to sigmoid
    # Fully connected layer
    if bias:
        y = tf.keras.layers.Dense(units=num_classes, activation='softmax', bias_initializer = tf.keras.initializers.Constant(init_bias(y_train)))(y)
    else:
        y = tf.keras.layers.Dense(units=num_classes, activation='softmax')(y)
    
    quant_inputs = [quant_input, inputs2] if PSD else quant_input
    quant_submodel = tf.keras.Model(inputs=quant_inputs, outputs=y, name="quant_body")
    return fp32_submodel, quant_submodel

# Sequence learner model, based on an LSTM layer.
# drop to MLP 
def seq_model(input_shape): 
    inp = tf.keras.layers.Input((input_shape,1)) # shape is seq_len*classes, 1
    # x = tf.keras.layers.LSTM(32)(inp)
    x = tf.keras.layers.Flatten()(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32,'relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(5, 'softmax')(x)
    model = tf.keras.models.Model(inputs = inp, outputs = out)
    return model
