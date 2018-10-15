def build_model(out_dims, input_shape=(224, 224, 3)):
    inputs_dim = Input(input_shape)
    x = Lambda(lambda x: x / 255.0)(inputs_dim) #在模型里进行归一化预处理
 
    x = InceptionV3(include_top=False,
                weights='imagenet',
                input_tensor=None,
                input_shape=(224, 224, 3),
                pooling=max)(x)
 
    squeeze = GlobalAveragePooling2D()(x)
 
    excitation = Dense(units=2048 // 11)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=2048)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, 2048))(excitation)
 
    scale = multiply([x, excitation])
 
    x = GlobalAveragePooling2D()(scale)
    dp_1 = Dropout(0.6)(x)
    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('sigmoid')(fc2) #此处注意，为sigmoid函数
    model = Model(inputs=inputs_dim, outputs=fc2)
    return model
