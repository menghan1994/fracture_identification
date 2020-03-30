
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Dropout
from keras import initializers

def model_generate(training_X):
    input_node = len(training_X[0,:])
    output_node = 2
    #[128,128,128]
    layer_node = [256,256,256]

    model = tf.keras.Sequential()

    model.add(layers.Dense(layer_node[0], activation = 'relu',input_shape=(input_node,),
                           kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)))
    model.add(keras.layers.Dropout(0.4))
    
    for n in layer_node:
        model.add(keras.layers.Dense(n,activation = 'relu', kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)))
        model.add(keras.layers.Dropout(0.4))
        
    model.add(keras.layers.Dense(output_node, activation=tf.nn.softmax))
    
    model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #model.summary()
   
    return model
