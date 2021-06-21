import tensorflow as tf
import numpy as np
import horovod.tensorflow.keras as hvd
from datetime import datetime
import os 

# Initialize Horovod
hvd.init()

# pin GPU to the worker
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if len(gpus) > 0:
    print(len(gpus), hvd.local_rank)
    gpu = gpus[hvd.local_rank()]
    tf.config.experimental.set_visible_devices(gpu, 'GPU')
    if hvd.local_rank() < len(gpus):
        gpu = gpus[hvd.local_rank()]
        tf.config.experimental.set_visible_devices(gpu, 'GPU')
    """
os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
dim = x_train.shape[1]

backbone = tf.keras.applications.ResNet50(include_top=False, input_shape=(dim,dim, 3), classes=10)
model_advanced = tf.keras.models.Sequential()
model_advanced.add(backbone)
model_advanced.add(tf.keras.layers.Flatten())
model_advanced.add(tf.keras.layers.Dense(128, activation='relu'))
model_advanced.add(tf.keras.layers.Dense(10, activation='softmax'))

# Horovod: add Horovod DistributedOptimizer.
opt = tf.optimizers.Adam(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model_advanced.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              experimental_run_tf_function=False)
print(model_advanced.summary())

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

t_start = datetime.now()

history = model_advanced.fit(x_train, y_train, 
        epochs=12 // hvd.size(), batch_size=512, 
        validation_data=(x_test, y_test), 
        callbacks=callbacks,
        verbose=1 if hvd.rank() == 0 else 0)
t_end = datetime.now()
print('Elapsed time:', t_end - t_start)
print(model_advanced.evaluate(x_test,  y_test, verbose=2))

