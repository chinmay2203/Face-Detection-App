import tensorflow as tf

model = tf.keras.models.load_model('face_model.h5')

print(model.summary())