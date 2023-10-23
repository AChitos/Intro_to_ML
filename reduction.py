import pandas as pd
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE, MDS
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

# Authors: Oskar Gryglewski (s5133122), Guillermo Quintanilla (s5191270), Andreas Hitos (s4807642)

# Loading the data
breast = load_breast_cancer()

# Convert the dataset to a pandas dataframe
df_breast = pd.DataFrame(breast.data, columns=breast.feature_names)
breast_target = pd.DataFrame(breast.target)
data = df_breast

# Data visualization:
#df_breast.hist(bins=50, figsize=(20, 15))
#plt.show()

# Data scaling
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# T-distributed Stochastic Neighbor Embedding
# Chosen number of dimensions -> 2
breast_embedded_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=40).fit_transform(normalized_data)

# Scatter plot of the t-SNE results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(breast_embedded_tsne[:, 0], breast_embedded_tsne[:, 1], c=breast.target, cmap='viridis')
plt.colorbar(scatter, label='Target')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE Visualization')
plt.show()

# Multidimensional scaling
breast_embedded_mds = MDS(n_components=2, normalized_stress='auto').fit_transform(normalized_data)

# Scatter plot of the MDS results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(breast_embedded_mds[:, 0], breast_embedded_mds[:, 1], c=breast.target, cmap='viridis')
plt.colorbar(scatter, label='Target')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('MDS Visualization')
plt.show()


# Number of dimentions
encoding_dim = 2

data_1 = keras.Input(shape=(30,))

# Encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(data_1)
# Decoded reconstruction of the input
decoded = layers.Dense(30, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(data_1, decoded)
# This model maps an input to its encoded representation
encoder = keras.Model(data_1, encoded)

# Encoded 2-dimensional input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Early Stopping
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

# Stops after 3 unsuccesful repetitions
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

# train the data
x_train, x_test, _, _ = train_test_split(normalized_data, breast_target, test_size=0.2)

# training for 50 epochs
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=50,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)

# Scatter plot of the autoencoder results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(encoded[:, 0], encoded[:, 1], c=breast.target[:encoded.shape[0]], cmap='viridis')
plt.colorbar(scatter, label='Target')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Autoencoder Visualization')
plt.show()


