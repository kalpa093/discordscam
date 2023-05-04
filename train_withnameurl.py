import transformer

# Define the parameters for the model
embedding_dim = 64
num_heads = 8
dff = 128
learning_rate = 5e-6
number_of_classes = 2
# Define the input layers for each text source
input_layer_1 = tf.keras.layers.Input(shape=(maxlen,))
input_layer_2 = tf.keras.layers.Input(shape=(maxlen_2,))
input_layer_3 = tf.keras.layers.Input(shape=(maxlen_3,))
# Define the embedding layer for each text source
embedding_layer_1 = TokenAndPositionEmbedding(maxlen, max_features, embedding_dim)
embedding_layer_2 = TokenAndPositionEmbedding(maxlen_2, max_features_2, embedding_dim)
embedding_layer_3 = TokenAndPositionEmbedding(maxlen_3, max_features_3, embedding_dim)
# Embed each input layer using the corresponding embedding layer
embedded_1 = embedding_layer_1(input_layer_1)
embedded_2 = embedding_layer_2(input_layer_2)
embedded_3 = embedding_layer_3(input_layer_3)
# Concatenate the embedded inputs from each text source
concatenated = tf.keras.layers.Concatenate(axis=1)([embedded_1, embedded_2, embedded_3])
# Apply the transformer block to the concatenated embeddings
transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
transformed = transformer_block(concatenated)
# Pool over the time dimension
pooled = tf.keras.layers.GlobalAveragePooling1D()(transformed)
# Apply dropout and a dense layer
dropout_1 = tf.keras.layers.Dropout(learning_rate)(pooled)
dense_1 = tf.keras.layers.Dense(64, activation="relu")(dropout_1)
dropout_2 = tf.keras.layers.Dropout(learning_rate)(dense_1)
# Output layer
output_layer = tf.keras.layers.Dense(number_of_classes, activation="softmax")(dropout_2)
# Define the model
model = tf.keras.Model(inputs=[input_layer_1, input_layer_2, input_layer_3], outputs=output_layer)
# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Fit the model
print("Train chatting data with user name and url")
model.fit(x=[train_X, train_X_name, train_X_url], y=train_y, batch_size=32, epochs=3, validation_data=([val_X, val_X_name, val_X_url], val_y))
pred = model.predict([test_X, test_X_name, test_X_url])
pred_1d=pred[:,1].flatten()
pred_class = np.where(pred_1d>0.5,1,0)
get_clf_eval(test_y, pred_class)
