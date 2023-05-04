import transformer

embedding_dim = 64  # 각 단어의 임베딩 벡터의 차원
num_heads = 8  # 어텐션 헤드의 수
dff = 256  # 포지션 와이즈 피드 포워드 신경망의 은닉층의 크기
learning_rate = 5e-6  # 러닝 레이트
number_of_classes = 2 # 분류할 클래스 수

inputs = tf.keras.layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, max_features, embedding_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embedding_dim, num_heads, dff)
x = transformer_block(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dropout(learning_rate)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(learning_rate)(x)
outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

print("Train chatting data")
model.fit(train_X, train_y, batch_size=32, epochs=3, validation_data=(val_X, val_y))
pred = model.predict(test_X)
pred_1d=pred[:,1].flatten()
pred_class = np.where(pred_1d>0.5,1,0)
get_clf_eval(test_y, pred_class)
