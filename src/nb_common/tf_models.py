def build_lstm(seq_len, n_feats, n_classes, hidden=64, dropout=0.3):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        raise
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_feats)),
        keras.layers.LSTM(hidden),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(max(32, hidden//2), activation='relu'),
        keras.layers.Dropout(max(0.1, dropout/2)),
        keras.layers.Dense(n_classes, activation='softmax')
    ])
    return model


def build_cnn1d(seq_len, n_feats, n_classes, filters=64, kernel_size=3, dropout=0.2):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inp)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs=inp, outputs=out)
    return model


def build_cnn_bilstm(seq_len, n_feats, n_classes, filters=64, kernel_size=3, hidden=64, dropout=0.3):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = keras.layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inp)
    x = keras.layers.Bidirectional(keras.layers.LSTM(hidden, return_sequences=False))(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs=inp, outputs=out)


def build_transformer_tiny(seq_len, n_feats, n_classes, head_size=32, num_heads=2, ff_dim=64, num_transformer_blocks=1, dropout=0.1):
    try:
        import tensorflow as tf
        from tensorflow import keras
    except Exception:
        raise
    inp = keras.layers.Input(shape=(seq_len, n_feats))
    x = inp
    for _ in range(num_transformer_blocks):
        attn = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
        x = keras.layers.Add()([x, attn])
        x = keras.layers.LayerNormalization()(x)
        ff = keras.layers.Dense(ff_dim, activation='relu')(x)
        x = keras.layers.Add()([x, ff])
        x = keras.layers.LayerNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inputs=inp, outputs=out)


def export_tflite(model, out_path):
    try:
        import tensorflow as tf
    except Exception:
        raise
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        fh.write(tflite_model)
    return out_path
