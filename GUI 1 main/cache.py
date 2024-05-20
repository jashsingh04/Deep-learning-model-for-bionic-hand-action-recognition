def quantization_aware():
    quantize_model = tfmot.quantization.keras.quantize_model
    model=trained_model
    # Annotate the layers you want to skip quantization
    skip_quantization_layers = [tf.keras.layers.Conv1D]

    for layer in model.layers:
        if isinstance(layer, tuple(skip_quantization_layers)):
            # Skip quantization for Conv1D layer
            continue
        # Otherwise, annotate the layer for quantization
        layer = tfmot.quantization.keras.quantize_annotate_layer(layer)

    # Compile the quantization-aware model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model