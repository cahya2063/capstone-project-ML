from keras.models import model_from_json

def load_model_from_files(json_path, weights_path):
    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

