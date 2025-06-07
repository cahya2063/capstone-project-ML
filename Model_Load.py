from keras.models import model_from_json

def load_model_from_files(json_path: str, weights_path: str):
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    return model
