import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

class TumorPredictor:
    def __init__(self, model_path='model/my_model.h5'):
        model_path = os.path.join(os.getcwd(), model_path)
        self.model = load_model(model_path)

    def read_image(self, image_path):
        image = load_img(image_path, target_size=(299, 299))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path):
        image = self.read_image(image_path)
        prediction = self.model.predict(image)[0][0]
        # Print the prediction result
        if prediction < 0.5:
            result, result_message = "cancerous", f"Prediction: It's cancerous with a confidence of {1 - prediction:.2%}"
        else:
            result, result_message = "non_cancerous", f"Prediction: It's non_cancerous with a confidence of {prediction:.2%}"
        return result, result_message

if __name__ == "__main__":
    test_directory_path = r"C:\Users\Youssef\Downloads\New folder (3)\data set\Gastric Slice Dataset\test"
    cancer_directory_path = os.path.join(test_directory_path, "cancer")
    non_cancer_directory_path = os.path.join(test_directory_path, "non_cancer_subset00")

    predictor = TumorPredictor()

    true_labels = []
    predicted_labels = []

    for root, dirs, files in os.walk(cancer_directory_path):
        for file in files:
            image_path = os.path.join(root, file)
            true_label = "cancerous"
            predicted_label, _ = predictor.predict(image_path)

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            print(f"Predicted: {predicted_label}, True: {true_label}, Image: {image_path}")

    for root, dirs, files in os.walk(non_cancer_directory_path):
        for file in files:
            image_path = os.path.join(root, file)
            true_label = "non_cancerous"
            predicted_label, _ = predictor.predict(image_path)

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            print(f"Predicted: {predicted_label}, True: {true_label}, Image: {image_path}")

    accuracy = np.mean(np.array(true_labels) == np.array(predicted_labels))
    print(f"Accuracy: {accuracy:.2%}")
