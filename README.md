# Digit Recognition using Deep Learning

This project implements a deep learning model to recognize handwritten digits (0-9) using the MNIST dataset. The model is trained using the Keras library with TensorFlow as the backend.

## Project Structure

- `digit_recognition.ipynb`: The Jupyter notebook containing the code for training the model and testing it on custom images.
- `digit/`: Folder containing custom digit images (`digit1.png` to `digit19.png`) for testing the model.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/mitali-dxt/Digit_Recognition_Project.git
    cd Digit_Recognition_Project
    ```

2. Install the required packages:

    ```bash
    pip install tensorflow keras numpy matplotlib opencv-python
    ```

## Usage

1. **Training the Model**: The model is trained on the MNIST dataset, which is loaded and preprocessed in the Jupyter notebook. The model architecture consists of:
    - An input layer with 784 neurons (28x28 pixels, flattened).
    - Two hidden layers with 32 and 64 neurons respectively, using ReLU activation.
    - An output layer with 10 neurons (one for each digit), using softmax activation.

    The model is compiled using categorical crossentropy loss and the Adam optimizer. It is then trained for 10 epochs with a batch size of 100.

2. **Testing the Model**: The model is evaluated on the test set of the MNIST dataset. Additionally, it is tested on custom images from the `digits/` directory.

3. **Running the Notebook**:
    - Upload the `digits/` folder to the Google Colab environment.
    - Run the cells in the `Deep_Learning_Project.ipynb` notebook.
    - The notebook will iterate through each image in the `digits` folder, preprocess it, make predictions, and display the results.

4. **Prediction on Custom Images**: The notebook will process each image in the `digits` folder, predict the digit, and display both the image and the prediction.

## Example Output

![image](https://github.com/mitali-dxt/Digit_Recognition_Project/assets/131600078/2e974943-fb3b-4ee0-9edc-71f8e5f3cc1d)

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a new Pull Request.

## Acknowledgements

- The MNIST dataset (http://yann.lecun.com/exdb/mnist/).
- The Keras library (https://keras.io/).
- TensorFlow (https://www.tensorflow.org/).


