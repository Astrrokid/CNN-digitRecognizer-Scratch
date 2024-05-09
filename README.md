# Neural Network for Handwritten Digit Recognition

This project implements a neural network from scratch in Python using NumPy for recognizing handwritten digits. The neural network architecture consists of one input layer, one hidden layer with ReLU activation function, and one output layer with softmax activation function.

## Dataset
The dataset used is the MNIST dataset, which contains grayscale images of handwritten digits (0-9). Each image is of size 28x28 pixels, resulting in a total of 784 features.

## Code Overview
- `train.csv`: CSV file containing the MNIST training data.
- `Scratch Neural Network.ipynb`: Jupyter Notebook containing the Python code for building, training, and testing the neural network.
- `README.md`: This markdown file providing an overview of the project.

### Code Steps
1. **Importing Libraries:** Importing necessary libraries such as NumPy, Pandas, and Matplotlib.
2. **Loading and Preprocessing Data:** Loading the MNIST training data from the CSV file and preprocessing it.
3. **Initialization of Parameters:** Initializing weights and biases for the neural network.
4. **Activation Functions:** Implementing ReLU and softmax activation functions.
5. **Forward Propagation:** Implementing forward propagation through the neural network.
6. **One-Hot Encoding:** Encoding the labels using one-hot encoding.
7. **Backpropagation:** Implementing backpropagation to compute gradients.
8. **Optimization:** Updating parameters using gradient descent.
9. **Training the Model:** Running gradient descent to train the neural network.
10. **Testing the Model:** Evaluating the model's performance on the test dataset.

## Results
The trained neural network achieved an accuracy of approximately 86.7% on the test dataset, demonstrating its effectiveness in recognizing handwritten digits.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Usage
1. Ensure all required libraries are installed.
2. Run the `Neural_Network_MNIST.ipynb` Jupyter Notebook to load data, train, and test the neural network.

## Author
Nsobundu Chukwudalu C
## License
This project is licensed under the [MIT License](LICENSE).
