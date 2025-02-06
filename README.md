# Fish Classification Using Custom CNN Model

## Project Overview
This project involves building a fish classification model using a **Custom Convolutional Neural Network (CNN)** for identifying fish species from images. The model is trained on an image dataset and uses techniques like image augmentation to improve generalization. The project leverages TensorFlow and Keras to construct the CNN architecture and train the model.

### Key Features
- **Custom CNN Architecture**: A CNN model built from scratch with convolutional, pooling, and dense layers.
- **Data Augmentation**: Applied to the training dataset using techniques like rotation, width/height shifts, and horizontal flips to enhance model robustness.
- **Transfer Learning**: Although not used here, the project has the flexibility to incorporate pre-trained models like VGG16, ResNet50, and MobileNet.
- **Early Stopping**: To prevent overfitting, the model uses an early stopping callback that halts training when no improvement in validation loss is observed.

### Model Performance
- **Accuracy**: The Custom CNN model achieves good accuracy on the testing data.
- **Confusion Matrix & Classification Report**: The performance of the model is evaluated using confusion matrices and classification reports to measure precision, recall, and F1-score.

### Requirements
To run this project, you'll need the following Python libraries:

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas
