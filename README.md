# EEG Data Classification Using Deep Learning

This repository contains a notebook for classifying EEG data using deep learning models. The notebook is structured as follows:

## 1. Introduction
This section provides the objective and background for the study. The aim is to classify EEG data into different states (e.g., task vs. rest) using deep learning techniques.

## 2. Data Loading
This section includes the code and explanations for loading the EEG data from `.edf` files. It ensures that the data is properly preprocessed and ready for analysis.

## 3. Power Spectral Density (PSD) Analysis
In this section, the Power Spectral Density (PSD) is calculated for the EEG data. The PSD analysis helps in understanding the frequency components of the signals, which is crucial for feature extraction.

## 4. Data Preparation for Models
This section describes how the data is prepared for input into the deep learning models. It includes feature extraction, normalization, and splitting the data into training and testing sets.

## 5. Deep Learning Models
This section covers the implementation and training of two deep learning models:

### Model 1: EEGNet
- Implementation details of the EEGNet model, a specialized neural network designed for EEG data.
- Code for training the EEGNet model on the prepared data.

### Model 2: Vision Transformer (ViT)
- Implementation details of the Vision Transformer (ViT) model adapted for EEG data.
- Code for training the ViT model on the prepared data.

## 6. Evaluation
This section summarizes the results of the trained models. It includes metrics such as accuracy, precision, recall, and F1-score to evaluate the performance of each model.

## Results

The notebook provides a detailed analysis of the performance of each model. Below are the key metrics for both models:

### EEGNet
- **Accuracy**: 0.9333
- **Precision**: 0.8333
- **Recall**: 1.0
- **F1-score**: 0.9091

### Custom Transformer
- **Accuracy**: 0.9333
- **Precision**: 0.8333
- **Recall**: 1.0
- **F1-score**: 0.9091

## Conclusion

Both EEGNet and the Vision Transformer models show excellent performance in classifying EEG data, with identical metrics in this evaluation. The choice of model can depend on other factors such as computational efficiency and implementation complexity.

## Usage

To run the notebook, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```
2. Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```
3. Ensure you have all the required dependencies installed. You can use the following command to install them:
    ```bash
    pip install -r requirements.txt
    ```
4. Open the notebook and run the cells sequentially:
    ```bash
    jupyter notebook Deep_learning_eeg_task.ipynb
    ```

## License

This project is licensed under the MIT License.

## Acknowledgements

Include any acknowledgements or references here.

