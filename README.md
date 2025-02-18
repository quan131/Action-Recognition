# **Action recognition using C2D and LSTM**

This project implements an action recognition model using a combination of 2D Convolutional Neural Networks (C2D) and Long Short-Term Memory (LSTM) networks.

## Repository structure
```bash
├───input
│   │───(video test)
│   │───...
├───output
│   ├───(results)
│   ├───...
├───action-recognition-c2d-lstm.ipynb
├───CNN_LSTM.py
│───test.py
├───classInd.txt 
└───README.md
```

**Description:**
- Dataset: UCF101
    + This project uses the UCF101 dataset. However, instead of utilizing the entire dataset, we focus on a subset that includes only 20 classes, with the class names stored in the classInd.txt file.
- action-recognition-c2d-lstm.ipynb: Reads and processes videos to create a dataset, builds a model combining 2D CNN and LSTM, and then trains the model for action recognition in videos.
- CNN_LSTM.py: Defines a CNN-LSTM model for action recognition in videos.
- test.py: Test the action recognition model using video or webcam.

**Usage**
1. Clone the repository
```bash
git clone https://github.com/quan131/Action-Recognition.git
```
2. Data processing and model training: action-recognition-c2d-lstm.ipynb - you can upload this file to Kaggle and run it. We can take advantage of the datasets available on Kaggle to train the model. Then save the checkpoint file to your local machine to use test.py.
3. Test the model: test.py - You can test the model using a video or webcam.

**DEMO**
1. Identify by video and save the results.
[Video demo](https://drive.google.com/file/d/11BJWdPFZnXdVwE1n_y5aR4djginbrcnQ/view?usp=sharing)
2. Identify by webcam.
[Webcam demo](https://drive.google.com/file/d/1OKg3oM78-7oyfE8wyF1lOeqpxhaMPjUi/view?usp=sharing)


This README offers a clear overview of the project's purpose, dataset, implementation details, and usage instructions. Feel free to expand it further as your project evolves with additional functionality.