<h1 align="center" id="title">MNIST Handwritten Digit Recognition with PyTorch</h1>

<p align="center"><img src="https://socialify.git.ci/mjkj09/HandwrittenDigitsAI/image?font=Inter&amp;language=1&amp;name=1&amp;owner=1&amp;pattern=Solid&amp;theme=Light" alt="project-image"></p>

<p id="description">A project demonstrating how to train a Convolutional Neural Network (CNN) on the MNIST dataset to recognize handwritten digits.</p>

<p align="center"><img src="https://img.shields.io/github/contributors-anon/mjkj09/HandwrittenDigitsAI?style=for-the-badge" alt="shields"> <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&amp;logo=python&amp;logoColor=ffdd54" alt="shields"> <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&amp;logo=PyTorch&amp;logoColor=white" alt="shields"> <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&amp;logo=scikit-learn&amp;logoColor=white" alt="shields"> <img src="https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&amp;logo=streamlit&amp;logoColor=white" alt="shields"></p>


<br><h2>Project Screenshots:</h2>

<img src="https://snipboard.io/nhNzsE.jpg" alt="project-screenshot">

  
<br><h2>🧐 Features</h2>

Here're some of the project's best features:

*   Convolutional Neural Network that classifies digits (0–9) in 28×28 grayscale images.
*   Batch Normalization and Dropout to help stabilize training and reduce overfitting.
*   Data Augmentation (random rotations shifts scale and shear) to improve generalization.
*   Automatic checkpointing of the best model during training.
*   Confusion matrix generation for performance analysis.
*   Streamlit canvas for drawing digits and obtaining real-time predictions.

<br><h2>🛠️ Installation Steps:</h2>

<p>1. Clone this repository or download it as a ZIP:</p>

```
git clone https://github.com/YourUsername/HandwrittenDigitsAI.git 
cd HandwrittenDigitsAI
```

<br><p>2. (Optional) Create and activate a virtual environment:</p>

```
python -m venv venv 
venv\Scripts\activate
```

<br><p>3. Install the required packages:</p>

```
pip install -r requirements.txt
```

<br><h2>⚙️ Usage:</h2>

<p>Training the Model:</p>
1. Confirm that a models/ folder exists (the script will create it if needed).

2. Run:
```
python train.py
```

3. The script downloads the MNIST dataset to data/ (if not already present) and begins training.

4. Each epoch prints the loss and accuracy on the test set.

5. The best model is automatically saved to models/ under a name that includes the best accuracy and a timestamp.

6. A confusion matrix is generated and saved to confusion_matrices/.


<br><p>Running the Streamlit App</p>
1. Ensure that a trained model (e.g., mnist_cnn_xxx.pth) is present in the models/ folder.

2. Launch the app:
```
streamlit run app.py
```

3. A local webpage will open.

4. Draw a digit on the black canvas using white strokes.

5. The app resizes and normalizes your drawing, then displays the predicted digit alongside a probability distribution.


<br><h2>📁 Repository Structure:</h2>

HandwrittenDigitsAI/
├── CNN.py               # Defines the CNN model class
├── train.py             # Script to train/evaluate the model and save the best checkpoint
├── app.py               # Streamlit application for real-time digit recognition
├── models/              # Folder storing saved .pth model files
├── confusion_matrices/  # Folder for generated confusion matrix images
├── data/                # MNIST dataset is downloaded here
└── README.md            # This README file
