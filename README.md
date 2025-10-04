🐶 Dog Breed Classification with ResNet34 – README
📌 Overview

This project is a dog breed image classification system built using PyTorch and ResNet34. It classifies dog images into 120 different breeds using transfer learning with a custom classifier head.

The workflow includes:

Label encoding

Data loading & transformation

ResNet34-based model definition

Model training and validation

Test prediction generation with class probabilities

CSV submission file and JSON label mapping export

📁 Dataset

The dataset consists of:

train/: Folder containing dog images for training (id.jpg)

test/: Folder containing dog images to predict (id.jpg)

labels.csv: CSV with two columns: id (image filename without extension) and breed (dog breed)

🧠 Model Architecture

Base Model: ResNet34 pretrained on ImageNet

Custom Head:

Flatten

Linear → ReLU → BatchNorm

Linear → ReLU → Dropout → BatchNorm

Output Layer (Linear) with 120 output classes

Frozen Feature Extractor: The ResNet34 convolutional layers are frozen (no training)

🔧 Setup & Dependencies

Install required packages:

pip install torch torchvision pandas scikit-learn pillow


Ensure the following files/folders are present:

- train/
- test/
- labels.csv

🔄 Data Preprocessing

Image transformations:

Training: Resize, Horizontal Flip, Rotation, Color Jitter, ToTensor

Validation/Test: Resize, ToTensor

Labels are encoded into integers using dict(breed: index) and saved as breed_mapping.json.

🏋️‍♂️ Training

Optimizer: Adam

Loss: CrossEntropyLoss

Epochs: 10

Batch size: 64

Device: CPU (modifiable)

During training, the model logs:

Training Loss

Validation Loss

Accuracy

🧪 Testing & Submission

Each test image is passed through the trained model

Probabilities are calculated using softmax

Results saved to test_submission.csv with columns: file, breed_1, breed_2, ..., breed_120

📦 Output Files

resnet_custom_head.pth: Saved model weights

test_submission.csv: Prediction probabilities for each test image

breed_mapping.json: Dictionary mapping breed names to label indices

📊 Example Output (test_submission.csv)
file	breed1	breed2	...	breed120
abc123.jpg	0.01	0.23	...	0.005
def456.jpg	0.005	0.001	...	0.90
📈 Future Improvements

Fine-tune more layers of ResNet34 for better performance

Use GPU for faster training

Add metrics like F1-score per class

Integrate a validation confusion matrix

📜 License

This project is open-source and free to use under the MIT License.

🔗 Credits

Pretrained ResNet34 from torchvision

Data from dog breed classification competition / dataset
