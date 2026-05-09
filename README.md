# BreastCare AI: Precision Screening & Support

BreastCare AI is a full-stack medical imaging application designed for the early detection of breast cancer. This platform integrates Deep Learning models with a secure, patient-centered web environment.

##  Key Features

* **AI-Powered Analysis:** Upload mammograms for real-time diagnostic predictions using DenseNet121 and EfficientNet architectures.
* **Secure Authentication:** User data is protected using **PBKDF2 with SHA-256** password hashing.
* **Survivor Stories:** A dedicated space for community narratives and emotional support.
* **History Dashboard:** Securely track and review previous scan results over time.

##  Technology Stack

* **Frontend:** HTML5, CSS3, JavaScript
* **Backend:** Flask (Python)
* **Database:** SQLAlchemy (PostgreSQL/SQLite)
* **AI/ML:** TensorFlow/Keras, OpenCV (Trained on medical imaging datasets)
* **Security:** SHA-256 Hashing, Environment Variables (`python-dotenv`)
* **Environment:** Conda (macOS / MacBook Air)

##  Project Structure

```text
├── backend.py              # Main Flask application logic
├── templates/              # HTML templates (landing, login, dashboard)
├── static/                 # CSS, JS, and image assets
├── models/                 # Pre-trained deep learning models (.h5)
├── BreastCare_Model.ipynb  # Research and training notebook
├── .env                    # Environment variables (NOT pushed to GitHub)
└── .gitignore              # Files excluded from version control


The project first has a model build from scratch which has the model definition as --

model=models.Sequential([
    layers.Input((224,224,3)),
    augmentation,
    layers.Conv2D(32,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(64,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(128,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(256,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation=activation),
    layers.Dense(64,activation=activation),
    layers.Dropout(0.5),
    layers.Dense(2,activation='softmax',dtype='float32')
])

This model due to the datasets high precentage of non cancer images to cancer images fails to accurately predict the cancer images in the testing phase.The model has a testing accuracy of -- 71.14%

Then I used the pre trained model of Efficient B3 and then used transfer learning to train this new model to the Breast Cancer Dataset.To protect the model from failing on slightly different or angled images, I used data augmentation during the training phase of the model so it learns on different kinds on images and not just on the particular images in the training dataset.The data augmentation was done by --

model = tf.keras.Sequential([
    layers.Input((224,224,3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.4),
    layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

This model performs well on the testing dataset with a testing accuracy of -- 92.62%
