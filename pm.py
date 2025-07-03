import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from Sub_Functions.Load_data import Load_data2  # Assuming your function returns (340, 3, 227, 227)

# Load and preprocess data
feat, label = Load_data2()

# Reshape to (N, 227, 227, 3)
feat = feat.transpose(0, 2, 3, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    feat, label, test_size=0.25, stratify=label, random_state=42)

# Normalize data using ImageDataGenerator for train and test
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping

# Train the model
model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False),
    epochs=20,

)

# Evaluate the model
loss, accuracy2 = model.evaluate(test_datagen.flow(X_test, y_test, batch_size=32, shuffle=False))
print("The accuracy using model.evaluate: ", accuracy2)
print("The loss using model.evaluate: ", loss)

# Predictions for accuracy score and confusion matrix
y_pred_probs = model.predict(test_datagen.flow(X_test, batch_size=32, shuffle=False))
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Accuracy score
accuracy1 = accuracy_score(y_test, y_pred)
print("The accuracy using accuracy_score: ", accuracy1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
model.save("proposed_model.h5")