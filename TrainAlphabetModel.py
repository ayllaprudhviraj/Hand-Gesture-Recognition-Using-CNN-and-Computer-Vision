import cv2
import numpy as np
import os
import shutil
import tensorflow as tf
from pathlib import Path
from string import ascii_uppercase

from Application import Application, MODEL_DIR, preprocess_roi


PROJECT_DIR = Path(__file__).resolve().parent
CLASS_DIRS = ('0', *ascii_uppercase)
CLASS_NAMES = ('blank', *ascii_uppercase)
IMAGE_SHAPE = (128, 128)
MODEL_PATH = MODEL_DIR / 'model_alphabet_candidate.keras'


def load_split(name):
    images = []
    labels = []
    split_dir = PROJECT_DIR / 'dataSet' / name

    for label, class_name in enumerate(CLASS_DIRS):
        image_paths = sorted((split_dir / class_name).glob('*.jpg'))
        if not image_paths:
            raise FileNotFoundError(f'No training images found in {split_dir / class_name}')

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f'Unable to read image: {image_path}')
            processed = preprocess_roi(image)
            images.append(cv2.resize(processed, IMAGE_SHAPE))
            labels.append(label)

    return (
        np.asarray(images, dtype=np.float32)[..., np.newaxis],
        np.asarray(labels, dtype=np.int32)
    )


def split_training_indices(labels):
    rng = np.random.default_rng(42)
    training_indices = []
    validation_indices = []

    for label in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        validation_count = max(1, int(len(class_indices) * 0.15))
        validation_indices.extend(class_indices[:validation_count])
        training_indices.extend(class_indices[validation_count:])

    return np.asarray(training_indices), np.asarray(validation_indices)


def cascade_predictions(main, auxiliary, images):
    dru_model, tkdi_model, smn_model = auxiliary
    main_predictions = main.predict(images, batch_size=64, verbose=0)
    dru_predictions = dru_model.predict(images, batch_size=64, verbose=0)
    tkdi_predictions = tkdi_model.predict(images, batch_size=64, verbose=0)
    smn_predictions = smn_model.predict(images, batch_size=64, verbose=0)

    class_indices = {name: index for index, name in enumerate(CLASS_NAMES)}
    predictions = main_predictions.argmax(axis=1)

    for index, class_index in enumerate(predictions):
        symbol = CLASS_NAMES[class_index]
        if symbol in ('D', 'R', 'U'):
            symbol = ('D', 'R', 'U')[int(dru_predictions[index].argmax())]
        if symbol in ('D', 'I', 'K', 'T'):
            symbol = ('D', 'I', 'K', 'T')[int(tkdi_predictions[index].argmax())]
        if symbol in ('M', 'N', 'S'):
            symbol = ('M', 'N', 'S')[int(smn_predictions[index].argmax())]
        predictions[index] = class_indices[symbol]

    return predictions


def main():
    training_images, training_labels = load_split('trainingData')
    test_images, test_labels = load_split('testingData')
    train_indices, validation_indices = split_training_indices(training_labels)

    base_model = Application.load_model('model_new')
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                0.025, fill_mode='constant', fill_value=255.0
            ),
            tf.keras.layers.RandomTranslation(
                0.04, 0.04, fill_mode='constant', fill_value=255.0
            ),
            tf.keras.layers.RandomZoom(
                0.05, fill_mode='constant', fill_value=255.0
            )
        ],
        name='alphabet_augmentation'
    )
    inputs = tf.keras.Input(shape=(*IMAGE_SHAPE, 1))
    outputs = base_model(augmentation(inputs))
    candidate = tf.keras.Model(inputs, outputs)
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    candidate.fit(
        training_images[train_indices],
        training_labels[train_indices],
        validation_data=(
            training_images[validation_indices],
            training_labels[validation_indices]
        ),
        batch_size=64,
        epochs=12,
        callbacks=callbacks,
        verbose=2
    )

    candidate = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    clean_model_path = MODEL_DIR / 'model_alphabet_candidate_clean.keras'
    candidate.save(str(clean_model_path))
    os.replace(clean_model_path, MODEL_PATH)
    auxiliary = [
        Application.load_model(name)
        for name in ('model-bw_dru', 'model-bw_tkdi', 'model-bw_smn')
    ]
    candidate_predictions = cascade_predictions(candidate, auxiliary, test_images)

    baseline = Application.load_model('model_new')
    baseline_predictions = cascade_predictions(baseline, auxiliary, test_images)

    candidate_accuracy = float(np.mean(candidate_predictions == test_labels))
    baseline_accuracy = float(np.mean(baseline_predictions == test_labels))
    a_index = CLASS_NAMES.index('A')
    a_mask = test_labels == a_index
    a_accuracy = float(np.mean(candidate_predictions[a_mask] == test_labels[a_mask]))
    baseline_a_accuracy = float(np.mean(baseline_predictions[a_mask] == test_labels[a_mask]))
    print(f'Existing cascade test accuracy: {baseline_accuracy:.2%}')
    print(f'Candidate cascade test accuracy: {candidate_accuracy:.2%}')
    print(f'Existing A test accuracy: {baseline_a_accuracy:.2%}')
    print(f'Candidate A test accuracy: {a_accuracy:.2%}')
    print(f'Candidate saved for evaluation: {MODEL_PATH}')

    if candidate_accuracy > baseline_accuracy and a_accuracy >= baseline_a_accuracy:
        deployed_path = MODEL_DIR / 'model_alphabet.keras'
        shutil.copy2(MODEL_PATH, deployed_path)
        print(f'Promoted improved alphabet model: {deployed_path}')
    else:
        print('Kept the current app model because the candidate did not meet both accuracy checks.')


if __name__ == '__main__':
    main()
