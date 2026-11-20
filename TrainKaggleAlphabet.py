import os
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Application import (
    Application,
    MODEL_DIR,
    hand_landmark_features,
    preprocess_roi,
    _keras_load_model,
)
from TrainAlphabetModel import (
    CLASS_NAMES,
    cascade_predictions,
    split_training_indices,
)


PROJECT_DIR = Path(__file__).resolve().parent
KAGGLE_DATA_DIR = PROJECT_DIR / 'dataSet' / 'kaggleASLAlphabet' / 'dataset'
KAGGLE_CLASSES = tuple(
    letter for letter in CLASS_NAMES[1:] if letter not in ('J', 'Z')
)
KAGGLE_CANDIDATE_PATH = MODEL_DIR / 'model_alphabet_kaggle_candidate.keras'
MN_SVM_CANDIDATE_PATH = MODEL_DIR / 'mn_landmark_svm_candidate.xml'
TRAIN_SAMPLES_PER_CLASS = 75
TEST_SAMPLES_PER_CLASS = 20
IMAGE_SHAPE = (128, 128)


def load_images(paths, class_index):
    images = []
    labels = []
    for image_path in paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Unable to read training image: {image_path}')
        processed = preprocess_roi(image)
        images.append(cv2.resize(processed, IMAGE_SHAPE))
        labels.append(class_index)
    return images, labels


def sample_paths(paths, count, rng, label, split):
    if len(paths) < count:
        raise ValueError(
            f'Need {count} {split} images for class {label}, found {len(paths)}.'
        )
    indices = rng.choice(len(paths), count, replace=False)
    return [paths[index] for index in indices]


def load_kaggle_splits():
    if not KAGGLE_DATA_DIR.is_dir():
        raise FileNotFoundError(f'Kaggle dataset not found at {KAGGLE_DATA_DIR}')

    rng = np.random.default_rng(42)
    class_indices = {name: index for index, name in enumerate(CLASS_NAMES)}
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    train_image_paths, test_image_paths = [], []

    for letter in KAGGLE_CLASSES:
        class_dir = KAGGLE_DATA_DIR / f'{letter}-samples'
        paths = sorted(class_dir.glob('*.jpg'))
        required = TRAIN_SAMPLES_PER_CLASS + TEST_SAMPLES_PER_CLASS
        if len(paths) < required:
            raise ValueError(f'{class_dir} needs at least {required} images.')
        order = rng.permutation(len(paths))
        test_paths = [paths[index] for index in order[:TEST_SAMPLES_PER_CLASS]]
        train_paths = [
            paths[index]
            for index in order[
                TEST_SAMPLES_PER_CLASS:TEST_SAMPLES_PER_CLASS + TRAIN_SAMPLES_PER_CLASS
            ]
        ]
        images, labels = load_images(train_paths, class_indices[letter])
        train_images.extend(images)
        train_labels.extend(labels)
        train_image_paths.extend(train_paths)
        images, labels = load_images(test_paths, class_indices[letter])
        test_images.extend(images)
        test_labels.extend(labels)
        test_image_paths.extend(test_paths)

    for label in ('0', 'J', 'Z'):
        class_index = class_indices['blank' if label == '0' else label]
        train_paths = sorted(
            (PROJECT_DIR / 'dataSet' / 'trainingData' / label).glob('*.jpg')
        )
        test_paths = sorted(
            (PROJECT_DIR / 'dataSet' / 'testingData' / label).glob('*.jpg')
        )
        train_paths = sample_paths(
            train_paths, TRAIN_SAMPLES_PER_CLASS, rng, label, 'training'
        )
        test_paths = sample_paths(
            test_paths, TEST_SAMPLES_PER_CLASS, rng, label, 'testing'
        )
        images, labels = load_images(train_paths, class_index)
        train_images.extend(images)
        train_labels.extend(labels)
        train_image_paths.extend(train_paths)
        images, labels = load_images(test_paths, class_index)
        test_images.extend(images)
        test_labels.extend(labels)
        test_image_paths.extend(test_paths)

    train_images = np.asarray(train_images, dtype=np.float32)[..., np.newaxis]
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_images = np.asarray(test_images, dtype=np.float32)[..., np.newaxis]
    test_labels = np.asarray(test_labels, dtype=np.int32)
    train_order = rng.permutation(len(train_labels))
    test_order = rng.permutation(len(test_labels))
    return (
        train_images[train_order],
        train_labels[train_order],
        test_images[test_order],
        test_labels[test_order],
        [train_image_paths[index] for index in train_order],
        [test_image_paths[index] for index in test_order],
    )


def build_model():
    inputs = tf.keras.Input(shape=(*IMAGE_SHAPE, 1))
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                0.04, fill_mode='constant', fill_value=255.0
            ),
            tf.keras.layers.RandomTranslation(
                0.05, 0.05, fill_mode='constant', fill_value=255.0
            ),
            tf.keras.layers.RandomZoom(
                0.08, fill_mode='constant', fill_value=255.0
            ),
        ],
        name='kaggle_alphabet_augmentation',
    )
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(x)

    for filters in (24, 48, 96):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(
            filters, 3, padding='same', use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(192, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name='kaggle_alphabet')


def load_baseline():
    active_path = MODEL_DIR / 'model_alphabet_candidate.keras'
    if active_path.is_file():
        return _keras_load_model(str(active_path), compile=False)
    return Application.load_model('model_new')


def model_predictions(model, images):
    return model.predict(images, batch_size=64, verbose=0).argmax(axis=1)


def new_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_DIR / 'hand_landmarker.task')
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


def image_landmark_features(image_path, landmarker):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f'Unable to read Kaggle hand image: {image_path}')
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(image)
    if not result.hand_landmarks:
        return None
    return hand_landmark_features(result.hand_landmarks[0])


def train_mn_landmark_svm(image_paths, labels):
    m_index = CLASS_NAMES.index('M')
    n_index = CLASS_NAMES.index('N')
    features, targets = [], []
    landmarker = new_landmarker()
    try:
        for image_path, label in zip(image_paths, labels):
            if label not in (m_index, n_index):
                continue
            feature = image_landmark_features(image_path, landmarker)
            if feature is not None:
                features.append(feature)
                targets.append(0 if label == m_index else 1)
    finally:
        landmarker.close()

    counts = np.bincount(np.asarray(targets, dtype=np.int32), minlength=2)
    if np.min(counts) < 50:
        raise ValueError(
            f'Not enough M/N hand landmarks to train specialist: {counts.tolist()}'
        )

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    if not svm.train(
        np.asarray(features, dtype=np.float32),
        cv2.ml.ROW_SAMPLE,
        np.asarray(targets, dtype=np.int32),
    ):
        raise RuntimeError('OpenCV could not train the M/N landmark classifier.')
    return svm


def apply_mn_landmark_svm(predictions, image_paths, svm):
    adjusted = predictions.copy()
    m_index = CLASS_NAMES.index('M')
    n_index = CLASS_NAMES.index('N')
    landmarker = new_landmarker()
    try:
        for index, (prediction, image_path) in enumerate(
            zip(predictions, image_paths)
        ):
            if prediction not in (m_index, n_index):
                continue
            feature = image_landmark_features(image_path, landmarker)
            if feature is None:
                continue
            _, result = svm.predict(feature.reshape(1, -1))
            adjusted[index] = m_index if int(result[0, 0]) == 0 else n_index
    finally:
        landmarker.close()
    return adjusted


def class_accuracies(predictions, labels):
    return {
        index: float(np.mean(predictions[labels == index] == index))
        for index in range(len(CLASS_NAMES))
    }


def main():
    (
        train_images,
        train_labels,
        kaggle_test_images,
        kaggle_test_labels,
        train_image_paths,
        test_image_paths,
    ) = load_kaggle_splits()
    train_indices, validation_indices = split_training_indices(train_labels)
    baseline = load_baseline()
    candidate = build_model()
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    if KAGGLE_CANDIDATE_PATH.exists():
        KAGGLE_CANDIDATE_PATH.unlink()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(KAGGLE_CANDIDATE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]
    candidate.fit(
        train_images[train_indices],
        train_labels[train_indices],
        validation_data=(
            train_images[validation_indices],
            train_labels[validation_indices],
        ),
        batch_size=32,
        epochs=40,
        callbacks=callbacks,
        verbose=2,
    )

    candidate = tf.keras.models.load_model(
        str(KAGGLE_CANDIDATE_PATH), compile=False
    )
    clean_path = MODEL_DIR / 'model_alphabet_kaggle_candidate_clean.keras'
    candidate.save(str(clean_path))
    os.replace(clean_path, KAGGLE_CANDIDATE_PATH)

    mn_svm = train_mn_landmark_svm(train_image_paths, train_labels)
    if MN_SVM_CANDIDATE_PATH.exists():
        MN_SVM_CANDIDATE_PATH.unlink()
    mn_svm.save(str(MN_SVM_CANDIDATE_PATH))

    auxiliary = [
        Application.load_model(name)
        for name in ('model-bw_dru', 'model-bw_tkdi', 'model-bw_smn')
    ]
    baseline_kaggle = cascade_predictions(
        baseline, auxiliary, kaggle_test_images
    )
    candidate_raw = model_predictions(candidate, kaggle_test_images)
    candidate_kaggle = apply_mn_landmark_svm(
        candidate_raw, test_image_paths, mn_svm
    )
    baseline_accuracy = float(np.mean(baseline_kaggle == kaggle_test_labels))
    candidate_accuracy = float(np.mean(candidate_kaggle == kaggle_test_labels))
    accuracies = class_accuracies(candidate_kaggle, kaggle_test_labels)

    print(f'Kaggle held-out accuracy: {baseline_accuracy:.2%} -> {candidate_accuracy:.2%}')
    print(
        'Candidate accuracy after M/N landmark correction: '
        f'{np.mean(candidate_raw == kaggle_test_labels):.2%} -> {candidate_accuracy:.2%}'
    )
    for index, class_name in enumerate(CLASS_NAMES):
        print(f'{class_name}: {accuracies[index]:.2%}')

    legacy_classes = (CLASS_NAMES.index('blank'), CLASS_NAMES.index('J'), CLASS_NAMES.index('Z'))
    weak_classes = [
        CLASS_NAMES[index]
        for index, accuracy in accuracies.items()
        if accuracy < 0.50
    ]
    weak_legacy_classes = [
        CLASS_NAMES[index]
        for index in legacy_classes
        if accuracies[index] < 0.65
    ]
    if candidate_accuracy >= 0.80 and not weak_classes and not weak_legacy_classes:
        shutil.copy2(
            MN_SVM_CANDIDATE_PATH,
            MODEL_DIR / 'mn_landmark_svm.xml',
        )
        shutil.copy2(
            KAGGLE_CANDIDATE_PATH,
            MODEL_DIR / 'model_alphabet.keras',
        )
        print('Promoted Kaggle-trained full-alphabet model.')
    else:
        print(
            'Kept the previous model. Promotion requires at least 80% held-out '
            'accuracy, at least 50% for every class, and at least 65% for blank/J/Z.'
        )
        if weak_classes:
            print('Classes needing more training data: ' + ', '.join(weak_classes))


if __name__ == '__main__':
    main()
