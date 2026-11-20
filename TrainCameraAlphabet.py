import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from Application import Application, MODEL_DIR, preprocess_roi, _keras_load_model
from TrainAlphabetModel import (
    CLASS_DIRS,
    CLASS_NAMES,
    cascade_predictions,
    load_split,
    split_training_indices,
)


PROJECT_DIR = Path(__file__).resolve().parent
CAMERA_DATA_DIR = PROJECT_DIR / 'dataSet' / 'cameraData'
CAMERA_CANDIDATE_PATH = MODEL_DIR / 'model_alphabet_camera_candidate.keras'
MIN_TRAIN_SAMPLES_PER_CLASS = 24
MIN_TEST_SAMPLES_PER_CLASS = 8
IMAGE_SHAPE = (128, 128)


def load_camera_split(split, minimum_per_class):
    images = []
    labels = []

    for label, class_name in enumerate(CLASS_DIRS):
        class_dir = CAMERA_DATA_DIR / split / class_name
        image_paths = sorted(class_dir.glob('*.jpg'))
        if len(image_paths) < minimum_per_class:
            raise ValueError(
                f'{class_dir} has {len(image_paths)} images; '
                f'capture at least {minimum_per_class} for every class.'
            )

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f'Unable to read camera sample: {image_path}')
            processed = preprocess_roi(image)
            images.append(cv2.resize(processed, IMAGE_SHAPE))
            labels.append(label)

    return (
        np.asarray(images, dtype=np.float32)[..., np.newaxis],
        np.asarray(labels, dtype=np.int32),
    )


def balance_training_data(images, labels):
    rng = np.random.default_rng(42)
    class_indices = [np.flatnonzero(labels == label) for label in range(len(CLASS_NAMES))]
    sample_count = min(len(indices) for indices in class_indices)
    selected = np.concatenate([
        rng.choice(indices, sample_count, replace=False)
        for indices in class_indices
    ])
    rng.shuffle(selected)
    return images[selected], labels[selected]


def class_accuracies(predictions, labels):
    return {
        class_index: float(np.mean(predictions[labels == class_index] == class_index))
        for class_index in range(len(CLASS_NAMES))
    }


def load_models():
    active_path = MODEL_DIR / 'model_alphabet.keras'
    if active_path.is_file():
        baseline = _keras_load_model(str(active_path), compile=False)
        candidate = _keras_load_model(str(active_path), compile=False)
    else:
        baseline = Application.load_model('model_new')
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
                ),
            ],
            name='camera_alphabet_augmentation',
        )
        inputs = tf.keras.Input(shape=(*IMAGE_SHAPE, 1))
        candidate = tf.keras.Model(inputs, base_model(augmentation(inputs)))

    if candidate.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f'Expected {len(CLASS_NAMES)} output classes, got {candidate.output_shape[-1]}'
        )
    return baseline, candidate


def main():
    camera_train_images, camera_train_labels = load_camera_split(
        'trainingData', MIN_TRAIN_SAMPLES_PER_CLASS
    )
    camera_test_images, camera_test_labels = load_camera_split(
        'testingData', MIN_TEST_SAMPLES_PER_CLASS
    )
    camera_train_images, camera_train_labels = balance_training_data(
        camera_train_images, camera_train_labels
    )
    train_indices, validation_indices = split_training_indices(camera_train_labels)

    baseline, candidate = load_models()
    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    if CAMERA_CANDIDATE_PATH.exists():
        CAMERA_CANDIDATE_PATH.unlink()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(CAMERA_CANDIDATE_PATH),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]
    candidate.fit(
        camera_train_images[train_indices],
        camera_train_labels[train_indices],
        validation_data=(
            camera_train_images[validation_indices],
            camera_train_labels[validation_indices],
        ),
        batch_size=32,
        epochs=20,
        callbacks=callbacks,
        verbose=2,
    )

    candidate = tf.keras.models.load_model(
        str(CAMERA_CANDIDATE_PATH), compile=False
    )
    clean_path = MODEL_DIR / 'model_alphabet_camera_candidate_clean.keras'
    candidate.save(str(clean_path))
    os.replace(clean_path, CAMERA_CANDIDATE_PATH)

    auxiliary = [
        Application.load_model(name)
        for name in ('model-bw_dru', 'model-bw_tkdi', 'model-bw_smn')
    ]
    baseline_camera = cascade_predictions(
        baseline, auxiliary, camera_test_images
    )
    candidate_camera = cascade_predictions(
        candidate, auxiliary, camera_test_images
    )
    baseline_camera_accuracy = float(np.mean(baseline_camera == camera_test_labels))
    candidate_camera_accuracy = float(np.mean(candidate_camera == camera_test_labels))
    baseline_camera_classes = class_accuracies(
        baseline_camera, camera_test_labels
    )
    candidate_camera_classes = class_accuracies(
        candidate_camera, camera_test_labels
    )

    original_test_images, original_test_labels = load_split('testingData')
    baseline_original = cascade_predictions(
        baseline, auxiliary, original_test_images
    )
    candidate_original = cascade_predictions(
        candidate, auxiliary, original_test_images
    )
    baseline_original_accuracy = float(
        np.mean(baseline_original == original_test_labels)
    )
    candidate_original_accuracy = float(
        np.mean(candidate_original == original_test_labels)
    )

    print(f'Camera test accuracy: {baseline_camera_accuracy:.2%} -> {candidate_camera_accuracy:.2%}')
    print(f'Original test accuracy: {baseline_original_accuracy:.2%} -> {candidate_original_accuracy:.2%}')
    for class_index, class_name in enumerate(CLASS_NAMES):
        print(
            f'{class_name}: '
            f'{baseline_camera_classes[class_index]:.2%} -> '
            f'{candidate_camera_classes[class_index]:.2%}'
        )

    regressed_classes = [
        CLASS_NAMES[index]
        for index in range(len(CLASS_NAMES))
        if candidate_camera_classes[index] < baseline_camera_classes[index] - 0.15
    ]
    if (
        candidate_camera_accuracy > baseline_camera_accuracy
        and candidate_original_accuracy >= baseline_original_accuracy - 0.02
        and not regressed_classes
    ):
        shutil.copy2(
            CAMERA_CANDIDATE_PATH,
            MODEL_DIR / 'model_alphabet.keras',
        )
        print('Promoted camera-trained alphabet model.')
    else:
        print(
            'Kept the current app model. The candidate must improve camera accuracy, '
            'preserve original accuracy, and avoid large class regressions.'
        )
        if regressed_classes:
            print('Classes needing more camera samples: ' + ', '.join(regressed_classes))


if __name__ == '__main__':
    main()
