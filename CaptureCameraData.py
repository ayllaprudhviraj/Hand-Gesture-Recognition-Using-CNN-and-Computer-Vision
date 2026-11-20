import time
from pathlib import Path
from string import ascii_uppercase

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / 'dataSet' / 'cameraData'
LABELS = ('0', *ascii_uppercase)
SPLITS = ('trainingData', 'testingData')
BURST_SECONDS = 1.0
FRAME_INTERVAL_SECONDS = 0.12


def ensure_data_directories():
    for split in SPLITS:
        for label in LABELS:
            (DATA_DIR / split / label).mkdir(parents=True, exist_ok=True)


def count_images(split, label):
    return sum(1 for _ in (DATA_DIR / split / label).glob('*.jpg'))


def save_sample(roi, split, label):
    path = DATA_DIR / split / label / f'{time.time_ns()}.jpg'
    if not cv2.imwrite(str(path), roi):
        raise OSError(f'Unable to save camera sample: {path}')


def main():
    ensure_data_directories()
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError('Could not open camera 0. Close other camera apps and try again.')

    split_index = 0
    selected_label = None
    counts = {
        (split, label): count_images(split, label)
        for split in SPLITS
        for label in LABELS
    }
    burst_end = 0.0
    last_sample = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError('Camera stopped returning frames during capture.')

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            size = int(min(height, width) * 0.88)
            x1 = (width - size) // 2
            y1 = (height - size) // 2
            x2, y2 = x1 + size, y1 + size
            roi = frame[y1:y2, x1:x2].copy()

            now = time.monotonic()
            split = SPLITS[split_index]
            if selected_label is not None and now < burst_end:
                if now - last_sample >= FRAME_INTERVAL_SECONDS:
                    save_sample(roi, split, selected_label)
                    counts[(split, selected_label)] += 1
                    last_sample = now
            elif burst_end:
                burst_end = 0.0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            status = f'Split: {split} | Label: {selected_label or "-"}'
            help_text = 'A-Z select letter | 0 = no hand | SPACE = 1s capture | TAB = split | ESC = quit'
            count_text = (
                f'Saved: {counts[(split, selected_label)]}'
                if selected_label is not None else 'Select a label first'
            )
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, help_text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.putText(frame, count_text, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if now < burst_end:
                cv2.putText(frame, 'CAPTURING - move hand slightly', (12, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Alphabet camera data capture', frame)
            cv2.imshow('Capture crop', roi)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            if key == 9:
                split_index = 1 - split_index
                burst_end = 0.0
                continue
            if key == 32:
                if selected_label is None:
                    print('Select a letter or 0 before capturing.')
                else:
                    burst_end = time.monotonic() + BURST_SECONDS
                    last_sample = 0.0
                continue
            if 0 <= key < 128:
                label = chr(key).upper()
                if label in LABELS:
                    selected_label = label
                    burst_end = 0.0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
