# Importing Libraries

import numpy as np

import cv2
import os, sys
import time
import importlib
import json
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pathlib import Path

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker


# Prefer tensorflow.keras (commonly available); fall back to standalone keras if needed.
_model_from_json = None
_keras_load_model = None
_keras_custom_objects = {}
for module_name in ('tensorflow.keras.models', 'keras.models'):
    try:
        mod = importlib.import_module(module_name)
        _model_from_json = getattr(mod, 'model_from_json')
        _keras_load_model = getattr(mod, 'load_model')
        keras_module_name = module_name.rsplit('.models', 1)[0]
        keras_module = importlib.import_module(keras_module_name)
        layers_module = importlib.import_module(keras_module_name + '.layers')
        _keras_custom_objects = {
            'Sequential': keras_module.Sequential,
            **{
                layer_name: getattr(layers_module, layer_name)
                for layer_name in (
                    'InputLayer', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'Dropout'
                )
            }
        }
        break
    except Exception:
        _model_from_json = None
if _model_from_json is None:
    raise ImportError("Neither tensorflow.keras.models nor keras.models could be imported")
model_from_json = _model_from_json
if sys.platform == 'win32':
    tensorflow = importlib.import_module('tensorflow')
    tensorflow.config.threading.set_intra_op_parallelism_threads(4)
    tensorflow.config.threading.set_inter_op_parallelism_threads(2)
MODEL_DIR = Path(__file__).resolve().parent / 'Models'
HAND_LANDMARKER_PATH = MODEL_DIR / 'hand_landmarker.task'


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    threshold = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    _, result = cv2.threshold(
        threshold, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return result


def hand_landmark_features(landmarks):
    points = np.asarray(
        [(point.x, point.y, point.z) for point in landmarks],
        dtype=np.float32
    )
    if points.shape != (21, 3):
        return None

    wrist = points[0]
    palm_width = np.linalg.norm(points[17, :2] - points[5, :2])
    if palm_width < 1e-6:
        return None

    across = (points[17, :2] - points[5, :2]) / palm_width
    along = points[9, :2] - wrist[:2]
    along -= np.dot(along, across) * across
    along_length = np.linalg.norm(along)
    if along_length < 1e-6:
        return None
    along /= along_length

    relative_xy = points[:, :2] - wrist[:2]
    features = np.column_stack(
        (
            relative_xy @ across / palm_width,
            relative_xy @ along / palm_width,
            (points[:, 2] - wrist[2]) / palm_width,
        )
    )
    return features.reshape(-1)


#Application :

class Application:

    def __init__(self):

        self.hs = SpellChecker()
        self._suggestion_word = None
        self._suggestions = []
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.roi_center = (0.5, 0.5)
        self.roi_drag_offset = None
        self.display_roi_box = None
        self.mn_landmark_svm = None
        alphabet_model = MODEL_DIR / 'model_alphabet.keras'
        if alphabet_model.is_file():
            self.loaded_model = _keras_load_model(str(alphabet_model), compile=False)
            self.use_subclassifiers = self.loaded_model.name != 'kaggle_alphabet'
            if self.loaded_model.name == 'kaggle_alphabet':
                mn_svm_path = MODEL_DIR / 'mn_landmark_svm.xml'
                if not mn_svm_path.is_file():
                    raise FileNotFoundError(
                        f'Missing landmark classifier: {mn_svm_path}'
                    )
                self.mn_landmark_svm = cv2.ml.SVM_load(str(mn_svm_path))
            print("Loaded trained alphabet model")
        else:
            self.loaded_model = self.load_model('model_new')
            self.use_subclassifiers = True
        if self.use_subclassifiers:
            self.loaded_model_dru = self.load_model('model-bw_dru')
            self.loaded_model_tkdi = self.load_model('model-bw_tkdi')
            self.loaded_model_smn = self.load_model('model-bw_smn')
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(HAND_LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        self.frame_timestamp_ms = 0

        self.symbols = ascii_uppercase
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in self.symbols:
          self.ct[i] = 0
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1080x720")
        self.root.minsize(1000, 680)

        self.panel = tk.Label(self.root)
        self.panel.place(x = 20, y = 90, width = 640, height = 360)
        self.panel.bind("<ButtonPress-1>", self.start_roi_drag)
        self.panel.bind("<B1-Motion>", self.move_roi)
        self.panel.bind("<ButtonRelease-1>", self.stop_roi_drag)
        
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x = 690, y = 90, width = 340, height = 340)

        self.T = tk.Label(self.root)
        self.T.place(x = 20, y = 10)
        self.T.config(text = "Sign Language To Text Conversion", font = ("Courier", 24, "bold"))

        self.instruction = tk.Label(self.root)
        self.instruction.place(x = 20, y = 55)
        self.instruction.config(
            text = "Put your hand inside the blue box (drag to move). For A, make a fist with the thumb alongside; an open palm is not a letter.",
            font = ("Arial", 12)
        )

        self.panel3 = tk.Label(self.root, anchor = "w")
        self.panel3.place(x = 825, y = 445, width = 200, height = 32)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 690, y = 445)
        self.T1.config(text = "Character:", font = ("Courier", 18, "bold"))

        self.panel4 = tk.Label(self.root, anchor = "w")
        self.panel4.place(x = 770, y = 485, width = 260, height = 32)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 690, y = 485)
        self.T2.config(text = "Word:", font = ("Courier", 18, "bold"))

        self.panel5 = tk.Label(self.root, anchor = "nw", justify = "left", wraplength = 250)
        self.panel5.place(x = 820, y = 525, width = 210, height = 70)

        self.T3 = tk.Label(self.root)
        self.T3.place(x = 690, y = 525)
        self.T3.config(text = "Sentence:", font = ("Courier", 18, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 690, y = 605)
        self.T4.config(text = "Suggestions:", fg = "red", font = ("Courier", 18, "bold"))

        self.bt1 = tk.Button(self.root, command = self.action1, height = 1, width = 12, state = tk.DISABLED)
        self.bt1.place(x = 690, y = 645)

        self.bt2 = tk.Button(self.root, command = self.action2, height = 1, width = 12, state = tk.DISABLED)
        self.bt2.place(x = 810, y = 645)

        self.bt3 = tk.Button(self.root, command = self.action3, height = 1, width = 12, state = tk.DISABLED)
        self.bt3.place(x = 930, y = 645)


        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()

    def start_roi_drag(self, event):
        if self.display_roi_box is None:
            return
        x1, y1, x2, y2 = self.display_roi_box
        if x1 <= event.x <= x2 and y1 <= event.y <= y2:
            self.roi_drag_offset = (event.x - x1, event.y - y1)

    def move_roi(self, event):
        if self.roi_drag_offset is None or self.display_roi_box is None:
            return
        x1, y1, x2, y2 = self.display_roi_box
        box_size = x2 - x1
        left = max(0, min(640 - box_size, event.x - self.roi_drag_offset[0]))
        top = max(0, min(360 - box_size, event.y - self.roi_drag_offset[1]))
        self.roi_center = (
            (left + box_size / 2) / 640,
            (top + box_size / 2) / 360
        )

    def stop_roi_drag(self, _event):
        self.roi_drag_offset = None

    def detect_hand_pose(self, roi):
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.frame_timestamp_ms = max(
            self.frame_timestamp_ms + 1,
            time.monotonic_ns() // 1_000_000
        )
        result = self.hand_landmarker.detect_for_video(image, self.frame_timestamp_ms)
        if not result.hand_landmarks:
            self.current_hand_features = None
            self.hand_detected = False
            return False, False

        landmarks = result.hand_landmarks[0]
        self.hand_detected = True
        self.current_hand_features = hand_landmark_features(landmarks)

        def distance(first, second):
            return np.hypot(
                landmarks[first].x - landmarks[second].x,
                landmarks[first].y - landmarks[second].y
            )

        finger_extension = [
            distance(tip, 0) / max(distance(pip, 0), 1e-6)
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
        ]
        palm_width = max(distance(5, 17), 1e-6)
        thumb_extended = distance(4, 5) > palm_width * 0.45
        open_hand = (
            all(ratio > 1.1 for ratio in finger_extension)
            and thumb_extended
            and distance(8, 20) > palm_width * 0.9
        )

        index_base = np.array((landmarks[5].x, landmarks[5].y))
        pinky_base = np.array((landmarks[17].x, landmarks[17].y))
        thumb_tip = np.array((landmarks[4].x, landmarks[4].y))
        across_fingers = (pinky_base - index_base) / palm_width
        thumb_side = np.dot(thumb_tip - index_base, across_fingers) / palm_width
        a_pose = bool(
            all(ratio < 1.25 for ratio in finger_extension)
            and thumb_extended
            and thumb_side < -0.05
        )
        return open_hand, a_pose

    def detect_open_hand(self, roi):
        open_hand, _ = self.detect_hand_pose(roi)
        return open_hand

    @staticmethod
    def load_model(name):
        with (MODEL_DIR / f'{name}.json').open(encoding='utf-8') as model_file:
            config = json.load(model_file)

        layers = config.get('config', {}).get('layers', [])
        if layers and layers[0].get('class_name') != 'InputLayer':
            first_layer_config = layers[0].get('config', {})
            input_shape = first_layer_config.pop('batch_input_shape', None)
            if input_shape:
                layers.insert(0, {
                    'class_name': 'InputLayer',
                    'config': {
                        'batch_shape': input_shape,
                        'dtype': first_layer_config.get('dtype', 'float32'),
                        'sparse': False,
                        'ragged': False,
                        'name': first_layer_config['name'] + '_input'
                    }
                })
        elif len(layers) > 1:
            layers[1].get('config', {}).pop('batch_input_shape', None)

        model = model_from_json(
            json.dumps(config),
            custom_objects=_keras_custom_objects
        )
        model.load_weights(str(MODEL_DIR / f'{name}.h5'))
        return model


    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            camera_frame = cv2.flip(frame, 1)

            frame_height, frame_width = camera_frame.shape[:2]
            roi_size = int(min(frame_height, frame_width) * 0.88)
            center_x = self.roi_center[0] * frame_width
            center_y = self.roi_center[1] * frame_height
            x1 = max(0, min(frame_width - roi_size, int(center_x - roi_size / 2)))
            y1 = max(0, min(frame_height - roi_size, int(center_y - roi_size / 2)))
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            display_x1 = round(x1 * 640 / frame_width)
            display_y1 = round(y1 * 360 / frame_height)
            display_size = round(roi_size * 640 / frame_width)
            self.display_roi_box = (
                display_x1,
                display_y1,
                display_x1 + display_size,
                display_y1 + display_size
            )
            cv2.rectangle(camera_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 2)
            roi = camera_frame[y1:y2, x1:x2]
            display_frame = cv2.resize(camera_frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            res = preprocess_roi(roi)
            open_hand, a_pose = self.detect_hand_pose(roi)
            self.predict(
                res,
                force_blank=open_hand or not self.hand_detected,
                force_a=a_pose
            )

            preview = cv2.resize(res, (340, 340), interpolation=cv2.INTER_AREA)
            self.current_image2 = Image.fromarray(preview)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            if not self.hand_detected:
                display_symbol = "No hand"
            elif self.current_symbol == 'blank':
                display_symbol = "No letter"
            else:
                display_symbol = self.current_symbol
            self.panel3.config(text = display_symbol, font = ("Courier", 20, "bold"))

            self.panel4.config(text = self.word, font = ("Courier", 30))

            self.panel5.config(text = self.str,font = ("Courier", 30))

            predicts = self.get_suggestions()
            for index, button in enumerate((self.bt1, self.bt2, self.bt3)):
                if index < len(predicts):
                    button.config(
                        text = predicts[index],
                        font = ("Courier", 20),
                        state = tk.NORMAL
                    )
                else:
                    button.config(text = "", state = tk.DISABLED)

        self.root.after(5, self.video_loop  )

    def get_suggestions(self):
        word = self.word.strip().lower()
        if word != self._suggestion_word:
            self._suggestion_word = word
            self._suggestions = []
            if len(word) >= 2:
                candidates = self.hs.candidates(word)
                if candidates:
                    correction = self.hs.correction(word)
                    self._suggestions = sorted(
                        candidates,
                        key = lambda candidate: (
                            candidate != correction,
                            -self.hs.word_frequency.dictionary.get(candidate, 0),
                            candidate
                        )
                    )[:3]
        return self._suggestions

    def choose_suggestion(self, index):
        predicts = self.get_suggestions()
        if index >= len(predicts):
            return
        if self.str and not self.str.endswith(" "):
            self.str += " "
        self.str += predicts[index]
        self.word = ""

    def predict(self, test_image, force_blank=False, force_a=False):
        if force_a:
            self.current_symbol = 'A'
        elif force_blank:
            self.current_symbol = 'blank'
        else:
            test_image = cv2.resize(test_image, (128, 128))
            image = test_image.reshape(1, 128, 128, 1)
            result = self.loaded_model.predict(image, verbose=0)[0]
            classes = ('blank', *ascii_uppercase)
            self.current_symbol = classes[int(np.argmax(result))]

            if self.use_subclassifiers:
                if self.current_symbol in ('D', 'R', 'U'):
                    result_dru = self.loaded_model_dru.predict(image, verbose=0)[0]
                    self.current_symbol = ('D', 'R', 'U')[int(np.argmax(result_dru))]

                if self.current_symbol in ('D', 'I', 'K', 'T'):
                    result_tkdi = self.loaded_model_tkdi.predict(image, verbose=0)[0]
                    self.current_symbol = ('D', 'I', 'K', 'T')[int(np.argmax(result_tkdi))]

                if self.current_symbol in ('M', 'N', 'S'):
                    result_smn = self.loaded_model_smn.predict(image, verbose=0)[0]
                    self.current_symbol = ('M', 'N', 'S')[int(np.argmax(result_smn))]

            if (
                self.current_symbol in ('M', 'N')
                and self.mn_landmark_svm is not None
                and self.current_hand_features is not None
            ):
                _, prediction = self.mn_landmark_svm.predict(
                    self.current_hand_features.reshape(1, -1)
                )
                self.current_symbol = ('M', 'N')[int(prediction[0, 0])]

        if self.current_symbol == 'blank':
            for i in self.symbols:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for i in self.symbols:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if tmp < 0:
                    tmp *= -1
                if tmp <= 20:
                    self.ct['blank'] = 0
                    for i in self.symbols:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0
            for i in self.symbols:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def action1(self):
        self.choose_suggestion(0)

    def action2(self):
        self.choose_suggestion(1)

    def action3(self):
        self.choose_suggestion(2)
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        self.hand_landmarker.close()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    print("Starting Application...")
    Application().root.mainloop()