import tensorflow as tf
from PIL import Image
import numpy as np
import glob, os

interpreter = tf.lite.Interpreter(model_path='assets/tflite/model_unquant.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [line.strip() for line in open('assets/tflite/labels.txt')]
print("Labels:", labels)

print("\n--- Testing on Curated Test Set (realistic backgrounds) ---")
for stage in ['unripe', 'ripening', 'ripe', 'overripe']:
    files = glob.glob(f'papaya_curated_dataset/test/{stage}/*.jpg')[:3]
    for f in files:
        im = Image.open(f).convert('RGB').resize((224, 224))
        # Note: the Keras model has Rescaling(1./127.5, offset=-1.0) built-in at the input layer!
        # So raw input expected is [0..255] float32!
        input_data = np.array(im, dtype=np.float32)[np.newaxis, ...]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        top = np.argmax(output)
        print(f"[{stage.upper():8s}] {os.path.basename(f):30s} -> {labels[top]:18s} (Conf: {output[top]*100:.1f}%) | {np.round(output, 2)}")
