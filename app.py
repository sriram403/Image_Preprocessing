from flask import Flask, jsonify, render_template, request
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/augment-images', methods=['POST'])
def augment_images():
    # Get user input from the form
    S = int(request.form.get('size'))
    A_I_N = int(request.form.get('aug_needed'))
    R_R = int(request.form.get('rotation'))
    S_R = float(request.form.get('shear'))
    Z_R = float(request.form.get('zoom'))
    H_F = request.form.get('horizontal_flip') == 'true'
    B_R = tuple(map(float, request.form.get('brightness').split(',')))
    W_S_R = float(request.form.get('width_shift'))
    H_S_R = float(request.form.get('height_shift'))
    C_S_R = float(request.form.get('channel_shift'))
    V_F = request.form.get('vertical_flip') == 'true'
    R = 1.0 / 255.0

    # DataGen Object
    datagen = ImageDataGenerator(
        rotation_range=R_R,
        shear_range=S_R,
        zoom_range=Z_R,
        horizontal_flip=H_F,
        brightness_range=B_R,
        width_shift_range=W_S_R,
        height_shift_range=H_S_R,
        channel_shift_range=C_S_R,
        vertical_flip=V_F,
        rescale=R)

    # Preprocessing Images
    dataset = []
    image_directory = r'images/'
    augment_directory = r'static/Augmented-images'

    path = Path(augment_directory)
    print(path)
    path.mkdir(exist_ok=True)

    my_images = os.listdir(image_directory)
    for i, image_name in enumerate(my_images):
        if (image_name.split('.')[1] == 'jpg'):
            image_path = os.path.join(image_directory, image_name)
            image = plt.imread(image_path)
            image = Image.fromarray((image).astype(np.uint8))  # Convert float image to uint8
            image = image.resize((S, S))
            dataset.append(np.array(image))

    # Augmenting Images
    x = np.array(dataset)
    i = 0
    for batch in datagen.flow(x, batch_size=16, save_to_dir=augment_directory, save_prefix='Aug', save_format='jpg'):
        i += 1
        if i > A_I_N:
            break

    # Rename the augmented images starting from 0
    renamed_files = os.listdir(augment_directory)
    renamed_files.sort()  # Sort the file names
    for count, filename in enumerate(renamed_files):
        new_filename = f"Aug_{count}.jpg"
        os.rename(os.path.join(augment_directory, filename), os.path.join(augment_directory, new_filename))
    image_count = len(renamed_files)
    return render_template('result.html',image_count = image_count)

if __name__ == '__main__':
    app.run()
