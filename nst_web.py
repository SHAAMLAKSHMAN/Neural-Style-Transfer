import os
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg19
from io import BytesIO
from PIL import Image
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return Image.fromarray(x)

def style_transfer(content_path, style_path, iterations=50):
    # Load models
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1']
    
    # Feature extractor
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=outputs_dict)
    
    # Preprocess images
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)
    generated_image = tf.Variable(content_image, dtype=tf.float32)
    
    # Optimizer
    optimizer = tf.optimizers.Adam(learning_rate=5.0)
    
    # Training loop
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            input_tensor = tf.concat([content_image, style_image, generated_image], axis=0)
            features = feature_extractor(input_tensor)
            
            # Content loss
            content_features = features[content_layer][0]
            gen_content_features = features[content_layer][2]
            c_loss = tf.reduce_mean(tf.square(gen_content_features - content_features))
            
            # Style loss
            s_loss = 0
            for layer in style_layers:
                style_features = features[layer][1]
                gen_style_features = features[layer][2]
                s_loss += tf.reduce_mean(tf.square(gram_matrix(style_features) - gram_matrix(gen_style_features)))
            
            total_loss = 1e-4 * c_loss + 1.0 * s_loss
        
        gradients = tape.gradient(total_loss, [generated_image])
        optimizer.apply_gradients(zip(gradients, [generated_image]))
    
    return deprocess_image(generated_image.numpy())

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    return tf.matmul(features, tf.transpose(features))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Save uploaded files
        content_file = request.files['content']
        style_file = request.files['style']
        
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
        content_file.save(content_path)
        style_file.save(style_path)
        
        # Process images
        start_time = time.time()
        result_image = style_transfer(content_path, style_path)
        processing_time = round(time.time() - start_time, 2)
        
        # Save result
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        result_image.save(result_path)
        
        return render_template('result.html', 
                            time=processing_time,
                            content_image='uploads/content.jpg',
                            style_image='uploads/style.jpg',
                            result_image='results/result.jpg')
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
