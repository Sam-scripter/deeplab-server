import tensorflow as tf
import numpy as np
from PIL import Image

def load_deeplab_model(model_path):
    """Load the DeepLab model from a frozen graph."""
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.compat.v1.import_graph_def(graph_def, name='')
    return graph

def run_deeplab_inference(graph, image):
    """Run DeepLab inference on an image."""
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('ImageTensor:0')
        output_tensor = graph.get_tensor_by_name('SemanticPredictions:0')

        #Preprocess the image
        image = image.resize((513, 513)) #deeplab input size
        input_data = np.array(image, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        #Run inference
        output_data = sess.run(output_tensor, feed_dict={input_tensor: input_data})

        return output_data[0] #return the segmentation mask
    
def estimate_body_measurements(segmentation_mask):
    """Estimate body measurements from the segmentation mask."""
    body_mask = (segmentation_mask == 15)  # Class 15 is typically "person"

    # Estimate height
    height_pixels = np.sum(body_mask, axis=0).max()

    # Estimate shoulder width (top part of the body)
    shoulder_width_pixels = np.sum(body_mask[100:150, :], axis=0).max()  # Example pixel range for shoulder

    # Estimate waist width (middle part of the body)
    waist_width_pixels = np.sum(body_mask[250:300, :], axis=0).max()  # Example pixel range for waist

    # Estimate leg length
    leg_length_pixels = np.sum(body_mask[300:, :], axis=0).max()  # Example pixel range for legs

    return {
        'height_pixels': height_pixels,
        'shoulder_width_pixels': shoulder_width_pixels,
        'waist_width_pixels': waist_width_pixels,
        'leg_length_pixels': leg_length_pixels
    }

def calculate_real_measurements(estimated_pixels):
    """Calculate real-world measurements from pixel estimates."""
    scaling_factor = 170 / 513  # Based on assumed height of 170cm and image size of 513px
    real_measurements = {k: v * scaling_factor for k, v in estimated_pixels.items()}
    
    return real_measurements
