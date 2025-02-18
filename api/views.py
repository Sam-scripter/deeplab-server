from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np
from .utils import load_deeplab_model, run_deeplab_inference, estimate_body_measurements, calculate_real_measurements

# Load the DeepLab model
MODEL_PATH = "deeplab_model/frozen_inference_graph.pb"
graph = load_deeplab_model(MODEL_PATH)

@csrf_exempt
def segment_body(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = Image.open(image_file)

        # Run DeepLab inference
        segmentation_mask = run_deeplab_inference(graph, image)

        # Estimate body measurements
        estimated_pixels = estimate_body_measurements(segmentation_mask)
        real_measurements = calculate_real_measurements(estimated_pixels)

        # Return the measurements
        return JsonResponse({
            'height': real_measurements['height_pixels'],
            'shoulder_width': real_measurements['shoulder_width_pixels'],
            'waist_width': real_measurements['waist_width_pixels'],
            'leg_length': real_measurements['leg_length_pixels']
        })
    return JsonResponse({'error': 'Invalid request'}, status=400)

