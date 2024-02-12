from django.shortcuts import render
from .forms import UploadForm 
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from io import BytesIO
def predict_tumor(request):
    
    if request.method == 'POST':
        
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            
            # Get the uploaded image from the form
            uploaded_image = request.FILES['image']
            
            # Convert the InMemoryUploadedFile to bytes using BytesIO
            image_bytes = BytesIO(uploaded_image.read())
            # Load the trained model
            model = load_model("D:\\django_project\\main\\mlproject\\scripts\\brainModel.h5")
            

            # Define class labels
            class_labels = ['glioma', 'meningioma', 'no-tumor', 'pituitary']

            # Load and preprocess the image
            img = image.load_img(image_bytes, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.  # Rescale pixel values to [0, 1]

            # Make prediction
            predictions = model.predict(img_array)
            
            # Get predicted class index
            predicted_class_index = np.argmax(predictions[0])
            
            # Get predicted class label
            predicted_class_label = class_labels[predicted_class_index]
            
            # Return the predicted class label along with the form to the index.html template
            return render(request, 'index.html', {'predicted_class_label': predicted_class_label})
    else:
        form = UploadForm()
    return render(request, 'index.html', {'form': form})
