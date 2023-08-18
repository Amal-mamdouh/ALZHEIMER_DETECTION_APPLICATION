# ALZHEIMER_DETECTION_APPLICATION

## Description
This repository contains the source code for a web application that can detect Alzheimer's disease from brain scan images. The application is built using Flask, Keras, and Firebase.

Trained on a robust dataset of brain MRI scans, our custom Convolutional Neural Network (CNN) model achieves an impressive detection accuracy of 95.78%. The model is seamlessly integrated into the user-friendly web application developed using Python and the Flask web framework.

The application allows users to upload brain scan images and receive a prediction of the disease stage (Mild, Moderate, Very Mild, or Non Demented) using the pre-trained CNN model. The predicted results are displayed on the web page along with the uploaded image.

##Run the application:
 open PyCharm and run app.py 

## Usage
1. On the home page, click the "Scan" button to go to the scanning page.

2. Upload a brain scan image using the provided file upload button.

3. Click the "Submit" button to process the image and view the prediction.

## Acknowledgements
- The deep learning model used in this application is trained on the [Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset](http://adni.loni.usc.edu/).
- Special thanks to the developers of Flask, Keras, and Firebase for providing the tools and libraries necessary to build this application.
