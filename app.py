import os
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import streamlit as st
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
from PIL import Image
import lzma

detector = MTCNN()

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

with lzma.open('embedding.pkl.xz', 'rb') as f:
    feature_list = pickle.load(f)
feature_list=feature_list[:5000]
filenames = pickle.load(open('filenames.pkl', 'rb'))
st.title('Which bollywood celebrity are you?')


def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('data.zip', uploaded_img.name), 'wb') as f:
            f.write(uploaded_img.getbuffer())
            print('True')
        return True
    except:
        print('False')
        return False


def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    face = img[y:y + height, x:x + width]

    # face_detection completed

    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def recommend(feature_list, feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


uploaded_img = st.file_uploader('Choose an image')

if uploaded_img is not None:

    
    display_image = Image.open(uploaded_img)
    st.image(display_image)
    features = extract_features(os.path.join('uploads', uploaded_img.name), model, detector)
    index_pos = recommend(feature_list, features)
    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
    col1, col2 = st.columns(2)

    with col1:
        st.header('Your uploaded image')
        st.image(display_image, width=300)
    with col2:
        st.header("Seems Like " + predicted_actor)
        st.image(filenames[index_pos], width=300)
# import urllib
#
# from flask import Flask, request, jsonify
# import cv2
# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from mtcnn import MTCNN
# from PIL import Image
# import requests
# from flask_cors import CORS
#
# app = Flask(__name__)
# CORS(app)
#
# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#
# detector = MTCNN()
#
#
# @app.route('/capture2', methods=['POST'])
# def capture2():
#     try:
#         datas = request.get_json(force=True)
#         products=[]
#
#         # Assuming 'Products' and 'res' are keys in the JSON data
#         data = datas['Products']
#         res = datas['res']
#         for i in data:
#             response = requests.get(i['image'])
#             file_path = "uploaded_image.jpg"
#             with open(file_path, 'wb') as file:
#                 file.write(response.content)
#             result = extract_features(file_path, model, detector)
#
#             whether = cosine_similarity(result.reshape(1, -1), np.array(res).reshape(1, -1))[0][0]
#             if(whether>0.5):
#                 products.append(i)
#                 print(i['image'], whether)
#
#         return products
#
#         # print(type(result))
#         # print(data['imgArray'])
#         # # arr=np.array([data['imgArray']['image']])
#         # l = []
#         # for i in data['imgArray']:
#         #     l.append(i['image'])
#         #     print(l)
#     except Exception as e:
#         print('Error:', str(e))
#         return jsonify(success=False, message='Internal Server Error')
#
#
# @app.route('/capture', methods=['POST'])
# def capture():
#     try:
#         if 'file' in request.files:
#             uploaded_file = request.files['file']
#
#             # Save the uploaded file
#             file_path = 'uploaded_image.jpg'
#             uploaded_file.save(file_path)
#
#         # Here, you can perform face recognition or any other processing
#         result = extract_features(file_path, model, detector)
#         # For demonstration purposes, send a dummy response
#         response = jsonify(result.tolist())
#         return response
#
#     except Exception as e:
#         print('Error:', str(e))
#         return jsonify(success=False, message='Internal Server Error')
#
#
# def extract_features(image_src, model, detector):
#     # Decode the image using OpenCV
#     img = cv2.imread(image_src)
#     # results = detector.detect_faces(img)
#     # print(results)
#     # x, y, width, height = results[0]['box']
#     #
#     # face = img[y:y + height, x:x + width]
#
#     # face_detection completed
#
#     image = Image.fromarray(img)
#     image = image.resize((224, 224))
#
#     face_array = np.asarray(image)
#
#     face_array = face_array.astype('float32')
#
#     expanded_img = np.expand_dims(face_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#     return result
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
