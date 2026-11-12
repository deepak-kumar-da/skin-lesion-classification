import streamlit as st, torch, torch.nn as nn, numpy as np
from torchvision import models, transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NAMES = {'akiec': 'Actinic keratosis / Bowen', 'bcc': 'Basal cell carcinoma',
         'bkl': 'Benign keratosis', 'df': 'Dermatofibroma',
         'mel': 'Melanoma', 'nv': 'Melanocytic nevus', 'vasc': 'Vascular lesion'}
IMG, MEL_BOOST = 256, 1.6   # set MEL_BOOST to the value printed by `print(best)`
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

@st.cache_resource
def load_model():
    m = models.efficientnet_b0()
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    m.load_state_dict(torch.load('effb0_r2b_s42.pt', map_location='cpu'))
    return m.eval()

model = load_model()
st.title('Skin Lesion Classifier (HAM10000, 7 classes)')
st.caption('Educational project, not a medical device. Not for diagnosis.')
mode = st.sidebar.radio('Mode', ['Balanced', 'High-sensitivity (melanoma)'])
file = st.file_uploader('Upload a dermoscopy image', type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        p = (model(x).softmax(1) + model(torch.flip(x, [3])).softmax(1)
             + model(torch.flip(x, [2])).softmax(1)) / 3
    p = p[0].numpy()
    scale = np.ones(7)
    if mode.startswith('High'):
        scale[CLASSES.index('mel')] = MEL_BOOST
    adj = p * scale; adj /= adj.sum()
    pred = int(adj.argmax())

    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
    rgb = np.array(img.resize((IMG, IMG))).astype(np.float32) / 255
    c1, c2 = st.columns(2)
    c1.image(rgb, caption='Input', use_container_width=True)
    c2.image(show_cam_on_image(rgb, g, use_rgb=True), caption='Grad-CAM', use_container_width=True)

    st.subheader(f'{NAMES[CLASSES[pred]]} ({adj[pred]:.0%})')
    top = np.argsort(adj)[::-1][:3]
    st.bar_chart({NAMES[CLASSES[i]]: float(adj[i]) for i in top})