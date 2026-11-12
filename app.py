import streamlit as st, torch, torch.nn as nn, numpy as np
from torchvision import models, transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title='Skin Lesion Classifier', page_icon='🔬', layout='centered')

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NAMES = {'akiec': 'Actinic keratosis / Bowen', 'bcc': 'Basal cell carcinoma',
         'bkl': 'Benign keratosis', 'df': 'Dermatofibroma',
         'mel': 'Melanoma', 'nv': 'Melanocytic nevus (mole)', 'vasc': 'Vascular lesion'}
IMG, MEL_BOOST = 256, 1.6   # keep your tuned value here
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

MODES = {
    'Standard': ('Balanced across all lesion types.', 1.0),
    'Cautious': ('Flags possible melanoma more readily: fewer missed melanomas, '
                 'but more false alarms.', MEL_BOOST),
}

st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: 2.5rem;}
.title {text-align: center; font-size: 2.6rem; font-weight: 800; margin: 0 0 1.5rem 0;}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 16px;
       padding: 1.3rem 1.5rem; margin-top: 1.2rem;}
.small {font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; opacity: .6;}
.name {font-size: 1.9rem; font-weight: 700; margin-top: .2rem;}
.conf {opacity: .7; margin-bottom: 1rem;}
.lbl {display: flex; justify-content: space-between; font-size: .95rem; margin: .7rem 0 .25rem;}
.track {height: 10px; border-radius: 6px; background: rgba(128,128,128,.2);}
.fill {height: 10px; border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    m = models.efficientnet_b0()
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    m.load_state_dict(torch.load('effb0_r2b_s42.pt', map_location='cpu'))
    return m.eval()

def pct(p):
    return '<1%' if p < 0.01 else f'{p:.0%}'

def bar(label, p, top):
    color = '#2563eb' if top else '#94a3b8'
    return (f'<div class="lbl"><span>{label}</span><span>{pct(p)}</span></div>'
            f'<div class="track"><div class="fill" style="width:{max(p*100, 1):.1f}%;'
            f'background:{color}"></div></div>')

model = load_model()
st.markdown('<div class="title">Skin Lesion Classifier</div>', unsafe_allow_html=True)

mode = st.radio('Detection mode', list(MODES), horizontal=True)
st.caption(MODES[mode][0])
file = st.file_uploader('Upload a dermoscopy image (JPG or PNG)', type=['jpg', 'jpeg', 'png'])

if file:
    with st.spinner('Analyzing image...'):
        img = Image.open(file).convert('RGB')
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            p = (model(x).softmax(1) + model(torch.flip(x, [3])).softmax(1)
                 + model(torch.flip(x, [2])).softmax(1)) / 3
        scale = np.ones(7)
        scale[CLASSES.index('mel')] = MODES[mode][1]
        adj = p[0].numpy() * scale
        adj /= adj.sum()
        pred = int(adj.argmax())
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        rgb = np.array(img.resize((IMG, IMG))).astype(np.float32) / 255
        overlay = show_cam_on_image(rgb, g, use_rgb=True)

    c1, c2 = st.columns(2)
    c1.image(rgb, caption='Your image', use_container_width=True)
    c2.image(overlay, caption='Where the model looked (red = most important)',
             use_container_width=True)

    top = np.argsort(adj)[::-1][:3]
    bars = ''.join(bar(NAMES[CLASSES[i]], float(adj[i]), k == 0) for k, i in enumerate(top))
    st.markdown(f'<div class="card"><div class="small">Most likely</div>'
                f'<div class="name">{NAMES[CLASSES[pred]]}</div>'
                f'<div class="conf">{pct(float(adj[pred]))} confidence</div>{bars}</div>',
                unsafe_allow_html=True)
    if adj[pred] < 0.5:
        st.caption('Low confidence: treat this result with extra caution.')

st.caption('Educational project. Not a medical device and not a substitute for professional advice.')
