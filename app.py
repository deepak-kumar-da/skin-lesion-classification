import glob, os
import streamlit as st, torch, torch.nn as nn, numpy as np
from torchvision import models, transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(page_title='Skin Lesion Classifier', page_icon='🔬', layout='centered')

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
INFO = {
    'akiec': ("Actinic keratosis / Bowen's disease", 'A rough, scaly patch caused by long-term sun damage. It can develop into skin cancer if left untreated.'),
    'bcc': ('Basal cell carcinoma', 'The most common type of skin cancer. It grows slowly and rarely spreads, but it needs medical treatment.'),
    'bkl': ('Benign keratosis', 'A non-cancerous growth, such as a sun spot or age spot. Usually harmless.'),
    'df': ('Dermatofibroma', 'A small, firm bump in the skin. Harmless in almost all cases.'),
    'mel': ('Melanoma', 'A serious type of skin cancer that starts in pigment-producing cells. Early detection matters most.'),
    'nv': ('Melanocytic nevus (mole)', 'A common mole. Usually harmless, but changes in size, shape or color should be checked.'),
    'vasc': ('Vascular lesion', 'A growth made of blood vessels, such as a cherry angioma. Usually harmless.'),
}
IMG, MEL_BOOST = 256, 1.6   # keep your tuned value here
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

MODES = {
    'Standard': ('Balanced across all lesion types, with the best overall accuracy.', 1.0),
    'Cautious': ('Flags possible melanoma more readily. It misses fewer melanomas, but gives more false alarms.', MEL_BOOST),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stMarkdown, .stMarkdown *, .stRadio label, .stCaption, .stSelectbox label, .stFileUploader label {font-family: 'Inter', sans-serif;}
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 800px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 60%, #3b82f6 100%); border-radius: 22px; padding: 2.6rem 2rem; text-align: center; color: #fff; box-shadow: 0 12px 32px rgba(37,99,235,.28); margin-bottom: 1.6rem;}
.hero .pill {display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px; padding: .3rem .9rem; font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;}
.hero .h {font-size: 2.7rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.15; margin: .9rem 0 0;}
.hero .sub {font-size: 1.08rem; line-height: 1.7; opacity: .93; max-width: 560px; margin: .9rem auto 0;}
.step {font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #2563eb; margin: 1.8rem 0 .4rem;}
.card {background: #fff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.6rem 1.7rem; margin-top: 1.2rem; box-shadow: 0 4px 18px rgba(15,23,42,.06);}
.small {font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #64748b;}
.name {font-size: 1.9rem; font-weight: 800; letter-spacing: -.01em; margin: .3rem 0 .6rem; color: #0f172a;}
.badge {display: inline-block; border-radius: 999px; padding: .3rem .85rem; font-size: .85rem; font-weight: 600;}
.badge.ok {background: #dcfce7; color: #166534;}
.badge.mid {background: #fef3c7; color: #92400e;}
.badge.low {background: #fee2e2; color: #991b1b;}
.desc {font-size: 1.02rem; line-height: 1.7; color: #334155; margin-top: 1rem;}
.lbl {display: flex; justify-content: space-between; font-size: .98rem; font-weight: 500; color: #1e293b; margin: .85rem 0 .3rem;}
.track {height: 10px; border-radius: 6px; background: #e2e8f0;}
.fill {height: 10px; border-radius: 6px;}
.foot {text-align: center; font-size: .82rem; line-height: 1.7; color: #64748b; margin-top: 2.5rem;}
[data-testid="stImage"] img {border-radius: 14px;}
[data-testid="stFileUploader"] section {background: #fff; border: 1.5px dashed #cbd5e1; border-radius: 14px;}
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
            f'<div class="track"><div class="fill" style="width:{max(p * 100, 1):.1f}%;background:{color}"></div></div>')

model = load_model()

st.markdown('<div class="hero"><span class="pill">AI-assisted analysis</span>'
            '<div class="h">Skin Lesion Classifier</div>'
            '<div class="sub">Upload a dermoscopy image to see the most likely lesion type, '
            'how confident the model is, and which area of the image it focused on.</div></div>',
            unsafe_allow_html=True)

st.markdown('<div class="step">Step 1 · Choose a detection mode</div>', unsafe_allow_html=True)
mode = st.radio('Detection mode', list(MODES), horizontal=True, label_visibility='collapsed')
st.caption(MODES[mode][0])

st.markdown('<div class="step">Step 2 · Upload an image</div>', unsafe_allow_html=True)
file = st.file_uploader('Upload a dermoscopy image (JPG or PNG)', type=['jpg', 'jpeg', 'png'],
                        label_visibility='collapsed')
samples = sorted(glob.glob('samples/*.jpg'))
sample = None
if not file and samples:
    i = st.selectbox('No image? Try a sample', range(len(samples)),
                     format_func=lambda k: f'Sample {k + 1}', index=None, placeholder='Choose a sample image')
    if i is not None:
        sample = samples[i]

source = file or sample
if source:
    st.markdown('<div class="step">Step 3 · Results</div>', unsafe_allow_html=True)
    with st.spinner('Analyzing image...'):
        img = Image.open(source).convert('RGB')
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
    c2.image(overlay, caption='Where the model looked (red = most important)', use_container_width=True)

    name, desc = INFO[CLASSES[pred]]
    conf = float(adj[pred])
    level, cls = (('High confidence', 'ok') if conf >= 0.75
                  else ('Moderate confidence', 'mid') if conf >= 0.5 else ('Low confidence', 'low'))
    top = np.argsort(adj)[::-1][:3]
    bars = ''.join(bar(INFO[CLASSES[i]][0], float(adj[i]), k == 0) for k, i in enumerate(top))
    st.markdown(f'<div class="card"><div class="small">Most likely result</div>'
                f'<div class="name">{name}</div>'
                f'<span class="badge {cls}">{level} · {pct(conf)}</span>'
                f'<div class="desc">{desc}</div>'
                f'<div class="small" style="margin-top:1.5rem">Top 3 possibilities</div>{bars}</div>',
                unsafe_allow_html=True)
    if conf < 0.5:
        st.warning('The model is not confident about this image. Treat the result with extra caution.')

st.markdown('<div class="step">About</div>', unsafe_allow_html=True)
with st.expander('How this model works'):
    st.markdown(
        '**Model:** EfficientNet-B0, fine-tuned from ImageNet weights on HAM10000 '
        '(10,015 dermoscopy images, 7 lesion types).\n\n'
        '**Fair testing:** images of the same lesion never appear in both training and testing, '
        'so the scores below are not inflated.\n\n'
        '**Standard mode:** 82% accuracy, 0.71 macro-F1.\n\n'
        '**Cautious mode:** finds about 89% of melanomas (65% in Standard), at the cost of more false alarms.\n\n'
        '**Heatmap:** Grad-CAM shows which part of the image influenced the result most.')

st.markdown('<div class="foot">Educational project. Not a medical device and not a substitute for professional advice.<br>'
            'Built with PyTorch, EfficientNet-B0, Grad-CAM and Streamlit · Data: HAM10000 (Tschandl et al., 2018)</div>',
            unsafe_allow_html=True)
