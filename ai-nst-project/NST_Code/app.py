import os
# pyrefly: ignore [missing-import]
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import io
import numpy as np
import easyocr

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'canvasflow-dev-key')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['TEMPLATES_AUTO_RELOAD'] = True
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content      = FileField('Content Image')
    content_path = HiddenField()
    style_path   = HiddenField()
    alpha        = FloatField('Alpha', default=1.0)
    submit       = SubmitField('Run Style Transfer')


# ── Model loading ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_base_dir = os.path.dirname(os.path.abspath(__file__))
_vgg_path  = os.environ.get(
    'VGG_PATH',
    os.path.join(_base_dir, 'vgg_normalised.pth')
)
_decoder_path = os.environ.get(
    'DECODER_PATH',
    os.path.join(_base_dir, 'experiment', 'final_exp', 'decoder_final.pth')
)

encoder = VGGEncoder(_vgg_path).to(device)
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(_decoder_path, map_location=device))

encoder.eval()
decoder.eval()

# Initialize OCR reader for automatic text detection (cached models)
ocr_reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'))


# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def create_text_mask(content_pil, target_size, alpha=1.0):
    """
    Detect text bounding boxes in content image and produce a feathered grayscale mask.
    The mask intensity dynamically scales with alpha so text clarity responds to transfer strength:
      - alpha in [0.0, 0.5]: text stays mostly clear and readable (1.0 -> 0.80).
      - alpha in (0.5, 1.0]: text progressively tarnishes and blends into the art style (0.80 -> 0.10).
    Returns None if no text is detected.
    """
    orig_w, orig_h = content_pil.size
    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    np_img = np.array(content_pil.convert('RGB'))
    try:
        horizontal_list, free_list = ocr_reader.detect(np_img)
    except Exception:
        return None

    boxes = horizontal_list[0] if horizontal_list else []
    polys = free_list[0] if free_list else []

    if not boxes and not polys:
        return None

    mask = Image.new('L', target_size, 0)
    draw = ImageDraw.Draw(mask)
    pad = 4

    for box in boxes:
        # box is [x_min, x_max, y_min, y_max]
        x_min = max(0, int(box[0] * scale_x) - pad)
        x_max = min(target_w, int(box[1] * scale_x) + pad)
        y_min = max(0, int(box[2] * scale_y) - pad)
        y_max = min(target_h, int(box[3] * scale_y) + pad)
        draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

    for poly in polys:
        scaled_poly = [(int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in poly]
        draw.polygon(scaled_poly, fill=255)

    # Feather edges with Gaussian blur for seamless blending
    feathered = mask.filter(ImageFilter.GaussianBlur(radius=3))

    # Calculate dynamic text preservation weight based on alpha
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    if alpha_clamped <= 0.5:
        # Clear/preserved zone: 1.0 down to 0.80 (stays very clear and readable)
        text_weight = 1.0 - (alpha_clamped / 0.5) * 0.20
    else:
        # Tarnishing zone: 0.80 down to 0.10 (progressively tarnishes into style)
        progress = (alpha_clamped - 0.5) / 0.5
        text_weight = 0.80 - progress * 0.70

    # Scale the mask by text_weight so blending directly responds to alpha
    return feathered.point(lambda p: int(p * text_weight))


def run_style_transfer(content_image, style_image, alpha):
    """
    Run AdaIN style transfer with alpha-responsive text preservation.
    Returns the stylized PIL Image.
    """
    _transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])

    content_tensor = _transform(content_image).unsqueeze(0).to(device)
    style_tensor   = _transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats  = encoder(content_tensor, is_test=True)
        style_feats    = encoder(style_tensor,   is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_tensor = decoder(stylized_feats)

    # Convert stylized output to PIL
    stylized_pil = transforms.ToPILImage()(stylized_tensor.cpu().clone().squeeze(0).clamp(0, 1))

    # Automatic text preservation dynamically modulated by alpha:
    text_mask = create_text_mask(content_image, stylized_pil.size, alpha=alpha)
    if text_mask is not None:
        content_resized = content_image.resize(stylized_pil.size, Image.Resampling.LANCZOS)
        return Image.composite(content_resized, stylized_pil, text_mask)

    return stylized_pil


def save_image(img, path):
    """Save PIL Image or single-image batch tensor to disk as PNG/JPG."""
    if isinstance(img, Image.Image):
        img.save(path)
    else:
        image = img.cpu().clone().squeeze(0).clamp(0, 1)
        transforms.ToPILImage()(image).save(path)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    form           = UploadForm()
    result_image   = None
    content_filename = None
    style_filename   = None
    error            = None

    if form.validate_on_submit():
        # -- Content image --
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = secure_filename(form.content.data.filename)
                form.content.data.save(
                    os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
                )
                form.content_path.data = content_filename
            else:
                error = 'Content image format not supported. Please upload a PNG or JPG file.'
        else:
            content_filename = form.content_path.data or None

        # -- Style image --
        style_filename = form.style_path.data or None
        if not style_filename:
            error = 'Please select a style reference image before generating.'

        # -- Run inference --
        if content_filename and style_filename and not error:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path   = os.path.join('examples', style_filename)

            try:
                content_image = Image.open(content_path).convert('RGB')
                style_image   = Image.open(style_path).convert('RGB')
                alpha         = float(form.alpha.data)

                stylized = run_style_transfer(content_image, style_image, alpha)

                result_filename = 'stylized_' + content_filename
                result_path     = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized, result_path)

                result_image = result_filename

            except FileNotFoundError as e:
                error = 'One or more uploaded files could not be found. Please re-upload and try again.'
            except RuntimeError as e:
                error = 'Style transfer failed during model inference. Check that both images are valid and try again.'
            except Exception as e:
                error = f'An unexpected error occurred: {e}'

        elif not content_filename and not error:
            error = 'Please upload a content image before generating.'
        elif not style_filename and not error:
            error = 'Please select a style reference image before generating.'

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error,
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)