"""
Streamlit Web App for CalcModule v2
Advanced Calculator with Image OCR + Symbolic Calculus
"""

# ── Auto-install dependencies ─────────────────────────────────────────────────
import subprocess, sys, os
import sympy as sp
import numpy as np
from PIL import Image
import re
try:
    import cv2
except ImportError:
    cv2 = None

_REQUIRED = [
    "streamlit",
    "sympy",
    "numpy",
    "Pillow",
    "opencv-python-headless",
    "easyocr",
]

def _install_packages():
    """Install missing packages silently at first launch."""
    for pkg in _REQUIRED:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass

# Run once per interpreter session (not every Streamlit rerun)
_install_packages()

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CalcModule v2",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0d1b2a, #415a77, #778da9);
    color: #e0e0e0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}

/* Result cards */
.result-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    backdrop-filter: blur(6px);
}

/* Section headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(130deg, #63b3ed, #e0e1dd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #a0aec0;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(10deg, #1b263b, #1b263b);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: opacity 0.2s ease;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* Input boxes */
.stTextInput > div > input,
.stNumberInput > div > input {
    background: rgba(255,255,255,0.08) !important;
    color: #e0e0e0 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}

/* OCR preview box */
.ocr-box {
    background: rgba(99,179,237,0.08);
    border: 1px dashed rgba(99,179,237,0.4);
    border-radius: 10px;
    padding: 14px;
    font-family: monospace;
    font-size: 1rem;
    color: #90cdf4;
    margin: 8px 0;
}

/* Hero banner */
.hero {
    text-align: center;
    padding: 30px 0 10px 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #0d1b2a, #0d1b2a, #0d1b2a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #a0aec0;
    font-size: 1rem;
    margin-top: -6px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>∫ CalcModule v2</h1>
    <p>Advanced Symbolic Calculator · Image OCR · Engineered by a team of 3 hardworking psyducks</p>
</div>
""", unsafe_allow_html=True)

# ── CalculusCalculator (embedded, no import dependency) ───────────────────────
class CalculusCalculator:
    def __init__(self):
        self.x, self.y, self.z = sp.symbols('x y z')

    def differentiate(self, expr, var=None, order=1):
        if var is None:
            var = self.x
        return sp.diff(expr, var, order)

    def integrate(self, expr, var=None, lower=None, upper=None):
        if var is None:
            var = self.x
        if lower is not None and upper is not None:
            return sp.integrate(expr, (var, lower, upper))
        return sp.integrate(expr, var)

    def limit(self, expr, var=None, point=0, direction="+"):
        if var is None:
            var = self.x
        return sp.limit(expr, var, point, dir=direction)

    def taylor(self, expr, var=None, point=0, order=6):
        if var is None:
            var = self.x
        return sp.series(expr, var, point, order).removeO()

    def evaluate(self, expr, substitutions):
        return expr.subs(substitutions).evalf()

# ── OCR Helper ────────────────────────────────────────────────────────────────
def clean_math_expression(text: str) -> str:
    text = ' '.join(text.split())
    replacements = {
        'X': 'x', '×': '*', '÷': '/',
        '^': '**', '²': '**2', '³': '**3',
        'π': 'pi', ' ': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', text)
    text = re.sub(r'\)(\d)', r')*\1', text)
    text = re.sub(r'(\d)\(', r'\1*(', text)
    return text

# ── Cached EasyOCR reader (loaded once per server process, ~300 MB) ───────────
@st.cache_resource(show_spinner="Loading OCR model… (first time only)")
def _get_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False, verbose=False)

_MAX_OCR_PX = 1920  # cap longest side to avoid OOM on high-res Apple photos

def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Rotate image according to EXIF orientation tag (fixes iPhone portrait photos)."""
    try:
        exif = img._getexif()
        if exif is None:
            return img
        orientation = exif.get(274)  # 274 = Orientation tag
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img

def _resize_for_ocr(img: Image.Image) -> Image.Image:
    """Downscale to at most _MAX_OCR_PX on the longest side (saves memory)."""
    w, h = img.size
    longest = max(w, h)
    if longest > _MAX_OCR_PX:
        scale = _MAX_OCR_PX / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _preprocess_for_ocr(pil_image: Image.Image):
    """Return a contrast-enhanced grayscale image for better OCR accuracy."""
    try:
        import cv2, numpy as np
        img_np = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(thresh)
    except Exception:
        return pil_image  # fall back to original if cv2 unavailable

def ocr_image(pil_image: Image.Image) -> tuple[str, str]:
    """Run OCR using cached EasyOCR reader. Returns (text, engine_used)."""
    pil_image = _apply_exif_orientation(pil_image)  # fix iPhone portrait rotation
    pil_image = _resize_for_ocr(pil_image)           # cap resolution to avoid OOM
    processed = _preprocess_for_ocr(pil_image)
    try:
        reader = _get_ocr_reader()
        results = reader.readtext(np.array(processed.convert("RGB")), detail=0)
        return " ".join(results).strip(), "EasyOCR"
    except Exception as err:
        return f"[OCR failed: {err}]", "error"

# ── Session State ─────────────────────────────────────────────────────────────
if "expr_str" not in st.session_state:
    st.session_state["expr_str"] = "sin(x)**2 + x**2"
if "var_str" not in st.session_state:
    st.session_state["var_str"] = "x"

calc = CalculusCalculator()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("Psyduck.jpg", width=60)
    st.markdown("### Expression Settings")

    expr_input = st.text_input(
        "Expression",
        value=st.session_state["expr_str"],
        help="Use Python/SymPy syntax: sin(x)**2 + x**2"
    )
    var_input = st.text_input(
        "Variable",
        value=st.session_state["var_str"],
        help="Primary variable (e.g. x)"
    )

    st.markdown("---")
    st.markdown("**Quick examples**")
    examples = {
        "sin²(x) + x²": "sin(x)**2 + x**2",
        "x³ − 4x + 1": "x**3 - 4*x + 1",
        "e^(−x²)": "exp(-x**2)",
        "ln(x) / x": "log(x) / x",
        "1 / (1 − x)": "1 / (1 - x)",
    }
    for label, expr in examples.items():
        if st.button(label, key=f"ex_{label}"):
            st.session_state["expr_str"] = expr
            expr_input = expr
            st.rerun()

# ── Parse expression ──────────────────────────────────────────────────────────
parse_ok = False
sym_expr = None
sym_var = None

try:
    if not expr_input.strip():
        st.warning("⚠️ Please enter a mathematical expression.")
        st.stop()
    if not var_input.strip():
        st.warning("⚠️ Please enter a variable.")
        st.stop()

    sym_var = sp.symbols(var_input.strip())
    sym_expr = sp.sympify(expr_input.strip())
    parse_ok = True

    # Update session
    st.session_state["expr_str"] = expr_input
    st.session_state["var_str"] = var_input

    with st.sidebar:
        st.success("Expression converted successfully")
        st.latex(sp.latex(sym_expr))

except sp.SympifyError:
    st.sidebar.error("❌ Invalid expression syntax.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Error: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Format Guide**")
st.sidebar.code("sin, cos, tan, exp, log, sqrt\npi, E\n** for power\n* for multiply", language="")
# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ocr, tab_diff, tab_integ, tab_lim, tab_taylor, tab_eval = st.tabs([
    "Image OCR",
    "Differentiation",
    "Integration",
    "Limit",
    "Taylor Series",
    "Evaluation",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 0 · IMAGE OCR
# ─────────────────────────────────────────────────────────────────────────────
with tab_ocr:
    st.markdown('<div class="section-title">Extract Expression from Image</div>', unsafe_allow_html=True)
    st.markdown("Capture with your camera or upload a photo of a handwritten or printed mathematical expression.")
    st.markdown("Note: For smartphone users, please refrain from using the camera function under the 'Upload Image' tab.")

    # ── Input mode selector ───────────────────────────────────────────────────
    ocr_sub_upload, ocr_sub_camera = st.tabs(["Upload Image", "Use Camera"])

    pil_img = None  # will be set by whichever input is active

    with ocr_sub_upload:
        col_up, col_prev = st.columns([1, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Upload image (JPG / PNG / BMP)",
                type=["jpg", "jpeg", "png", "bmp", "tiff"],
                label_visibility="collapsed",
                key="ocr_upload",
            )
        if uploaded is not None:
            pil_img = Image.open(uploaded)
            with col_prev:
                st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with ocr_sub_camera:
        st.markdown("Point your camera at the expression and press **Take photo**.")
        camera_shot = st.camera_input("Take a photo", key="ocr_camera", label_visibility="collapsed")
        if camera_shot is not None:
            pil_img = Image.open(camera_shot)

    # ── Shared OCR processing ─────────────────────────────────────────────────
    def _show_ocr_results(img: Image.Image, btn_key: str, edit_key: str, load_key: str):
        result_key = f"ocr_result_{btn_key}"

        if st.button("Run OCR", key=btn_key):
            with st.spinner("Running OCR… (first run may take a moment to load models)"):
                raw_text, engine = ocr_image(img)
                is_error = raw_text.startswith("[")
                cleaned = "" if is_error else clean_math_expression(raw_text)
            # Persist results so they survive the next rerun
            st.session_state[result_key] = {
                "raw": raw_text, "cleaned": cleaned,
                "engine": engine, "is_error": is_error,
            }

        # Render results from session state (survives reruns)
        res = st.session_state.get(result_key)
        if res:
            if res["is_error"]:
                st.error(f"❌ {res['raw']}")
            else:
                st.caption(f"OCR engine used: **{res['engine']}**")
                st.markdown("**Raw OCR output:**")
                st.markdown(f'<div class="ocr-box">{res["raw"] if res["raw"] else "(empty)"}</div>', unsafe_allow_html=True)
                st.markdown("**Cleaned expression:**")
                st.markdown(f'<div class="ocr-box">{res["cleaned"] if res["cleaned"] else "(empty — try a clearer image)"}</div>', unsafe_allow_html=True)
                if res["cleaned"]:
                    edited = st.text_input("Edit before loading", value=res["cleaned"], key=edit_key)
                    if st.button("Load into Calculator", key=load_key):
                        st.session_state["expr_str"] = edited
                        st.session_state[result_key] = None  # clear results after loading
                        st.rerun()

    if pil_img is not None:
        src = "cam" if st.session_state.get("ocr_camera") is not None else "up"
        _show_ocr_results(pil_img, f"run_ocr_{src}", f"ocr_edit_{src}", f"load_ocr_{src}")






# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · DIFFERENTIATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_diff:
    st.markdown('<div class="section-title">∂ Symbolic Differentiation</div>', unsafe_allow_html=True)

    diff_order = st.number_input("Derivative order", min_value=1, max_value=10, value=1, step=1, key="diff_order")

    if st.button("Calculate Derivative", key="btn_diff"):
        try:
            result = calc.differentiate(sym_expr, sym_var, int(diff_order))
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.latex(
                rf"\frac{{d^{{{int(diff_order)}}}}}{{d{var_input}^{{{int(diff_order)}}}}}"
                rf"\left({sp.latex(sym_expr)}\right) = {sp.latex(result)}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_integ:
    st.markdown('<div class="section-title">∫ Symbolic Integration</div>', unsafe_allow_html=True)

    integ_type = st.radio("Type", ["Indefinite", "Definite"], horizontal=True, key="integ_type")

    if integ_type == "Definite":
        col1, col2 = st.columns(2)
        with col1:
            lb_str = st.text_input("Lower bound", "0", key="lb")
        with col2:
            ub_str = st.text_input("Upper bound", "1", key="ub")

        if st.button("Calculate Definite Integral", key="btn_def"):
            try:
                lb = sp.sympify(lb_str)
                ub = sp.sympify(ub_str)
                result = calc.integrate(sym_expr, sym_var, lb, ub)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.latex(
                    rf"\int_{{{sp.latex(lb)}}}^{{{sp.latex(ub)}}}"
                    rf"{sp.latex(sym_expr)}\,d{var_input} = {sp.latex(result)}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if st.button("Calculate Indefinite Integral", key="btn_indef"):
            try:
                result = calc.integrate(sym_expr, sym_var)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.latex(
                    rf"\int {sp.latex(sym_expr)}\,d{var_input} = {sp.latex(result)} + C"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · LIMIT
# ─────────────────────────────────────────────────────────────────────────────
with tab_lim:
    st.markdown('<div class="section-title">lim Limit Computation</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pt_str = st.text_input("Approach point", "0", key="lim_pt")
    with col2:
        direction = st.selectbox("Direction", ["+", "-", "+-"], index=2, key="lim_dir")

    if st.button("Calculate Limit", key="btn_lim"):
        try:
            point = sp.sympify(pt_str)
            result = calc.limit(sym_expr, sym_var, point, direction)
            dir_latex = {"+" : "^+", "-": "^-", "+-": ""}.get(direction, "")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.latex(
                rf"\lim_{{{var_input} \to {sp.latex(point)}{dir_latex}}}"
                rf"\left({sp.latex(sym_expr)}\right) = {sp.latex(result)}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · TAYLOR SERIES
# ─────────────────────────────────────────────────────────────────────────────
with tab_taylor:
    st.markdown('<div class="section-title">~ Taylor Series Expansion</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        center_str = st.text_input("Expansion center", "0", key="tay_center")
    with col2:
        tay_order = st.number_input("Order n", min_value=1, max_value=20, value=6, key="tay_order")

    if st.button("Compute Series", key="btn_taylor"):
        try:
            center = sp.sympify(center_str)
            result = calc.taylor(sym_expr, sym_var, center, int(tay_order))
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.latex(
                rf"T_{{{int(tay_order)}}}\left({sp.latex(sym_expr)},\;"
                rf"{var_input}={sp.latex(center)}\right) = {sp.latex(result)}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · NUMERIC EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="section-title">= Numeric Evaluation</div>', unsafe_allow_html=True)

    sub_val_str = st.text_input(f"Value for {var_input}", "1", key="eval_val")

    if st.button("Evaluate", key="btn_eval"):
        try:
            sub_val = sp.sympify(sub_val_str)
            result = calc.evaluate(sym_expr, {sym_var: sub_val})
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.latex(
                rf"f\left({var_input}={sp.latex(sub_val)}\right) = {sp.latex(result)}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "CalcModule v2 · Powered by SymPy, Streamlit, OpenCV, Pytesseract, &amp; Three hard working psyducks."
    "</p>",
    unsafe_allow_html=True
)

