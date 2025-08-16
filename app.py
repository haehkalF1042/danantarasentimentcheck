import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Load model & tokenizer
# =========================
MODEL_DIR = "models/indobert_finetuned"  # ganti sesuai lokasi modelmu

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# =========================
# Preprocessing (Sastrawi + regex)
# =========================
@st.cache_resource
def init_preprocessor():
    import re

    # Coba import Sastrawi, jika tidak ada: fallback
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        sastrawi_ok = True
    except Exception:
        sastrawi_ok = False

    stemmer = None
    stopwords = set()

    if sastrawi_ok:
        # stemmer
        stemmer = StemmerFactory().create_stemmer()
        # stopwords default + slang tambahan
        sw_factory = StopWordRemoverFactory()
        stopwords = set(sw_factory.get_stop_words())
        extra_slang = {
            "yg","dg","rt","dgn","ny","d","klo","kalo","amp","biar","bikin","bilang",
            "gak","ga","krn","nya","nih","sih","si","tau","tdk","tuh","utk","ya","jd",
            "jgn","sdh","aja","n","t","nyg","hehe","pen","u","nan","loh","&amp","yah",
            "sdg","litereli","bilek"
        }
        stopwords.update(extra_slang)

    # regex patterns
    url_pat = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
    mention_pat = re.compile(r"@\w+|#\w+")
    nonalpha_pat = re.compile(r"[^a-z\s]")  # buang angka, tanda baca, simbol (setelah lower)
    multispace_pat = re.compile(r"\s+")

    return {
        "sastrawi_ok": sastrawi_ok,
        "stemmer": stemmer,
        "stopwords": stopwords,
        "url_pat": url_pat,
        "mention_pat": mention_pat,
        "nonalpha_pat": nonalpha_pat,
        "multispace_pat": multispace_pat,
    }

PRE = init_preprocessor()

def preprocess_text(text: str):
   
    if text is None:
        text = ""
    t0 = text
    x = t0.lower()                                           # 1 lower
    t1 = x
    x = PRE["url_pat"].sub(" ", x)                           # 2 no_url
    t2 = x
    x = PRE["mention_pat"].sub(" ", x)                       # 3 no_mention
    t3 = x
    x = PRE["nonalpha_pat"].sub(" ", x)                      # 4 keep alphabet only
    t4 = x

    if PRE["sastrawi_ok"]:
        tokens = [tok for tok in x.split() if tok and tok not in PRE["stopwords"]]  # 5 stopwords
        t5 = " ".join(tokens)
        x = PRE["stemmer"].stem(t5)                          # 6 stemming
        t6 = x
    else:
        # fallback jika Sastrawi belum terpasang: lewati stopword+stemming
        t5 = "(skip - Sastrawi tidak tersedia)"
        t6 = "(skip - Sastrawi tidak tersedia)"

    x = PRE["multispace_pat"].sub(" ", x).strip()            # rapikan spasi
    final_text = x

    return {
        "original": t0,
        "lower": t1,
        "no_url": t2,
        "no_mention": t3,
        "clean_alpha": t4,
        "no_stopwords": t5,
        "stemmed": t6,
        "final": final_text
    }

# =========================
# Normalisasi label -> Bahasa Indonesia
# =========================
LABEL_MAP = {
    "LABEL_1": "Netral",
    "LABEL_2": "Positif",
    "LABEL_0": "Negatif",

}

def map_label(pred_id: int) -> str:
    raw = None
    if hasattr(model.config, "id2label") and isinstance(model.config.id2label, dict):
        raw = model.config.id2label.get(pred_id) or model.config.id2label.get(str(pred_id))
    if raw is None:
        raw = f"LABEL_{pred_id}"
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    key_1b = f"LABEL_1_BASED_{pred_id + 1}"
    if key_1b in LABEL_MAP:
        return LABEL_MAP[key_1b]
    if isinstance(raw, str) and raw.isdigit() and raw in LABEL_MAP:
        return LABEL_MAP[raw]
    return str(raw)

# =========================
# UI
# =========================
st.title("Danantara Tweet Sentiment")
st.warning("Disclaimer: Hasil analisis sentimen pada website ini hanya digunakan untuk kepentingan penelitian. Hasil tidak sepenuhnya akurat dan tidak dapat dijadikan rujukan mutlak.")

if not PRE["sastrawi_ok"]:
    st.warning("Sastrawi tidak terpasang. Preprocessing akan **melewati** stopword & stemming. Install dengan: `pip install sastrawi`.")

user_input = st.text_area(
    "Masukkan teks:",
    height=150,
    placeholder="Contoh: Menurut saya Danantara itu..."
)

show_steps = st.checkbox("Tampilkan tahapan preprocessing", value=True)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Teks masih kosong.")
    else:
        steps = preprocess_text(user_input)
        text_for_model = steps["final"] if steps["final"] else user_input  # fallback jika kosong total

        # Prediksi
        inputs = tokenizer(text_for_model, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_id = int(probs.argmax())
            label_id = map_label(pred_id)

        st.subheader("Hasil:")
        st.write(f"**Teks asli:** {user_input}")
        st.write(f"**Teks setelah preprocessing:** `{steps['final']}`")
        st.write(f"**Prediksi Sentimen:** {label_id}")

        if show_steps:
            with st.expander("Detail tiap tahap"):
                st.code(
                    f"""1. lower        : {steps['lower']}
2. hapus URL    : {steps['no_url']}
3. no mention   : {steps['no_mention']}
4. hanya alfabet: {steps['clean_alpha']}
5. - stopwords  : {steps['no_stopwords']}
6. - stemming   : {steps['stemmed']}
final           : {steps['final']}""",
                    language="text"
                )
