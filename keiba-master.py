import streamlit as st
import pandas as pd
import numpy as np
import io, requests, base64, json

# --- 共通関数 ---
def safe_f(v):
    try:
        s = str(v).replace('%','').replace('倍','').strip()
        return float(s) if s not in ['-','','None'] else 0.0
    except: return 0.0

def analyze_img(api_key, img_bytes):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    p = "競馬データ画像から 馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,鬼脚ランク,騎手勝率,単回値,複回値,枠バイアス の11項目をCSVで抽出して。説明不要。"
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    data = {"contents": [{"parts": [{"text": p}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}]}
    try:
        r = requests.post(url, headers={'Content-Type':'application/json'}, data=json.dumps(data), timeout=30)
        t = r.json()['candidates'][0]['content']['parts'][0]['text']
        # GitHubのバグを回避するため、記号を文字コード(chr)で指定して除去
        t = t.replace(chr(96), "").replace("csv", "")
        return t.strip()
    except Exception as e: return f"エラー:{e}"

# --- 画面構成 ---
st.set_page_config(page_title="競馬AI Ver 4.0")
st.markdown("<h3 style='text-align:center;'>🏇 競馬AI解析 Ver 4.0</h3>", unsafe_allow_html=True)

with st.expander("⚙️ 初期設定"):
    key = st.text_input("Gemini APIキー", type="password")

st.subheader("📸 画像解析")
f = st.file_uploader("写真を選択", type=["png", "jpg", "jpeg"])
if st.button("AI解析実行") and f:
    with st.spinner("解析中..."):
        st.session_state.data = analyze_img(key, f.read())

st.subheader("📝 データ確認")
raw = st.text_area("CSVデータ", value=st.session_state.get('data',''), height=150)

if st.button("🚀 期待値計算", type="primary") and raw:
    try:
        df = pd.read_csv(io.StringIO(raw.strip()), header=None)
        df.columns = ['馬番','馬名','枠','オッズ','上がり3F','ポジ','鬼脚','騎手','単回','複回','枠B'][:len(df.columns)]
        for c in df.columns: df[c] = df[c].apply(safe_f)
        
        df['score'] = ((df['単回']-df['単回'].mean())/df['単回'].std()).fillna(0) * 1.5 + ((df['上がり3F'].mean()-df['上がり3F'])/df['上がり3F'].std()).fillna(0) * 1.0
        df['ev'] = (np.exp(df['score'])/np.exp(df['score']).sum()) * df['オッズ']
        
        for _, r in df.sort_values('ev', ascending=False).iterrows():
            c = "#d32f2f" if r['ev'] >= 1.0 else "#0056b3"
            st.markdown(f"<div style='border-left:8px solid {c}; padding:10px; margin-bottom:5px; background:white;'><b>{int(r['馬番'])} {r['馬名']}</b> (期待値: <b style='color:{c}'>{r['ev']:.2f}</b>)</div>", unsafe_allow_html=True)
    except Exception as e: st.error(f"エラー: {e}")

if st.button("🗑️ クリア"):
    st.session_state.data = ""
    st.rerun()
