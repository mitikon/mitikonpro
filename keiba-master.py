import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import base64
import json

# ==========================================
# 状態管理
# ==========================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def clear_data_action():
    st.session_state.raw_data = ""

def safe_float(val, default_val=0.0):
    try:
        s = str(val).replace('%', '').strip()
        if s in ['-', 'ー', '', 'None', 'null']:
            return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# AI画像解析エンジン (Gemini API)
# ==========================================
def analyze_image_with_gemini(api_key, image_bytes):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt_text = """
    以下の競馬データ画像から、指定の11項目を抽出し、CSV形式（カンマ区切り）で出力してください。ヘッダーは不要です。
    【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,鬼脚ランク,騎手勝率,単回値,複回値,枠バイアス(秒)/m
    【絶対遵守ルール】
    1. コードブロック(```)やファイル出力は絶対に行わず、通常のテキスト文字だけで出力すること。
    2. ポジション評価は「逃げ・先行」などの文字ではなく、必ず「1〜5の数値」に変換して出力すること。
    3. 余計な挨拶や説明文は一切含めず、データのみを出力すること。
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }]
    }
    headers = {'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status() # エラーチェック
    data = response.json()
    
    # AIからの返答テキストを抽出
    extracted_text = data['candidates'][0]['content']['parts'][0]['text']
    
    # 【修正箇所】コードブロック記号が含まれてしまった場合の保険としての除去処理（改行エラーを修正しました）
    extracted_text = extracted_text.replace("
```csv", "").replace("```", "").strip()
    
    return extracted_text

# ==========================================
# コアエンジン Ver 4.0 (期待値・EV算出ロジック)
# ==========================================
def execute_ev_engine(df_raw):
    df = df_raw.copy()
    
    for col in ['オッズ', '単回値', '上がり3F順位', 'ポジション評価']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x))
            
    if df['単回値'].std() != 0:
        df['単回値_Z'] = (df['単回値'] - df['単回値'].mean()) / df['単回値'].std()
    else:
        df['単回値_Z'] = 0
        
    if df['上がり3F順位'].std() != 0:
        df['上がり_Z'] = (df['上がり3F順位'].mean() - df['上がり3F順位']) / df['上がり3F順位'].std()
    else:
        df['上がり_Z'] = 0

    df['実力スコア'] = (df['単回値_Z'] * 1.5) + (df['上がり_Z'] * 1.0)
    df['実力スコア_exp'] = np.exp(df['実力スコア'])
    df['予測勝率(%)'] = (df['実力スコア_exp'] / df['実力スコア_exp'].sum()) * 100
    df['期待値(EV)'] = (df['予測勝率(%)'] / 100) * df['オッズ']
    
    df['総合順位'] = df['期待値(EV)'].rank(ascending=False, method='min').astype(int)
    
    conditions = [
        (df['期待値(EV)'] >= 1.5),
        (df['期待値(EV)'] >= 1.1),
        (df['期待値(EV)'] >= 0.8),
        (df['期待値(EV)'] < 0.8)
    ]
    choices_g = ["S", "A", "R", "C"]
    choices_l = ["完全無欠の絶対軸", "特大のオッズバグ", "連下・ヒモ候補", "完全ノイズ"]
    choices_c = ["#d32f2f", "#ff9800", "#0056b3", "#757575"]
    choices_act = ["【単・複】厚め勝負", "【単勝】妙味狙い", "【通常】相手候補", "【消し】購入対象外"]
    
    df['判定'] = np.select(conditions, choices_g, default="C")
    df['ステータス'] = np.select(conditions, choices_l, default="完全ノイズ")
    df['color'] = np.select(conditions, choices_c, default="#757575")
    df['推奨馬券'] = np.select(conditions, choices_act, default="【消し】購入対象外")
    
    return df.sort_values('総合順位').reset_index(drop=True)

# ==========================================
# UIレイアウト
# ==========================================
st.set_page_config(page_title="競馬AI投資システム Ver 4.0", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; margin-bottom: 20px; }
    div.stButton > button[kind="primary"] { background-color: #d32f2f !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #8b0000 !important; }
    div.stButton > button[kind="secondary"] { background-color: #6c757d !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #495057 !important; }
    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 14px !important; }
    div[data-baseweb="textarea"] > div { border: 3px solid #d32f2f !important; border-radius: 8px !important; background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>競馬AI投資システム Ver 4.0</div>", unsafe_allow_html=True)

# セキュリティ対策: APIキー入力欄（パスワード形式）
with st.expander("🔑 システム設定 (初回のみ)", expanded=True):
    st.info("iPodのメモ帳に保存したAPIキーを貼り付けてください。(GitHubには保存されないため安全です)")
    api_key = st.text_input("Gemini APIキー", type="password")

st.markdown("<hr style='border:1px solid #ccc; margin: 10px 0;'>", unsafe_allow_html=True)

# 1. 画像アップロードエリア
st.markdown("<h4 style='color:#333; margin-top:0px; text-align:center;'>📸 1. 写メでデータを自動入力</h4>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("出馬表などの画像を選択", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if st.button("✨ 画像からデータをAI解読", use_container_width=True):
    if not api_key:
        st.error("上の「システム設定」にAPIキーを入力してください。")
    elif not uploaded_file:
        st.warning("画像を選択（または撮影）してください。")
    else:
        with st.spinner("AIが画像を解読中... (約5〜10秒)"):
            try:
                image_bytes = uploaded_file.read()
                result_text = analyze_image_with_gemini(api_key, image_bytes)
                if result_text:
                    st.session_state.raw_data = result_text # 解読結果をテキストエリアに流し込む
                    st.success("✅ 解読完了！下のデータ入力エリアに自動反映されました。")
            except Exception as e:
                st.error(f"解読に失敗しました。エラー詳細: {e}")

st.markdown("<h4 style='color:#0056b3; margin-top:20px; text-align:center;'>👀 2. 抽出データ (手動修正可能) 👀</h4>", unsafe_allow_html=True)

# ここが st.session_state.raw_data と連動します
pasted_data = st.text_area(
    "データ入力エリア", 
    key="raw_data", 
    height=200, 
    label_visibility="collapsed",
    placeholder="AI解読をするとここにデータが自動入力されます。\n（手動での貼り付けも可能です）"
)

st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 期待値(EV)解析を実行", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ データオールクリア", type="secondary", on_click=clear_data_action, use_container_width=True)
st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)

# 実行ボタン処理
if execute_btn:
    if not pasted_data.strip():
        st.error("データが入力されていません。")
    else:
        try:
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True, header=None)
            
            # ヘッダーがない前提（AIが出力するフォーマット）なので、強制的にカラム名を割り当てる
            if len(df_raw.columns) == 11:
                df_raw.columns = ['馬番','馬名','枠','オッズ','上がり3F順位','ポジション評価','鬼脚ランク','騎手勝率','単回値','複回値','枠バイアス(秒)/m']
            else:
                st.warning("⚠️ 読み取ったデータの項目数が11個ではありません。AIの解読ミスが含まれている可能性があります。")
                # とりあえず強引にカラム名を付ける（エラー回避）
                col_names = ['馬番','馬名','枠','オッズ','上がり3F順位','ポジション評価','鬼脚ランク','騎手勝率','単回値','複回値','枠バイアス(秒)/m']
                df_raw.columns = col_names[:len(df_raw.columns)] + [f"不明_{i}" for i in range(len(df_raw.columns) - 11)]

            df_final = execute_ev_engine(df_raw)
            
            st.markdown("<h2 style='text-align:center; color:#d32f2f;'>🎯 投資判定マトリクス</h2>", unsafe_allow_html=True)
            
            for _, row in df_final.iterrows():
                ev_color = "#d32f2f" if row['期待値(EV)'] >= 1.0 else "#0056b3"
                waku_str = row.get('枠', '-')
                
                html_block = f"""<div style='background:#fff; border-left:12px solid {row['color']}; padding:15px; border-radius:8px; margin-bottom:15px; border:2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:10px; margin-bottom:10px;'>
<div>
<span style='font-size:30px; font-weight:900; color:{row['color']};'>{row['判定']}</span>
<span style='margin-left:15px; font-size:20px; font-weight:bold; color:#111;'>#{row.get('馬番', '-')} {row.get('馬名', '不明')}</span>
<span style='margin-left:10px; font-size:14px; color:#666;'>({waku_str}枠)</span>
</div>
<div style='text-align:right;'>
<span style='color:{row['color']}; font-weight:bold; font-size:18px;'>{row['ステータス']}</span><br>
<span style='display:inline-block; background:#e9ecef; border:1px solid #ced4da; padding:4px 8px; font-size:12px; border-radius:4px; margin-top:4px; font-weight:bold;'>{row['推奨馬券']}</span>
</div>
</div>
<div style='display:flex; justify-content:space-between; font-size:15px; font-weight:bold; color:#333; background:#f8f9fa; padding:10px; border-radius:5px;'>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>🏆 総合順位<br><span style='font-size:22px; color:#d32f2f;'>{row['総合順位']}位</span></div>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>現在オッズ<br><span style='font-size:18px; color:#111;'>{row.get('オッズ', 0):.1f} 倍</span></div>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>システム勝率<br><span style='font-size:18px; color:#0056b3;'>{row['予測勝率(%)']:.1f} %</span></div>
<div style='flex:1; text-align:center;'>期待値 (EV)<br><span style='font-size:18px; color:{ev_color};'>{row['期待値(EV)']:.2f}</span></div>
</div>
</div>"""
                st.markdown(html_block, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"【エラー】データ処理中に問題が発生しました。文字化けや欠損がないか確認してください。詳細: {e}")
