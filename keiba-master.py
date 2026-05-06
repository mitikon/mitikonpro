import streamlit as st
import pandas as pd
import numpy as np
import io
import streamlit.components.v1 as components

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
# UIレイアウト (改修前のデザインを完全踏襲)
# ==========================================
st.set_page_config(page_title="競馬AI投資システム Ver 4.0", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; }
    div.stButton > button[kind="primary"] { background-color: #d32f2f !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #8b0000 !important; }
    div.stButton > button[kind="secondary"] { background-color: #6c757d !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #495057 !important; }
    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 14px !important; }
    div[data-baseweb="textarea"] > div { border: 3px solid #d32f2f !important; border-radius: 8px !important; background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>競馬AI投資システム Ver 4.0</div>", unsafe_allow_html=True)

st.info("🔴 以下のボタンで指示文をコピーし、AIに送信してください。")
copy_html = """
<button onclick="copyText()" style="background-color:#d32f2f; color:white; border:4px solid #b71c1c; border-radius:30px; padding:15px; font-size:18px; font-weight:bold; width:100%; cursor:pointer; box-shadow: 0 4px 0 #8b0000;">
👁 AI用データ解析指示 (11項目) をコピー
</button>
<script>
function copyText() {
    var text = "以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\\n\\n【絶対遵守ルール】\\n1. コードブロック(```)やファイル出力は絶対に行わず、通常のテキスト文字だけで出力すること。\\n2. ポジション評価は「逃げ・先行」などの文字ではなく、必ず「1〜5の数値」に変換して出力すること。";
    var el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    alert("コピーしました。Geminiの入力欄に貼り付けて送信してください。");
}
</script>
"""
components.html(copy_html, height=80)

st.markdown("<h4 style='color:#0056b3; margin-top:10px; text-align:center;'>👀 AI抽出データをここに貼り付け 👀</h4>", unsafe_allow_html=True)

pasted_data = st.text_area(
    "データ入力エリア", 
    key="raw_data", 
    height=200, 
    label_visibility="collapsed",
    placeholder="馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\\n（ここにペースト）"
)

st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 期待値(EV)解析を実行", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ データオールクリア", type="secondary", on_click=clear_data_action, use_container_width=True)
st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)

if execute_btn:
    if not pasted_data.strip():
        st.error("データが貼り付けられていません。")
    else:
        try:
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True)
            df_raw.columns = [str(c).strip().replace('　', '') for c in df_raw.columns]
            
            rename_dict = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old_col, new_col in rename_dict.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw.rename(columns={old_col: new_col}, inplace=True)

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
            st.error("【エラー】AIが出力したデータに「馬番,馬名...」の見出しが含まれているか確認してください。")
