import streamlit as st
import pandas as pd
import time
import io

# ==========================================
# 内部ロジック（計算式は旧システム+8項目のハイブリッド）
# ==========================================
def calc_hybrid_base(row):
    try:
        tan_ret = float(str(row.get('単回値', 0)).replace('%',''))
        fuku_ret = float(str(row.get('複回値', 0)).replace('%',''))
        odds = float(row.get('オッズ', 10))
        up3 = float(row.get('上がり3F順位', 10))
        pos = float(row.get('ポジション評価', 3))
        j_win = float(str(row.get('騎手勝率', 0)).replace('%',''))
        bias = float(row.get('枠バイアス(秒)', 0))
    except:
        return 0

    if tan_ret > 300: tan_ret = 80
    if fuku_ret > 300: fuku_ret = 70
    
    val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)
    spurt_bonus = 25 if (up3 <= 3.0 and pos >= 3.0) else 0
    k_rank = str(row.get('亀谷ランク', 'C')).upper()
    rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
    
    base = (100 - odds * 0.5) + (val_score * 0.3) + rank_bonus + (j_win * 0.3) + spurt_bonus
    return base - (bias * 10)

def run_v35(df):
    results = [{'馬番': int(r['馬番']), 'V35スコア': calc_hybrid_base(r)} for _, r in df.iterrows()]
    res = pd.DataFrame(results).sort_values(by='V35スコア', ascending=False).reset_index(drop=True)
    res['V35順位'] = res.index + 1
    return res

def run_v40(df, speed=16.6, length=2.4):
    results = [{'馬番': int(r['馬番']), '予測タイム': (120.0 - (calc_hybrid_base(r) * 0.1))} for _, r in df.iterrows()]
    res = pd.DataFrame(results)
    res['V40ハンデ'] = ((res['予測タイム'] - res['予測タイム'].min()) * speed) / length
    return res

def execute_fusion(df_raw, df_35, df_40):
    merged = pd.merge(pd.merge(df_raw, df_35, on='馬番'), df_40, on='馬番')
    results = []
    for _, row in merged.iterrows():
        rank, hc = row['V35順位'], row['V40ハンデ']
        if rank <= 2 and hc <= 1.0: g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単勝・複勝】厚め勝負"
        elif rank <= 4 and hc == 0.0: g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単勝】妙味狙い"
        elif rank == 1 and hc >= 5.0: g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】ヒモまで"
        elif rank > 4 and hc > 4.0: g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else: g, l, c, act = "R", "連下・ヒモ候補", "#0056b3", "【通常】相手候補"
        results.append({'馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 'V35順位': rank, 'V40ハンデ': round(hc, 1)})
    
    order = {"S": 1, "A": 2, "R": 3, "B": 4, "C": 5}
    df_f = pd.DataFrame(results)
    df_f['sort'] = df_f['判定'].map(order)
    return df_f.sort_values(by=['sort', 'V40ハンデ']).drop(columns=['sort']).reset_index(drop=True)

# ==========================================
# UI・レイアウト設計（画像レイアウトの再現）
# ==========================================
st.set_page_config(page_title="MASTER FUSION", layout="centered")

st.markdown("""
<style>
    /* 全体の背景を白基調に */
    .stApp { background-color: #f8f9fa; }
    
    /* タイトル */
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; margin-bottom: 5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
    .sub-title { text-align: center; color: #0056b3; font-weight: bold; font-size: 16px; margin-bottom: 30px; }
    
    /* 指示文エリアのヘッダー風 */
    .instruction-header { background-color: #fff0f5; border: 2px solid #0056b3; border-bottom: none; border-radius: 15px 15px 0 0; padding: 15px; text-align: center; color: #d32f2f; font-weight: bold; }
    
    /* 貼り付けエリアのヘッダー風 */
    .input-header { background-color: #e6f2ff; border: 2px solid #0056b3; border-bottom: none; border-radius: 15px 15px 0 0; padding: 15px; text-align: center; color: #d32f2f; font-weight: bold; margin-top: 30px; }
    
    /* テキストエリアのカスタマイズ（赤枠） */
    div[data-baseweb="textarea"] > div { border: 2px solid #d32f2f !important; border-radius: 0 0 15px 15px !important; background-color: #ffffff !important; }
    textarea { color: #333 !important; font-family: monospace !important; font-size: 14px !important; }

    /* Streamlitコードブロックの余白調整 */
    div.stCode { margin-bottom: 0px; }

    /* 実行ボタン（青地・赤枠） */
    div.stButton > button[kind="primary"] { background-color: #0056b3; color: white; border: 4px solid #d32f2f; border-radius: 30px; font-weight: bold; height: 60px; font-size: 18px; }
    div.stButton > button[kind="primary"]:hover { background-color: #004494; border-color: #b71c1c; color: white; }
    
    /* クリアボタン（グレー） */
    div.stButton > button[kind="secondary"] { background-color: #6c757d; color: white; border: none; border-radius: 30px; font-weight: bold; height: 60px; font-size: 18px; }
    div.stButton > button[kind="secondary"]:hover { background-color: #5a6268; color: white; }
    
    /* 結果表示カード */
    .result-card { background: #ffffff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); display: flex; align-items: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>競馬AI投資システム</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>(MASTER FUSION ハイブリッド版)</div>", unsafe_allow_html=True)

# セッション管理（クリアボタン用）
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = ""

# --- STEP 1: AI指示文エリア ---
st.markdown("<div class='instruction-header'>🔴 右上のアイコンで指示文をコピーし、AIに抽出させてください</div>", unsafe_allow_html=True)
st.code("以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\n【重要】1行目に項目名を必ず入れ、解説抜きでCSVデータのみを出力すること。", language="text")

# --- STEP 2: データ貼り付けエリア ---
st.markdown("<div class='input-header'>👀 AI抽出データ（確実な11項目）をここに貼り付け 👀</div>", unsafe_allow_html=True)
pasted_data = st.text_area("データ貼り付け", key="input_data", height=200, label_visibility="collapsed", placeholder="馬番,馬名,枠,オッズ,上がり3F順位...\n（ここにデータをペースト）")

st.write("") # スペース調整

# --- STEP 3: 実行＆クリアボタン ---
col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🚀 フュージョン解析を実行", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ 全クリア", type="secondary", use_container_width=True)

# クリア処理
if clear_btn:
    st.session_state['input_data'] = ""
    st.rerun()

# --- STEP 4: 解析と結果表示 ---
if run_btn:
    if not pasted_data or pasted_data.strip() == "":
        st.error("データが貼り付けられていません。")
    else:
        try:
            # データのクリーニング
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True)
            df_raw.columns = df_raw.columns.str.strip()
            
            # 項目名の揺れ補正
            col_map = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old, new in col_map.items():
                if old in df_raw.columns and new not in df_raw.columns:
                    df_raw.rename(columns={old: new}, inplace=True)
            
            # 解析実行
            df_35 = run_v35(df_raw)
            df_40 = run_v40(df_raw)
            df_final = execute_fusion(df_raw, df_35, df_40)
            
            st.markdown("<h3 style='text-align:center; color:#333; margin-top:30px;'>🎯 投資判定マトリクス</h3>", unsafe_allow_html=True)
            for _, row in df_final.iterrows():
                st.markdown(f"""
                <div class='result-card' style='border-left: 8px solid {row['color']};'>
                    <div style='flex:0.5; font-size:28px; color:{row['color']}; font-weight:900; text-align:center;'>{row['判定']}</div>
                    <div style='flex:2; padding-left:10px;'>
                        <span style='color:#666; font-size:12px;'>#{row['馬番']} [{row['枠']}枠]</span><br>
                        <strong style='color:#111; font-size:16px;'>{row['馬名']}</strong>
                    </div>
                    <div style='flex:1.5; font-size:13px; color:#444;'>
                        V3.5: <b style='color:#111;'>{row['V35順位']}位</b><br>
                        V4.0: <b style='color:#111;'>+{row['V40ハンデ']}身</b>
                    </div>
                    <div style='flex:2; text-align:right;'>
                        <span style='color:{row['color']}; font-weight:bold; font-size:14px;'>{row['ステータス']}</span><br>
                        <span style='display:inline-block; background:#f1f3f5; border:1px solid #ccc; padding:2px 8px; color:#333; font-size:11px; border-radius:4px; margin-top:4px;'>{row['推奨馬券']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"解析エラー: AIの出力したCSVデータを確認してください。({e})")
