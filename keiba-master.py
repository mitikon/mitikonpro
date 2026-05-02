import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components

# ==========================================
# Pro-Edition: 状態管理（確実な全クリア機構）
# ==========================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def clear_input():
    # Streamlitの仕様上、明示的に空文字を代入するのが最もバグが起きないクリア方法です
    st.session_state.raw_data = ""

# ==========================================
# Pro-Edition: コアエンジン（3.5 + 4.0 融合）
# ==========================================
def calc_hybrid_score(row):
    try:
        # ノイズ（%や空白）を完全に除去して数値化
        tan_ret = float(str(row.get('単回値', 0)).replace('%', '').strip() or 0)
        fuku_ret = float(str(row.get('複回値', 0)).replace('%', '').strip() or 0)
        odds = float(str(row.get('オッズ', 10)).strip() or 10)
        up3 = float(str(row.get('上がり3F順位', 10)).strip() or 10)
        pos = float(str(row.get('ポジション評価', 3)).strip() or 3)
        j_win = float(str(row.get('騎手勝率', 0)).replace('%', '').strip() or 0)
        bias = float(str(row.get('枠バイアス(秒)', 0)).strip() or 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
    except Exception:
        return 0

    # 3.0から継承：ノイズカット機能
    if tan_ret > 300: tan_ret = 80
    if fuku_ret > 300: fuku_ret = 70
    
    # 3.5：期待値・確率ベーススコア
    val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)
    
    # 3.0：好位差し（ゴールデンゾーン）ボーナス
    spurt_bonus = 25 if (up3 <= 3.0 and pos >= 3.0) else 0
    
    # 血統ランクボーナス
    rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
    
    # 基礎能力値の算出（枠バイアスの物理減点を含む）
    return (100 - odds * 0.5) + (val_score * 0.3) + rank_bonus + (j_win * 0.3) + spurt_bonus - (bias * 10)

def execute_master_fusion(df_raw):
    # 1. 各馬のスコアを計算
    results = []
    for _, r in df_raw.iterrows():
        results.append({'馬番': int(r['馬番']), '能力値': calc_hybrid_score(r)})
    
    df_calc = pd.DataFrame(results)
    
    # 2. V3.5（期待値順位）と V4.0（物理ハンデ）の算出
    df_calc['V35順位'] = df_calc['能力値'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['能力値'].max()
    df_calc['V40馬身'] = round(((max_score - df_calc['能力値']) * 0.1 * 16.6) / 2.4, 1)
    
    # 3. 元データと結合し、融合判定を下す
    merged = pd.merge(df_raw, df_calc, on='馬番')
    final_output = []
    
    for _, row in merged.iterrows():
        r35 = row['V35順位']
        h40 = row['V40馬身']
        
        # 究極のハイブリッド判定ロジック
        if r35 <= 2 and h40 <= 1.0:
            g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単勝・複勝】厚め勝負"
        elif r35 <= 4 and h40 == 0.0:
            g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単勝】妙味狙い"
        elif r35 == 1 and h40 >= 5.0:
            g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】ヒモまで"
        elif r35 > 6 and h40 > 3.0:
            g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else:
            g, l, c, act = "R", "連下・ヒモ候補", "#0056b3", "【通常】相手候補"
            
        final_output.append({
            '馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            'V35': r35, 'V40': h40
        })
        
    return pd.DataFrame(final_output).sort_values(by='V35').reset_index(drop=True)

# ==========================================
# Pro-Edition: UIレイアウト（旧システム完全再現）
# ==========================================
st.set_page_config(page_title="競馬AI投資システム", layout="centered")

st.markdown("""
<style>
    /* 全体背景とフォント */
    .stApp { background-color: #f8f9fa; font-family: sans-serif; }
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #0056b3; font-weight: bold; font-size: 16px; margin-bottom: 25px; }
    
    /* 指示文エリア（ピンク・青枠） */
    .instruction-box { background-color: #fff0f5; border: 2px solid #0056b3; border-radius: 15px; padding: 20px; margin-bottom: 20px; }
    .point-txt { text-align: left; font-size: 14px; font-weight: bold; color: #333; margin-bottom: 8px; line-height: 1.4; }
    .red-dot { color: #d32f2f; margin-right: 5px; }
    .blue-dot { color: #0056b3; margin-right: 5px; }

    /* データ入力エリア（水色・青枠） */
    .input-box { background-color: #e6f2ff; border: 2px solid #0056b3; border-radius: 15px; padding: 20px; margin-bottom: 25px; }
    .input-header { text-align: center; color: #d32f2f; font-weight: bold; font-size: 16px; margin-bottom: 15px; }
    
    /* テキストエリアの赤枠強制 */
    div[data-baseweb="textarea"] > div { border: 3px solid #d32f2f !important; border-radius: 8px !important; background-color: #fff !important; }

    /* 解析実行ボタン（青地・赤枠） */
    div.stButton > button[kind="primary"] { background-color: #0056b3 !important; color: white !important; border: 4px solid #d32f2f !important; border-radius: 50px !important; height: 75px !important; font-size: 22px !important; font-weight: bold !important; width: 100% !important; box-shadow: 0 4px 0 #003d7a !important; }
    div.stButton > button[kind="primary"]:active { transform: translateY(4px); box-shadow: none !important; }
    
    /* 全クリアボタン（グレー） */
    div.stButton > button[kind="secondary"] { background-color: #6c757d !important; color: white !important; border: none !important; border-radius: 50px !important; height: 75px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; box-shadow: 0 4px 0 #495057 !important; }
    div.stButton > button[kind="secondary"]:active { transform: translateY(4px); box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>競馬AI投資システム</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>(3.5 & 4.0 統合マスター版)</div>", unsafe_allow_html=True)

# --- 1. 指示文エリア ---
st.markdown("""
<div class='instruction-box'>
    <div class='point-txt'><span class='red-dot'>●</span> 【警告】抽出項目を最も正確だった「11項目」に固定しました。</div>
    <div class='point-txt'><span class='blue-dot'>●</span> 複雑なAIの推測を排除し、事実データ（展開・推進力・回収率）のみで勝負します。</div>
    <div class='point-txt'><span class='red-dot'>●</span> 下のボタンで指示文をコピーし、AIに抽出させてください。</div>
</div>
""", unsafe_allow_html=True)

# Pro仕様: Streamlitの制約を突破する純粋なHTML/JSコピーボタン
copy_button_html = """
<style>
.btn {
    background-color: #d32f2f; color: white; border: 4px solid #b71c1c; border-radius: 40px;
    padding: 16px; font-size: 18px; font-weight: bold; cursor: pointer; width: 95%; margin: 0 auto;
    box-shadow: 0 4px 0 #8b0000; display: block; font-family: sans-serif; transition: 0.1s; outline: none;
}
.btn:active { transform: translateY(4px); box-shadow: none; }
</style>
<button class="btn" onclick="copyText()">👁 AI用データ解析指示 (確実な11項目) をコピー</button>
<script>
function copyText() {
    var text = "以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)";
    var el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    alert("指示文をコピーしました。Geminiに貼り付けてください。");
}
</script>
"""
components.html(copy_button_html, height=85)

# --- 2. データ入力エリア ---
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
st.markdown("<div class='input-header'>👀 AI抽出データ（確実な11項目）をここに貼り付け 👀</div>", unsafe_allow_html=True)

# session_state と直接連動させることで「確実なクリア」を実現
pasted_data = st.text_area(
    "データ入力", 
    key="raw_data", 
    height=220, 
    label_visibility="collapsed",
    placeholder="馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\n（ここにペースト）"
)
st.markdown("</div>", unsafe_allow_html=True)

# --- 3. アクションボタン ---
col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 脈動・物理解析を実行", type="primary")
with col2:
    clear_btn = st.button("🗑️ 全クリア", type="secondary", on_click=clear_input)

# --- 4. 解析処理と結果表示 ---
if execute_btn:
    if not pasted_data.strip():
        st.error("データが貼り付けられていません。")
    else:
        try:
            # 見出し行の表記揺れや不要な空白を自動吸収するPro仕様のデータ読み込み
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True)
            df_raw.columns = df_raw.columns.str.strip()
            
            # 揺れ補正フィルター
            rename_dict = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old_col, new_col in rename_dict.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw.rename(columns={old_col: new_col}, inplace=True)

            # 解析実行
            df_final = execute_master_fusion(df_raw)
            
            st.markdown("<h3 style='text-align:center; color:#d32f2f; margin-top:25px; font-weight:900;'>🎯 投資判定マトリクス</h3>", unsafe_allow_html=True)
            
            for _, row in df_final.iterrows():
                st.markdown(f"""
                <div style='background:#fff; border-left:14px solid {row['color']}; padding:15px; border-radius:10px; margin-bottom:12px; display:flex; align-items:center; border:1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
                    <div style='flex:1.2;'>
                        <span style='font-size:32px; font-weight:900; color:{row['color']};'>{row['判定']}</span>
                        <span style='margin-left:10px; font-size:16px; font-weight:bold; color:#111;'>#{row['馬番']} {row['馬名']}</span>
                    </div>
                    <div style='flex:1; text-align:center; font-size:14px; color:#444; background:#f8f9fa; padding:8px; border-radius:6px;'>
                        V3.5 確率: <b style='color:#111;'>{row['V35']}位</b><br>
                        V4.0 物理: <b style='color:#111;'>+{row['V40']}身</b>
                    </div>
                    <div style='flex:1.2; text-align:right;'>
                        <span style='color:{row['color']}; font-weight:bold; font-size:16px;'>{row['ステータス']}</span><br>
                        <span style='display:inline-block; background:#e9ecef; border:1px solid #ced4da; padding:4px 10px; font-size:12px; border-radius:4px; margin-top:6px; color:#333; font-weight:bold;'>{row['推奨馬券']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error("【解析エラー】 データの形式が正しくありません。AIが出力した「見出し行（馬番,馬名...）」が含まれたCSVデータをそのまま貼り付けてください。")
