import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components

# ==========================================
# 内部エンジン（3.5 + 4.0 融合・防弾仕様）
# ==========================================
def calc_hybrid_score(row):
    # エラーで止まらないための初期値
    tan_ret, fuku_ret, odds, up3, pos, j_win, bias = 0.0, 0.0, 10.0, 10.0, 3.0, 0.0, 0.0
    k_rank = 'C'
    
    try:
        if '単回値' in row: tan_ret = float(str(row['単回値']).replace('%', '').strip() or 0)
        if '複回値' in row: fuku_ret = float(str(row['複回値']).replace('%', '').strip() or 0)
        if 'オッズ' in row: odds = float(str(row['オッズ']).strip() or 10)
        if '上がり3F順位' in row: up3 = float(str(row['上がり3F順位']).strip() or 10)
        
        # ポジションの文字化け対策
        if 'ポジション評価' in row:
            p_val = str(row['ポジション評価']).strip()
            if p_val == '逃げ': pos = 4.0
            elif p_val == '先行': pos = 5.0
            elif p_val == '差し': pos = 3.0
            elif p_val in ['追込', '追い込み']: pos = 1.0
            else: pos = float(p_val)
            
        if '騎手勝率' in row: j_win = float(str(row['騎手勝率']).replace('%', '').strip() or 0)
        if '枠バイアス(秒)' in row: bias = float(str(row['枠バイアス(秒)']).strip() or 0)
        if '亀谷ランク' in row: k_rank = str(row['亀谷ランク']).upper().strip()
    except Exception:
        pass # 個別の変換エラーは無視して続行

    # DNA: ノイズカット
    if tan_ret > 300: tan_ret = 80
    if fuku_ret > 300: fuku_ret = 70
    
    val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)
    spurt_bonus = 25 if (up3 <= 3.0 and pos >= 3.0) else 0
    rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
    
    return (100 - odds * 0.5) + (val_score * 0.3) + rank_bonus + (j_win * 0.3) + spurt_bonus - (bias * 10)

def execute_master_fusion(df_raw):
    results = [{'馬番': int(r['馬番']), '能力値': calc_hybrid_score(r)} for _, r in df_raw.iterrows()]
    df_calc = pd.DataFrame(results)
    
    df_calc['V35順位'] = df_calc['能力値'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['能力値'].max()
    df_calc['V40馬身'] = round(((max_score - df_calc['能力値']) * 0.1 * 16.6) / 2.4, 1)
    
    merged = pd.merge(df_raw, df_calc, on='馬番')
    final_output = []
    
    for _, row in merged.iterrows():
        r35, h40 = row['V35順位'], row['V40馬身']
        if r35 <= 2 and h40 <= 1.0: g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単勝・複勝】厚め勝負"
        elif r35 <= 4 and h40 == 0.0: g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単勝】妙味狙い"
        elif r35 == 1 and h40 >= 5.0: g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】ヒモまで"
        elif r35 > 6 and h40 > 3.0: g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else: g, l, c, act = "R", "連下・ヒモ候補", "#0056b3", "【通常】相手候補"
            
        final_output.append({'馬番': row['馬番'], '枠': row['枠'], '馬名': row['馬名'], '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 'V35': r35, 'V40': h40})
        
    return pd.DataFrame(final_output).sort_values(by='V35').reset_index(drop=True)

# ==========================================
# UIレイアウト（バグ排除の超シンプル仕様）
# ==========================================
st.set_page_config(page_title="競馬AI投資システム", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; }
    div.stButton > button[kind="primary"] { background-color: #d32f2f; color: white; border-radius: 10px; height: 60px; font-size: 20px; font-weight: bold; width: 100%; margin-top: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>競馬AI投資システム</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#0056b3; font-weight:bold;'>MASTER FUSION 安定稼働版</p>", unsafe_allow_html=True)

# 1. 巨大コピーボタン（指示文を超強化）
st.info("🔴 以下のボタンで指示文をコピーし、AIに送信してください。")
copy_html = """
<button onclick="copyText()" style="background-color:#d32f2f; color:white; border:none; border-radius:30px; padding:15px; font-size:18px; font-weight:bold; width:100%; cursor:pointer;">
👁 AI用データ解析指示 をコピー
</button>
<script>
function copyText() {
    var text = "以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』も独自算出すること。\\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒)\\n\\n【絶対遵守ルール】\\n1. コードブロック(
```)やファイル出力は絶対に行わず、通常のテキスト文字だけで出力すること。\\n2. ポジション評価は「逃げ・先行」などの文字ではなく、必ず「1〜5の数値」に変換して出力すること。";
    var el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    alert("コピーしました。Geminiの入力欄に貼り付けて送信してください。");
}
</script>
"""
components.html(copy_html, height=70)

# 2. データ貼り付けエリア（バグの原因だった装飾を全削除）
st.markdown("<h4 style='color:#0056b3; margin-top:20px;'>👀 AIが書き出したデータをここに貼り付け</h4>", unsafe_allow_html=True)
pasted_data = st.text_area("データ入力エリア", height=250, label_visibility="collapsed")

# 3. 解析実行
if st.button("🚀 解析を実行する", type="primary"):
    if not pasted_data.strip():
        st.error("データが貼り付けられていません。")
    else:
        try:
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True)
            df_raw.columns = df_raw.columns.str.strip()
            
            # 見出しの揺れ補正
            rename_dict = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old_col, new_col in rename_dict.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw.rename(columns={old_col: new_col}, inplace=True)

            df_final = execute_master_fusion(df_raw)
            
            st.markdown("<h3 style='text-align:center; color:#d32f2f; margin-top:30px;'>🎯 投資判定マトリクス</h3>", unsafe_allow_html=True)
            for _, row in df_final.iterrows():
                st.markdown(f"""
                <div style='background:#fff; border-left:10px solid {row['color']}; padding:15px; border-radius:8px; margin-bottom:10px; display:flex; align-items:center; border:1px solid #ccc;'>
                    <div style='flex:1;'>
                        <span style='font-size:28px; font-weight:bold; color:{row['color']};'>{row['判定']}</span>
                        <span style='margin-left:10px; font-weight:bold;'>#{row['馬番']} {row['馬名']}</span>
                    </div>
                    <div style='flex:1; text-align:center; font-size:14px;'>
                        V3.5: <b>{row['V35']}位</b> / V4.0: <b>+{row['V40']}身</b>
                    </div>
                    <div style='flex:1; text-align:right;'>
                        <span style='color:{row['color']}; font-weight:bold;'>{row['ステータス']}</span><br>
                        <span style='font-size:12px; background:#eee; padding:2px 6px;'>{row['推奨馬券']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error("データの読み込みに失敗しました。1行目に「馬番,馬名...」などの項目名が含まれているか確認してください。")
