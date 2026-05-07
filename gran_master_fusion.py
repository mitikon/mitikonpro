import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components
import math

# ==========================================
# 1. システム設定 ＆ タイトル
# ==========================================
st.set_page_config(page_title="Gran Master Fusion v4.2", layout="centered")

# CSS: 視認性と操作性を極限まで高めたチューニング
st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    .main-title { text-align: center; color: #b71c1c; font-weight: 900; font-size: 32px; line-height: 1.1; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #555; font-size: 16px; margin-bottom: 20px; }
    
    /* ボタン装飾：実行は赤、クリアはグレー、コピーはゴールド */
    div.stButton > button[kind="primary"] { 
        background-color: #b71c1c !important; color: white !important; 
        border-radius: 12px !important; height: 75px !important; 
        font-size: 22px !important; font-weight: bold !important; 
        width: 100% !important; border: none !important;
        box-shadow: 0 4px #821010; transition: 0.2s;
    }
    div.stButton > button[kind="primary"]:active { box-shadow: 0 0px #821010; transform: translateY(4px); }
    
    div.stButton > button[kind="secondary"] { 
        background-color: #454d55 !important; color: white !important; 
        border-radius: 12px !important; height: 75px !important; 
        font-size: 20px !important; width: 100% !important; border: none !important;
    }

    /* 入力エリア：視認性重視（漆黒文字・純白背景） */
    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 15px !important; }
    div[data-baseweb="textarea"] > div { border: 2px solid #b71c1c !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Gran Master Fusion</div><div class='sub-title'>- 物理・統計・AI統合解析システム v4.2 -</div>", unsafe_allow_html=True)

# ==========================================
# 2. 状態管理
# ==========================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def clear_data_action():
    st.session_state.raw_data = ""

# ==========================================
# 3. ユーティリティ（防弾仕様のデータ変換）
# ==========================================
def safe_float(val, default_val=0.0):
    try:
        s = str(val).replace('%', '').strip()
        if s in ['-', 'ー', '', 'None', 'null', 'nan']:
            return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# 4. コアエンジン（徹底チューニング版）
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    for _, row in df_raw.iterrows():
        # 回収率の「断崖絶壁」を排除し、クリッピング（上限設定）に変更
        tan_ret = min(safe_float(row.get('単回値', 0), 0), 250)
        fuku_ret = min(safe_float(row.get('複回値', 0), 0), 180)
        
        odds = safe_float(row.get('オッズ', 10), 10)
        up3 = safe_float(row.get('上がり3F順位', 10), 10)
        
        # ポジション評価の数値化
        p_val = str(row.get('ポジション評価', 3)).strip()
        if '逃' in p_val: pos = 4.5
        elif '先' in p_val: pos = 5.0
        elif '差' in p_val: pos = 3.0
        elif '追' in p_val: pos = 1.5
        else: pos = safe_float(p_val, 3.0)
            
        j_win = safe_float(row.get('騎手勝率', 0), 0)
        bias = safe_float(row.get('枠バイアス(秒)', 0), 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
        tokuchu = str(row.get('特注評価', 'C')).upper().strip()
        
        baban_val = safe_float(row.get('馬番', 0), 0)
        if baban_val == 0: continue
        baban = int(baban_val)
        bamei = str(row.get('馬名', '不明')).strip()
        waku = int(safe_float(row.get('枠', 0), 0))

        # オッズによるペナルティを「線形」から「非線形」へ変更
        odds_penalty = min(math.log10(odds + 1) * 15, 25) 
        
        # ボーナス算出
        val_score = (tan_ret * 0.4) + (fuku_ret * 0.6)
        spurt_bonus = 20 if (up3 <= 3.0 and pos >= 3.0) else 0
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        tokuchu_bonus = 18 if tokuchu == 'A' else 7 if tokuchu == 'B' else 0
        
        # 【算出1】旧評価点（基礎力）
        old_score = (100 - odds_penalty) + (val_score * 0.25) + rank_bonus + (j_win * 0.4) + spurt_bonus + tokuchu_bonus
        
        # 【算出2】V35評価（枠バイアス補正）
        v35_score = old_score - (bias * 12)
        
        results.append({
            '馬番': baban, '枠': waku, '馬名': bamei, 'オッズ': odds,
            '旧評価点': round(old_score, 1), 'V35点': round(v35_score, 1), '特注': tokuchu
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty: return df_calc
    
    # 【算出3】物理ハンデ（V40馬身）
    df_calc['総合順位'] = df_calc['V35点'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['V35点'].max()
    df_calc['V40馬身'] = round(((max_score - df_calc['V35点']) * 0.1 * 16.6) / 2.4, 1)

    # 判定ロジックの論理矛盾を解消
    final_output = []
    for _, r in df_calc.iterrows():
        rank = r['総合順位']
        hc = r['V40馬身']
        odds_val = r['オッズ']
        
        # 判定アルゴリズム：的中率×回収率のハイブリッド
        if rank == 1 and hc == 0.0:
            if odds_val <= 3.5: g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単・複】厚め勝負"
            else: g, l, c, act = "S", "期待値最高の1位", "#d32f2f", "【単勝】一本釣り"
        elif rank <= 4 and odds_val >= 12.0:
            g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単勝】妙味の極み"
        elif odds_val <= 4.0 and rank >= 5:
            g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】軽視妥当"
        elif rank > 7 and hc > 4.0:
            g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else:
            g, l, c, act = "R", "連下・相手候補", "#0056b3", "【通常】ヒモ穴注意"
        
        final_output.append({
            '総合順位': rank, '馬番': r['馬番'], '枠': r['枠'], '馬名': r['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            '旧評価点': r['旧評価点'], 'V35点': r['V35点'], 'V40馬身': hc, '特注': r['特注']
        })
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)

# ==========================================
# 5. UIレイアウト
# ==========================================

# --- 指示文コピー ---
st.info("🔴 以下の指示文をコピーし、Geminiに最新統計を検索させてください。")
# コードブロック（CSV）での出力を強制し、Geminiのデフォルト「コピー」ボタンを出現させる
copy_prompt = "以下の画像を解析し、JRA過去15年の『コース・血統・脚質統計』を検索して統合CSVを作成せよ。統計的に今回有利な条件（期待値が高い条件）に合致する馬を特定し、その理由と共に12列目の『特注評価(A,B,C)』を決定すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価\n\n【絶対遵守ルール】\n1. ポジション評価は必ず1〜5の数値にすること。\n2. ユーザーがワンクリックでコピーできるよう、出力データは必ず「```csv」と「```」で囲んだコードブロック形式で出力すること。余計な解説はCSVの外に書くこと。"

copy_html = f"""
<button onclick="copyText()" style="background-color:#d32f2f; color:white; border:none; border-radius:15px; padding:15px; font-size:18px; font-weight:bold; width:100%; cursor:pointer; box-shadow: 0 4px #8b0000;">
📋 AI用・精密統計解析指示をコピー
</button>
<script>
function copyText() {{
    var text = {repr(copy_prompt)};
    var el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    alert("コピー完了。Geminiに貼り付けて解析を開始してください。");
}}
</script>
"""
components.html(copy_html, height=85)

# --- 入力エリア ---
pasted_data = st.text_area("AI抽出データをペースト", key="raw_data", height=200, placeholder="馬番,馬名,枠,オッズ...")

col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 厳重チューニング解析を実行", type="primary", use_container_width=True)
with col2:
    st.button("🗑️ データクリア", type="secondary", on_click=clear_data_action, use_container_width=True)

# --- 解析結果 ---
if execute_btn:
    if not pasted_data.strip():
        st.error("データが空です。")
    else:
        try:
            # コードブロック記号（```csvや```）が混入していても無視して読み込めるようにクリーニング
            clean_data = pasted_data.replace("```csv", "").replace("```", "").strip()
            df_raw = pd.read_csv(io.StringIO(clean_data), skipinitialspace=True)
            df_final = execute_master_fusion(df_raw)
            
            if df_final.empty:
                st.warning("有効なデータが見つかりませんでした。")
            else:
                st.markdown("<h3 style='text-align:center;'>🎯 解析マトリクス結果</h3>", unsafe_allow_html=True)
                for _, row in df_final.iterrows():
                    html_card = f"""<div style='background:#fff; border-left:15px solid {row['color']}; padding:15px; border-radius:12px; margin-bottom:15px; box-shadow: 0 4px 10px rgba(0,0,0,0.08); border:1px solid #eee;'>
<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>
    <div>
        <span style='font-size:32px; font-weight:900; color:{row['color']};'>{row['判定']}</span>
        <span style='margin-left:15px; font-size:22px; font-weight:bold;'>#{row['馬番']} {row['馬名']}</span>
        <span style='background:#333; color:#fff; padding:2px 8px; border-radius:5px; font-size:12px; margin-left:10px;'>特注: {row['特注']}</span>
    </div>
    <div style='text-align:right;'>
        <div style='color:{row['color']}; font-weight:bold; font-size:16px;'>{row['ステータス']}</div>
        <div style='font-size:13px; color:#666;'>{row['推奨馬券']}</div>
    </div>
</div>
<div style='display:flex; justify-content:space-around; background:#f9f9f9; padding:10px; border-radius:8px;'>
    <div style='text-align:center;'>順位<br><b style='font-size:20px; color:#d32f2f;'>{row['総合順位']}</b></div>
    <div style='text-align:center;'>V35点<br><b style='font-size:18px;'>{row['V35点']}</b></div>
    <div style='text-align:center;'>V40馬身<br><b style='font-size:18px;'>+{row['V40馬身']}</b></div>
</div>
</div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"解析エラー: {e}")
