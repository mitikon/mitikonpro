import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components
import math

# ==========================================
# 1. システム設定 ＆ タイトル
# ==========================================
st.set_page_config(page_title="Gran Master Fusion v5.1", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    .main-title { text-align: center; color: #b71c1c; font-weight: 900; font-size: 34px; line-height: 1.1; margin-bottom: 5px; }
    .sub-title { text-align: center; color: #555; font-size: 16px; margin-bottom: 20px; font-weight: bold;}
    
    div.stButton > button[kind="primary"] { 
        background-color: #b71c1c !important; color: white !important; 
        border-radius: 12px !important; height: 70px !important; 
        font-size: 20px !important; font-weight: bold !important; 
        width: 100% !important; border: none !important;
        box-shadow: 0 4px #821010; transition: 0.2s;
    }
    div.stButton > button[kind="secondary"] { 
        background-color: #454d55 !important; color: white !important; 
        border-radius: 8px !important; height: 50px !important; 
        font-size: 16px !important; width: 100% !important; border: none !important;
    }
    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 15px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Gran Master Fusion</div><div class='sub-title'>- データ抽出強化版 v5.1 -</div>", unsafe_allow_html=True)

# ==========================================
# 2. 状態管理 ＆ サンプルデータ
# ==========================================
SAMPLE_CSV = """馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価
1,サンプルスター,1,2.5,1,5,A,18.5,120,95,0.0,A
2,アナババクシン,2,15.8,3,4,B,8.2,150,110,-0.1,B
3,ニゲキリキング,3,45.2,10,5,C,5.0,80,70,0.2,C"""

if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def set_sample_data():
    st.session_state.raw_data = SAMPLE_CSV

def clear_data_action():
    st.session_state.raw_data = ""

# ==========================================
# 3. ユーティリティ
# ==========================================
def safe_float(val, default_val=0.0):
    try:
        if pd.isna(val): return default_val
        s = str(val).replace('%', '').replace('秒', '').strip()
        if s in ['-', 'ー', '', 'None', 'null', 'nan']: return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# 4. コアエンジン (抽出エラー対策強化)
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    # カラム名の名寄せ（Geminiの出力揺れに対応）
    col_map = {c: c.strip().replace(' ', '') for c in df_raw.columns}
    df_raw = df_raw.rename(columns=col_map)
    
    # 必須カラムの別名対応
    name_fix = {
        '馬番': ['馬番', '番', '枠番'],
        '馬名': ['馬名', '名前'],
        'オッズ': ['オッズ', '単勝オッズ']
    }
    for target, aliases in name_fix.items():
        if target not in df_raw.columns:
            for alias in aliases:
                if alias in df_raw.columns:
                    df_raw.rename(columns={alias: target}, inplace=True)
                    break

    for _, row in df_raw.iterrows():
        baban = int(safe_float(row.get('馬番', 0)))
        bamei = str(row.get('馬名', '不明')).strip()
        if baban == 0 or bamei == '不明': continue # 馬番・馬名がない行はスキップ

        # スコア計算 (v5.0ロジック)
        tan_ret = min(safe_float(row.get('単回値', 0)), 300)
        fuku_ret = min(safe_float(row.get('複回値', 0)), 200)
        odds = safe_float(row.get('オッズ', 10), 10)
        up3 = safe_float(row.get('上がり3F順位', 10), 10)
        p_val = str(row.get('ポジション評価', 3)).strip()
        pos = 5.0 if '逃' in p_val else 4.0 if '先' in p_val else 2.5 if '差' in p_val else 1.0 if '追' in p_val else safe_float(p_val, 3.0)
        j_win = safe_float(row.get('騎手勝率', 0), 0)
        bias = safe_float(row.get('枠バイアス(秒)', 0), 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
        tokuchu = str(row.get('特注評価', 'C')).upper().strip()

        val_score = (tan_ret * 0.65) + (fuku_ret * 0.35) 
        spurt_bonus = 25 if (up3 <= 3.0 and pos <= 3.0) else 18 if pos >= 4.0 else 0
        rank_bonus = 18 if k_rank == 'A' else 9 if k_rank == 'B' else 0
        tokuchu_bonus = 22 if tokuchu == 'A' else 10 if tokuchu == 'B' else 0
        
        odds_factor = 25 if (odds >= 12.0 and val_score >= 110) else 15 if (odds <= 3.5 and (j_win >= 15.0 or tokuchu == 'A')) else -min(math.log10(odds + 1) * 8, 15)
        
        v50_score = (val_score * 0.3) + (j_win * 1.5) + spurt_bonus + rank_bonus + tokuchu_bonus + odds_factor - (bias * 15)
        v50_score = max(round(v50_score + 50, 1), 0)

        results.append({
            '馬番': baban, '枠': int(safe_float(row.get('枠', 0))), '馬名': bamei, 'オッズ': odds,
            'V50点': v50_score, '特注': tokuchu, '期待値': round(val_score, 1)
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty: return df_calc
    
    df_calc['総合順位'] = df_calc['V50点'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['V50点'].max()
    df_calc['V50馬身'] = round(((max_score - df_calc['V50点']) * 0.1 * 16.6) / 2.8, 1)

    final_output = []
    for _, r in df_calc.iterrows():
        rank, hc, odds_val = r['総合順位'], r['V50馬身'], r['オッズ']
        if rank == 1 and hc == 0.0:
            g, l, c, act = ("S", "鉄板級の絶対軸", "#b71c1c", "【馬連・3連複】軸指定") if odds_val <= 4.0 else ("SS", "激アツ単勝特注", "#ff1744", "【単複】全力買い")
        elif rank <= 4 and odds_val >= 15.0: g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【ワイド・複勝】妙味極大"
        elif odds_val <= 3.5 and rank >= 6: g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【完全見送り】罠馬"
        elif rank > 8 and hc > 3.5: g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else: g, l, c, act = "R", "連下・相手候補", "#0056b3", "【ヒモ】相手に組み込む"
        
        r_dict = r.to_dict()
        r_dict.update({'判定': g, 'ステータス': l, '推奨馬券': act, 'color': c})
        final_output.append(r_dict)
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)

# ==========================================
# 5. UIレイアウト
# ==========================================

# 指示コピーボタン
copy_prompt = "以下の画像を解析し、JRA過去15年の『コース・血統・脚質統計』を検索して統合CSVを作成せよ。統計的に今回有利な条件（期待値が高い条件）に合致する馬を特定し、その理由と共に12列目の『特注評価(A,B,C)』を決定すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価\n\n【絶対遵守ルール】\n1. ポジション評価は必ず1〜5の数値にすること。\n2. ユーザーがワンクリックでコピーできるよう、出力データは必ず「```csv」と「```」で囲んだコードブロック形式で出力すること。"

copy_html = f"""
<button onclick="copyText()" style="background-color:#d32f2f; color:white; border:none; border-radius:10px; padding:12px; font-size:16px; font-weight:bold; width:100%; cursor:pointer;">
📋 解析指示をコピー
</button>
<script>
function copyText() {{
    var text = {repr(copy_prompt)};
    var el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    alert("コピー完了。Geminiに貼り付けてください。");
}}
</script>
"""
components.html(copy_html, height=65)

# 入力エリア
pasted_data = st.text_area("AI抽出データをペースト", key="raw_data", height=180, placeholder="ここにデータを貼り付けるか、下のボタンでサンプルを入力してください")

col_a, col_b, col_c = st.columns(3)
with col_a:
    execute_btn = st.button("🚀 解析実行", type="primary", use_container_width=True)
with col_b:
    st.button("📝 サンプル入力", type="secondary", on_click=set_sample_data, use_container_width=True)
with col_c:
    st.button("🗑️ クリア", type="secondary", on_click=clear_data_action, use_container_width=True)

if execute_btn:
    if not pasted_data.strip():
        st.error("データが空です。Geminiからコピーしたデータを貼り付けてください。")
    else:
        try:
            clean_str = pasted_data.replace("```csv", "").replace("```", "").strip()
            df_raw = pd.read_csv(io.StringIO(clean_str), skipinitialspace=True)
            df_final = execute_master_fusion(df_raw)
            
            if df_final.empty:
                st.warning("有効な馬データ（馬番・馬名）が見つかりません。見出し行を確認してください。")
            else:
                st.markdown("<h3 style='text-align:center;'>🎯 解析結果</h3>", unsafe_allow_html=True)
                for _, row in df_final.iterrows():
                    html_card = f"""<div style='background:#fff; border-left:15px solid {row['color']}; padding:15px; border-radius:12px; margin-bottom:15px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);'>
<div style='display:flex; justify-content:space-between; align-items:center;'>
    <div>
        <span style='font-size:28px; font-weight:900; color:{row['color']};'>{row['判定']}</span>
        <span style='margin-left:15px; font-size:20px; font-weight:bold;'>#{int(row['馬番'])} {row['馬名']}</span>
    </div>
    <div style='text-align:right;'>
        <div style='color:{row['color']}; font-weight:bold; font-size:14px;'>{row['ステータス']}</div>
        <div style='font-size:12px; color:#666;'>{row['推奨馬券']}</div>
    </div>
</div>
<div style='display:flex; justify-content:space-around; background:#f9f9f9; padding:10px; border-radius:8px; margin-top:10px;'>
    <div style='text-align:center;'>順位<br><b>{row['総合順位']}</b></div>
    <div style='text-align:center;'>期待値<br><b>{row['期待値']}</b></div>
    <div style='text-align:center;'>V50点<br><b style='color:#d32f2f;'>{row['V50点']}</b></div>
    <div style='text-align:center;'>着差<br><b>+{row['V50馬身']}</b></div>
</div>
</div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"読み込みエラー: {e}")
