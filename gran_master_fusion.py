import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components
import math

# ==========================================
# 1. システム設定 ＆ タイトル
# ==========================================
st.set_page_config(page_title="Gran Master Fusion v5.0", layout="centered")

# CSS: 視認性と操作性を極限まで高めたチューニング
st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    .main-title { text-align: center; color: #b71c1c; font-weight: 900; font-size: 34px; line-height: 1.1; margin-bottom: 5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
    .sub-title { text-align: center; color: #555; font-size: 16px; margin-bottom: 20px; font-weight: bold;}
    
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

    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 15px !important; }
    div[data-baseweb="textarea"] > div { border: 2px solid #b71c1c !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Gran Master Fusion</div><div class='sub-title'>- 高回収×高的中 ハイブリッド解析エンジン v5.0 -</div>", unsafe_allow_html=True)

# ==========================================
# 2. 状態管理
# ==========================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def clear_data_action():
    st.session_state.raw_data = ""

# ==========================================
# 3. ユーティリティ
# ==========================================
def safe_float(val, default_val=0.0):
    try:
        s = str(val).replace('%', '').replace('秒', '').strip()
        if s in ['-', 'ー', '', 'None', 'null', 'nan']:
            return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# 4. コアエンジン（v5.0 高回収×高的中チューニング）
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    for _, row in df_raw.iterrows():
        # データ取得とクレンジング
        tan_ret = min(safe_float(row.get('単回値', 0), 0), 300) # 上限を少し解放
        fuku_ret = min(safe_float(row.get('複回値', 0), 0), 200)
        odds = safe_float(row.get('オッズ', 10), 10)
        up3 = safe_float(row.get('上がり3F順位', 10), 10)
        
        # ポジション評価の現代競馬への最適化（前有利を基本としつつ極端な追込は減点）
        p_val = str(row.get('ポジション評価', 3)).strip()
        if '逃' in p_val: pos = 5.0
        elif '先' in p_val: pos = 4.0
        elif '差' in p_val: pos = 2.5
        elif '追' in p_val: pos = 1.0
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

        # --- v5.0 スコアリングエンジン ---
        
        # 1. 回収率ベースの期待値スコア (穴馬発掘用)
        # 単勝の比重を上げ、一発の威力を高く評価
        val_score = (tan_ret * 0.65) + (fuku_ret * 0.35) 
        
        # 2. 展開・脚質シナジー (的中率ベース)
        # 「前に行ける馬」または「速い上がりを使える差し馬」を高評価
        spurt_bonus = 0
        if up3 <= 3.0 and pos <= 3.0: 
            spurt_bonus = 25 # 確実な決め手を持つ馬
        elif pos >= 4.0:
            spurt_bonus = 18 # 展開有利な先行馬

        # 3. 信頼度ボーナス
        rank_bonus = 18 if k_rank == 'A' else 9 if k_rank == 'B' else 0
        tokuchu_bonus = 22 if tokuchu == 'A' else 10 if tokuchu == 'B' else 0
        
        # 4. オッズ・ハイブリッド補正（ここがキモ）
        odds_factor = 0
        if odds >= 12.0 and val_score >= 110:
            # 期待値が高くオッズが甘い馬に強烈なバフ（高回収率へのブースト）
            odds_factor = 25 
        elif odds <= 3.5 and (j_win >= 15.0 or tokuchu == 'A'):
            # 実力のある圧倒的人気馬はペナルティを免除し加点（高的中率の担保）
            odds_factor = 15
        elif odds > 60.0:
            # 極端な大穴はノイズになるため微減点
            odds_factor = -10 
        else:
            # 中穴〜上位人気への緩やかなオッズ補正
            odds_factor = -min(math.log10(odds + 1) * 8, 15)

        # 総合得点算出（バイアスは影響度を調整）
        v50_score = (val_score * 0.3) + (j_win * 1.5) + spurt_bonus + rank_bonus + tokuchu_bonus + odds_factor - (bias * 15)
        
        # ベースライン調整
        v50_score = max(round(v50_score + 50, 1), 0)

        results.append({
            '馬番': baban, '枠': waku, '馬名': bamei, 'オッズ': odds,
            'V50点': v50_score, '特注': tokuchu, '期待値': round(val_score, 1)
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty: return df_calc
    
    # 順位と馬身差（スコア差）の計算
    df_calc['総合順位'] = df_calc['V50点'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['V50点'].max()
    df_calc['V50馬身'] = round(((max_score - df_calc['V50点']) * 0.1 * 16.6) / 2.8, 1) # 馬身差のスケールを現代風に微調整

    final_output = []
    for _, r in df_calc.iterrows():
        rank = r['総合順位']
        hc = r['V50馬身']
        odds_val = r['オッズ']
        
        # --- 判定ロジックの極限チューニング ---
        if rank == 1 and hc == 0.0:
            if odds_val <= 4.0: g, l, c, act = "S", "鉄板級の絶対軸", "#b71c1c", "【馬連・3連複】軸指定"
            elif odds_val >= 7.0: g, l, c, act = "SS", "激アツ単勝特注", "#ff1744", "【単複】全力買い"
            else: g, l, c, act = "S", "期待値最高の1位", "#d32f2f", "【単・軸】マルチ推奨"
            
        elif rank <= 4 and odds_val >= 15.0:
            g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【ワイド・複勝】妙味極大"
            
        elif odds_val <= 3.5 and rank >= 6:
            g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【完全見送り】罠馬"
            
        elif rank > 8 and hc > 3.5:
            g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
            
        else:
            g, l, c, act = "R", "連下・相手候補", "#0056b3", "【ヒモ】相手に組み込む"
        
        final_output.append({
            '総合順位': rank, '馬番': r['馬番'], '枠': r['枠'], '馬名': r['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            'V50点': r['V50点'], 'V50馬身': hc, '特注': r['特注'], '期待値': r['期待値']
        })
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)

# ==========================================
# 5. UIレイアウト
# ==========================================

st.info("🔴 以下の指示文をコピーし、Geminiに最新統計を検索させてください。")
copy_prompt = "以下の画像を解析し、JRA過去15年の『コース・血統・脚質統計』を検索して統合CSVを作成せよ。統計的に今回有利な条件（期待値が高い条件）に合致する馬を特定し、その理由と共に12列目の『特注評価(A,B,C)』を決定すること。\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価\n\n【絶対遵守ルール】\n1. ポジション評価は必ず1〜5の数値にすること。\n2. ユーザーがワンクリックでコピーできるよう、出力データは必ず「```csv」と「
```」で囲んだコードブロック形式で出力すること。余計なテキストは除外すること。"

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

pasted_data = st.text_area("AI抽出データをペースト", key="raw_data", height=200, placeholder="馬番,馬名,枠,オッズ...")

col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 V5.0 ハイブリッド解析を実行", type="primary", use_container_width=True)
with col2:
    st.button("🗑️ データクリア", type="secondary", on_click=clear_data_action, use_container_width=True)

if execute_btn:
    if not pasted_data.strip():
        st.error("データが空です。")
    else:
        try:
            clean_data = pasted_data.replace("```csv", "").replace("```", "").strip()
            df_raw = pd.read_csv(io.StringIO(clean_data), skipinitialspace=True)
            
            df_raw.columns = [str(c).strip().replace('　', '') for c in df_raw.columns]
            rename_dict = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old_col, new_col in rename_dict.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw.rename(columns={old_col: new_col}, inplace=True)

            df_final = execute_master_fusion(df_raw)
            
            if df_final.empty:
                st.warning("有効なデータが見つかりませんでした。見出し行のフォーマットを確認してください。")
            else:
                st.markdown("<h3 style='text-align:center;'>🎯 V5.0 解析マトリクス結果</h3>", unsafe_allow_html=True)
                for _, row in df_final.iterrows():
                    # UIをV5仕様にアップデート（期待値スコアの表示追加）
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
    <div style='text-align:center;'>ベース期待値<br><b style='font-size:18px; color:#0056b3;'>{row['期待値']}</b></div>
    <div style='text-align:center;'>V50 総合点<br><b style='font-size:18px; color:#d32f2f;'>{row['V50点']}</b></div>
    <div style='text-align:center;'>着差予測<br><b style='font-size:18px; color:#757575;'>+{row['V50馬身']}</b></div>
</div>
</div>"""
                    st.markdown(html_card, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"解析エラー: データ形式が正しくない可能性があります。({e})")
