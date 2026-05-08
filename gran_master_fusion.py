import streamlit as st
import pandas as pd
import io
import streamlit.components.v1 as components

# ==========================================
# 1. システム設定 ＆ タイトルアップデート
# ==========================================
# ページタイトルを「Gran Master Fusion」に設定
st.set_page_config(page_title="Gran Master Fusion", layout="centered")

# CSS: UIの見た目を整える
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-title { text-align: center; color: #d32f2f; font-weight: 900; font-size: 28px; line-height: 1.2; }
    
    /* 巨大な実行ボタンとクリアボタンの装飾 */
    div.stButton > button[kind="primary"] { background-color: #d32f2f !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #8b0000 !important; }
    div.stButton > button[kind="secondary"] { background-color: #6c757d !important; color: white !important; border-radius: 10px !important; height: 70px !important; font-size: 20px !important; font-weight: bold !important; width: 100% !important; border: 3px solid #495057 !important; }
    
    /* テキストエリアの文字色を真っ黒にし、背景を白に固定（防弾仕様） */
    textarea { color: #000000 !important; background-color: #ffffff !important; font-weight: bold !important; font-size: 14px !important; }
    div[data-baseweb="textarea"] > div { border: 3px solid #d32f2f !important; border-radius: 8px !important; background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# アプリメインタイトル
st.markdown("<div class='main-title'>Gran Master Fusion <br><span style='font-size:16px; color:#666;'>- 競馬AI投資システム -</span></div>", unsafe_allow_html=True)

# ==========================================
# 2. 状態管理（確実なデータオールクリア機構）
# ==========================================
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = ""

def clear_data_action():
    # テキストエリアの中身を強制的に空文字にリセット
    st.session_state.raw_data = ""

# ==========================================
# 3. ユーティリティ（どんな記号でも数字に変える防弾胃袋）
# ==========================================
def safe_float(val, default_val=0.0):
    try:
        # %や空白を消す
        s = str(val).replace('%', '').strip()
        # ハイフンや空欄ならデフォルト値（0や10など）を返す
        if s in ['-', 'ー', '', 'None', 'null']:
            return default_val
        return float(s)
    except:
        return default_val

# ==========================================
# 4. コアエンジン（特注評価ボーナス実装）
# ==========================================
def execute_master_fusion(df_raw):
    results = []
    for _, row in df_raw.iterrows():
        # 【修正】safe_floatを使って、ハイフンが来ても絶対にエラーにしない
        tan_ret = safe_float(row.get('単回値', 0), 0)
        fuku_ret = safe_float(row.get('複回値', 0), 0)
        odds = safe_float(row.get('オッズ', 10), 10)
        up3 = safe_float(row.get('上がり3F順位', 10), 10) # データ無し(-)は10位扱い
        
        p_val = str(row.get('ポジション評価', 3)).strip()
        if p_val == '逃げ': pos = 4.0
        elif p_val == '先行': pos = 5.0
        elif p_val == '差し': pos = 3.0
        elif p_val in ['追込', '追い込み']: pos = 1.0
        elif p_val in ['-', '']: pos = 3.0 # データ無し(-)は平均の3扱い
        else: pos = safe_float(p_val, 3.0)
            
        j_win = safe_float(row.get('騎手勝率', 0), 0)
        bias = safe_float(row.get('枠バイアス(秒)', 0), 0)
        k_rank = str(row.get('亀谷ランク', 'C')).upper().strip()
        
        # 【新規追加】第12の項目「特注評価」の読み込み
        tokuchu = str(row.get('特注評価', 'C')).upper().strip()
        
        # 馬番の取得とエラーチェック
        baban_val = safe_float(row.get('馬番', 0), 0)
        if baban_val == 0: continue # 馬番が取得できない行はデータ無しと見なしてスキップ
        baban = int(baban_val)
        
        bamei = str(row.get('馬名', '不明')).strip()
        waku = int(safe_float(row.get('枠', 0), 0))

        # DNA: 旧システムのノイズカット
        if tan_ret > 300: tan_ret = 80
        if fuku_ret > 300: fuku_ret = 70
        
        # DNA: 評価ボーナス算出
        val_score = (tan_ret * 0.5) + (fuku_ret * 0.5)
        spurt_bonus = 25 if (up3 <= 3.0 and pos >= 3.0) else 0
        rank_bonus = 15 if k_rank == 'A' else 10 if k_rank == 'B' else 5 if k_rank == 'C' else 0
        
        # 【新規追加】特注ボーナス（A評価なら+15点で中穴を一気に引き上げる）
        tokuchu_bonus = 15 if tokuchu == 'A' else 5 if tokuchu == 'B' else 0
        
        # 【算出1】旧システム評価 ＋ 特注ボーナス（バイアス抜きの基礎力）
        old_score = (100 - odds * 0.5) + (val_score * 0.3) + rank_bonus + (j_win * 0.3) + spurt_bonus + tokuchu_bonus
        
        # 【算出2】システム3.5評価（枠バイアスを組み込んだ総合期待値）
        v35_score = old_score - (bias * 10)
        
        results.append({
            '馬番': baban, '枠': waku, '馬名': bamei,
            '旧評価点': round(old_score, 1),
            'V35点': round(v35_score, 1),
            '特注': tokuchu # 表示用
        })

    df_calc = pd.DataFrame(results)
    if df_calc.empty:
        return df_calc
    
    # 【算出3】総合順位とシステム4.0（馬身差）
    df_calc['総合順位'] = df_calc['V35点'].rank(ascending=False, method='min').astype(int)
    max_score = df_calc['V35点'].max()
    # 物理ハンデ算出式（スコア差を秒数、さらに馬身へ換算）
    df_calc['V40馬身'] = round(((max_score - df_calc['V35点']) * 0.1 * 16.6) / 2.4, 1)

    # 最終判定ロジック
    final_output = []
    for _, r in df_calc.iterrows():
        rank = r['総合順位']
        hc = r['V40馬身']
        
        if rank <= 2 and hc <= 1.0: g, l, c, act = "S", "完全無欠の絶対軸", "#d32f2f", "【単・複】厚め勝負"
        elif rank <= 4 and hc == 0.0: g, l, c, act = "A", "特大のオッズバグ", "#ff9800", "【単勝】妙味狙い"
        elif rank == 1 and hc >= 5.0: g, l, c, act = "B", "危険な人気馬", "#9c27b0", "【見送り】ヒモまで"
        elif rank > 6 and hc > 3.0: g, l, c, act = "C", "完全ノイズ", "#757575", "【消し】購入対象外"
        else: g, l, c, act = "R", "連下・相手候補", "#0056b3", "【通常】相手候補"
        
        final_output.append({
            '総合順位': rank, '馬番': r['馬番'], '枠': r['枠'], '馬名': r['馬名'], 
            '判定': g, 'ステータス': l, '推奨馬券': act, 'color': c, 
            '旧評価点': r['旧評価点'], 'V35点': r['V35点'], 'V40馬身': hc, '特注': r['特注']
        })
        
    return pd.DataFrame(final_output).sort_values(by='総合順位').reset_index(drop=True)

# ==========================================
# 5. UIレイアウト ＆ 実行
# ==========================================

# --- A. 巨大コピーボタン ---
st.info("🔴 以下のボタンで指示文をコピーし、AI（Gemini）に送信してください。")
# 【新規追加】指示文を12項目（特注評価あり）にアップデート
copy_html = """
<button onclick="copyText()" style="background-color:#d32f2f; color:white; border:4px solid #b71c1c; border-radius:30px; padding:15px; font-size:18px; font-weight:bold; width:100%; cursor:pointer; box-shadow: 0 4px 0 #8b0000;">
👁 AI用データ解析指示 (特注含む12項目) をコピー
</button>
<script>
function copyText() {
    var text = "以下の画像を解析し統合CSVを作成せよ。JRA統計から『枠バイアス(秒)』を独自算出すること。さらに、対象レースの過去15年の傾向（コース適性や血統）を考慮し、独自の定性評価を『特注評価(A,B,C)』として12列目に追加せよ。\\n【必須項目】馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価\\n\\n【絶対遵守ルール】\\n1. コードブロック(```)やファイル出力は絶対に行わず、通常のテキスト文字だけで出力すること。\\n2. ポジション評価は「逃げ・先行」などの文字ではなく、必ず「1〜5の数値」に変換して出力すること。";
    var el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    alert("コピーしました。Geminiの入力欄に貼り付けて送信してください。");
}
</script>
"""
components.html(copy_html, height=80)

# --- B. データ貼り付けエリア ---
st.markdown("<h4 style='color:#0056b3; margin-top:10px; text-align:center;'>👀 AI抽出データをここに貼り付け 👀</h4>", unsafe_allow_html=True)

# セッション状態と連動する確実な入力欄
pasted_data = st.text_area(
    "データ入力エリア", 
    key="raw_data", 
    height=200, 
    label_visibility="collapsed",
    placeholder="馬番,馬名,枠,オッズ,上がり3F順位,ポジション評価,亀谷ランク,騎手勝率,単回値,複回値,枠バイアス(秒),特注評価\n（ここにペースト）"
)

# --- C. 実行 ＆ オールクリアボタン ---
st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    execute_btn = st.button("🚀 脈動・物理解析を実行", type="primary", use_container_width=True)
with col2:
    # 確実に動くデータ消去ボタン
    clear_btn = st.button("🗑️ データオールクリア", type="secondary", on_click=clear_data_action, use_container_width=True)
st.markdown("<hr style='border:1px solid #ccc; margin: 15px 0;'>", unsafe_allow_html=True)

# --- D. 解析実行とフルデータ開示 ---
if execute_btn:
    if not pasted_data.strip():
        st.error("データが貼り付けられていません。")
    else:
        try:
            # データのクリーニング（空白除去、見出しの揺れ補正）
            df_raw = pd.read_csv(io.StringIO(pasted_data.strip()), skipinitialspace=True)
            df_raw.columns = [str(c).strip().replace('　', '') for c in df_raw.columns]
            
            rename_dict = {'枠バイアス': '枠バイアス(秒)', '上がり順位': '上がり3F順位', 'ポジション': 'ポジション評価'}
            for old_col, new_col in rename_dict.items():
                if old_col in df_raw.columns and new_col not in df_raw.columns:
                    df_raw.rename(columns={old_col: new_col}, inplace=True)

            # 解析実行
            df_final = execute_master_fusion(df_raw)
            
            if df_final.empty:
                st.error("有効なデータが計算できませんでした。（1行目の見出しや、馬番が含まれているか確認してください）")
            else:
                st.markdown("<h2 style='text-align:center; color:#d32f2f;'>🎯 投資判定マトリクス</h2>", unsafe_allow_html=True)
                
                # 判定結果カードの生成
                for _, row in df_final.iterrows():
                    # Markdownのコードブロック誤認を防ぐため、HTMLのインデントを詰めて出力
                    html_block = f"""<div style='background:#fff; border-left:12px solid {row['color']}; padding:15px; border-radius:8px; margin-bottom:15px; border:2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
<div style='display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #eee; padding-bottom:10px; margin-bottom:10px;'>
<div>
<span style='font-size:30px; font-weight:900; color:{row['color']};'>{row['判定']}</span>
<span style='margin-left:15px; font-size:20px; font-weight:bold; color:#111;'>#{row['馬番']} {row['馬名']}</span>
<span style='margin-left:10px; font-size:14px; color:#666;'>({row['枠']}枠)</span>
<span style='margin-left:15px; font-size:14px; color:#fff; background-color:#333; padding:2px 8px; border-radius:4px;'>特注: {row['特注']}</span>
</div>
<div style='text-align:right;'>
<span style='color:{row['color']}; font-weight:bold; font-size:18px;'>{row['ステータス']}</span><br>
<span style='display:inline-block; background:#e9ecef; border:1px solid #ced4da; padding:4px 8px; font-size:12px; border-radius:4px; margin-top:4px; font-weight:bold;'>{row['推奨馬券']}</span>
</div>
</div>
<div style='display:flex; justify-content:space-between; font-size:15px; font-weight:bold; color:#333; background:#f8f9fa; padding:10px; border-radius:5px;'>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>🏆 総合順位<br><span style='font-size:22px; color:#d32f2f;'>{row['総合順位']}位</span></div>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>旧システム評価<br><span style='font-size:18px; color:#0056b3;'>{row['旧評価点']} 点</span></div>
<div style='flex:1; text-align:center; border-right:1px solid #ccc;'>システム 3.5<br><span style='font-size:18px; color:#0056b3;'>{row['V35点']} 点</span></div>
<div style='flex:1; text-align:center;'>システム 4.0<br><span style='font-size:18px; color:#d32f2f;'>+{row['V40馬身']} 身</span></div>
</div>
</div>"""
                    st.markdown(html_block, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"【エラー】データの解析に失敗しました。AIの出力が正しい形式（CSV）か確認してください。 (詳細: {e})")
