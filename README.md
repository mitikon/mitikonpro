# 部分空間正則化PCA＋先行シグナル予測λ

このリポジトリは**米国株式市場向けの部分空間正則化PCA＋先行シグナル予測λ専用**です。

競馬予想システム開発は独立リポジトリ **`mitikon/racing-prediction-system`** へ分離しました。このリポジトリには競馬コードを含めません。

## 「RSI」の定義

このリポジトリで「RSI」という略称は**Recursive Self-Improvement（再帰的自己改善）**のみを指します（`recursive_self_improvement.py`の`MarketRecursiveImprovementGate`など）。相対力指数（Relative Strength Index）と混同しないよう、テクニカル指標としての相対力指数は`relative_strength_feature.py`に実装し、コード・変数名・README・テストで「RSI」の略称を使いません。

## 固定した定義

- `lambda_reg = 0.10`: クラス別共分散行列を等方的ターゲットへ縮約する正則化係数
- `variance_target = 0.90`: PCAで保持する累積説明分散率
- 未来情報を使う backward fill は禁止
- 日次確定値のみを使用し、リアルタイム取引は対象外
- 予測は発生時点で固定し、結果判明後の後付け変更は禁止

## 最初の実装範囲

- SPY、QQQ、RSP、SMH、HYG、LQD、XLY、XLPの1～5日ラグ
- SMH−QQQ、RSP−SPY、HYG−LQD、XLY−XLPの内部乖離
- 任意でVIX9D−VIX3Mと20日平均に対する出来高比
- 相対力指数（Relative Strength Index）特徴量（5・7・14・21日）：水準、3営業日変化速度、50突破、30/70圏の継続・離脱
- 時系列ウォークフォワード検証
- 取引コスト控除後の年率、最大ドローダウン、勝率、取引率、方向精度

相対力指数の70/30を固定売買ルールにはしません。各相対力指数状態はPCAへの入力特徴量であり、
翌営業日の確定結果を取り込む日次再学習によって、先行性の有無と重みを更新します。

## 現段階の重要事項

これは利益を保証する完成モデルではなく、理論定義と検証規律を固定した検証可能な第1版です。S&P 500を上回れるかは、実データの未使用期間で比較して判断します。

## テスト

```bash
python -m pytest
```

## 日次確定データの取得

無料取得アダプターを使う場合は任意依存を追加します。

```bash
pip install '.[data]'
leading-lambda-collect --start 2010-01-01 --end-exclusive 2026-09-05 --output data/raw
```

`--end-exclusive`の日付は保存されません。取引時間中の未確定バーを混入させないため、最後に確定した米国営業日の翌日を指定します。出力CSVの欠損値は意図的に残し、未来方向の補完は行いません。

## 実データ検証

```bash
leading-lambda-validate --start 2015-01-01 --output artifacts/validation
```

SPYとQQQについて、ウォークフォワード予測を時点固定CSVに保存し、同じ期間の買い持ち成績と比較します。GitHub Actionsは米国市場終了後の火～土曜日（UTC 02:15、日本時間11:15）に実行し、結果を90日間の成果物として保存します。iPadの電源状態には依存しません。

各定期実行では、取得済みCSVの最終確定日をシグナル日として、次のNYSE営業日に対するSPY・QQQの `LONG / SHORT / NO_TRADE` も `forward_signal.json` へ固定します。入力データのSHA-256、生成時刻、学習最終日、確率を記録し、同じ固定ファイルの上書きを拒否します。次回実行時には前回成果物を読み、対象日の終値が確定していれば `settled_previous_signal.json` へ結果を分離保存します。

## NYSE永久運用型カレンダー

`exchange_calendars`の`XNYS`を基準に、土日、米国祝日、夏時間、短縮取引の正式な終了時刻を判定します。検証は必ず「最後に終了した営業日」までに限定されます。

災害・追悼など通常規則では予測できない臨時休場は、`config/exceptional_nyse_closures.json`へ`YYYY-MM-DD`形式で追加します。固定日付表ではなく、ライブラリ更新と例外追記によって将来も維持する構造です。

## 保守専用RSI

市場RSIとは完全に分離した`maintenance_rsi`パッケージが、予測中核の固定値、未来補完禁止、Python構文、危険なデシリアライズ、GitHub Actionsの最小権限とSHA固定を監視します。毎日の定期監査、PR・push時の全回帰テスト、週次CodeQL、Dependabotによる依存関係更新PRを実行します。

外部市場データは`ExternalDataGuard`で許可形式・サイズ・実行形式偽装・構文・ハッシュ・マルウェア検査を通過したものだけを利用できます。スキャナー不在時も既定で遮断し、不合格データは学習へ渡さず内容ハッシュ名で隔離します。

保守専用RSIには、予測ロジックの自己変更、mainへの自動マージ、売買執行、保有資産の自動決済権限を与えません。修正は隔離ブランチとPR、全テスト、確認を経て反映します。

## RSI再帰的自己改善（Recursive Self-Improvement）直結運転

相対力指数（Relative Strength Index）とは無関係に、`MarketRecursiveImprovementGate`と`recursive_runtime`が改善候補を世代管理します。候補の設定・親世代・Gitコミットを固定し、各営業日の入力、現行予測、候補予測、候補マニフェストを結果判明前にSHA-256付きでFreezeします。その後に発生した20営業日以上の結果だけで現行版と比較し、試行記録とのハッシュ不一致は評価対象にできません。

**逐次検定（Sequential Evidence）**: 損失改善の判定は固定閾値ではなく、`sequential_loss_improvement_test`によるWald SPRT（逐次確率比検定）で行います。誤って昇格させる確率（既定5%）と誤って棄却する確率（既定10%）を明示的に制御し、ノイズによる偽陽性を抑えます。統計的に明確に劣る候補は、20営業日の全期間を待たず（最短5営業日から）早期棄却できます。ただし昇格は必ず20営業日以上の全期間評価を経てからのみ行われ、部分的な期間での昇格は一切ありません。

**並列候補探索**: `PARALLEL_CANDIDATES`（既定3）個の候補を同じ営業日・同じ現行版に対して同時に検証します。1つの候補だけが劣ると判明するたびに次の候補へ差し替えるのではなく、複数の変異方向を並行して試すことで探索を高速化します。同一サイクルで複数候補が`PROMOTION_PROPOSED`に達した場合は、損失改善が最大の1件だけを実際に採用し、他は「合格したが不採用」として正直に記録します（`parameter_promotion_applied`）。

平均損失、最大上昇ETF・最大下落ETFの選出、取引コスト控除後収益、最大ドローダウンの全条件を通過した場合だけ`PROMOTION_PROPOSED`を出し、許可されたモデル設定を次回の正式予測へ自動昇格します。不合格候補は棄却し、別の候補を結果未確認の状態から検証します。この探索はGitHub Actionsの日次処理で永久に継続します。

自律更新の対象は相対力指数の算出期間、特徴量セット、特徴量重み、特徴量ラグ、中立帯、売買判定閾値だけです。固定中核の`lambda_reg=0.10`、`variance_target=0.90`、`min_samples=60`、ソースコード、`main`ブランチ、実売買は自動変更しません。ソース更新は従来どおりPRと全テストを必要とします。

各成果物には次を保存します（並列候補ごとに1つ; 2番目以降は`_2`・`_3`をファイル名に付与）。

- `forward_signal.json`: 現行世代の翌営業日予測
- `recursive_rsi_candidate_signal.json` / `_2` / `_3`: 各並列候補の同時予測
- `recursive_rsi_state.json`: 世代、並列候補（`candidate_slots`）、事前試行、結果評価、状態ハッシュ連鎖
- `settled_recursive_rsi_candidate.json` / `_2` / `_3`: 翌営業日に確定した各候補結果

同じ市場確定日で再実行された場合は最初の予測・候補・RSI状態をそのまま継承し、再計算や二重学習を行いません。

旧スキーマ（`market-recursive-runtime-v1`、候補1つのみ）の状態ファイルは、読み込み時に自動的に`market-recursive-runtime-v2`（並列候補）へ移行します。移行直後の初回実行では、引き継いだ1候補だけを決済し、新設のスロットは履歴なしとして扱われ、その回はエラーになりません。

## 誤差分類と候補根拠ログ

`error_classification.py`が、結果判明後の外れを**抽出漏れ・過大評価・最終除外・入力欠損・市場ノイズ**の5分類に振り分けます。分類は`forward_signal`の結果照合レポート（`signal_result_report.json`の`daily.error_classification`/`cumulative.error_classification`）へ毎回記録されます。

この分類は個々の外れを直ちに恒久ルールへ変換しません。`CandidateRationale`は最低`MIN_RATIONALE_SESSIONS`件（既定5件）の既知結果に基づく説明可能な根拠（`narrative`）を要求する書き込み一回限りのログで、1件の外れへの過剰適合や説明不能な重み変更を構造的に防ぎます。人間はこのログを読んだ上で、許可されたパラメータのみを変更する`MarketRsiCandidate`を提案し、その候補は結果判明前に凍結された未来のセッションでのみ`MarketRecursiveImprovementGate`により評価されます。分類ロジック自体が変更されていないことは保守専用RSI監査で検査し、削除するとCIが失敗します。

## 独立系・市場RSI自動学習ループ

`independent-market-rsi.yml`は部分空間正則化PCA本体へ接続せず、複数の改善候補を影運転します。毎営業日の確定データから全候補の翌営業日予測を結果判明前にFreezeし、市場終了後に方向正誤、予測誤差、上昇・下落1位ETF、仮想トレード成績を採点します。20営業日以上の未来データで、方向精度・誤差・仮想収益の全ゲートを通過した候補だけを独立ループ内の次世代設定へ昇格します。

実行可否は`exchange_calendars`の`XNYS`（NYSE）営業カレンダーと臨時休場設定で判定します。土日、米国祝日、臨時休場、同一営業日の重複実行ではデータ取得・採点・学習・予測をすべてスキップし、新しい市場終了済み営業日が1日増えた場合だけLOOPを1回進めます。夏時間と短縮取引日の正式な終了時刻もカレンダーから判定します。

学習は設定状態だけを更新し、過去予測、ソースコード、`main`、部分空間正則化PCA本体、実売買には自動反映しません。日付指定の確認は保存済みsettlementを指定して実行します。

```bash
leading-lambda-independent report --date 2026-09-22 --settlement artifacts/settlement.json
```
