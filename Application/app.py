import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import shap
import matplotlib.pyplot as plt
from st_aggrid import GridUpdateMode
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, confusion_matrix


# マッピング情報の読み込み
@st.cache_data
def load_mapping():
    return pd.read_csv("mapping.csv")

# キャッシュクリア用の関数
def clear_mapping_cache():
    load_mapping.clear()


# タイトル
st.title("不正口座検出システム")

# 学習済みモデルのロード
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("../models/balanced_fraud_model.model")
    return model

# キャッシュをクリアして最新のmapping.csvを読み込み
clear_mapping_cache()
model = load_model()
mapping_df = load_mapping()

# カラム名を日本語に変換する関数
def rename_columns(df):
    mapping_dict = dict(zip(mapping_df['feature_name'], mapping_df['description']))
    return df.rename(columns=mapping_dict)

# SHAP値を理由コードに変換する関数
def get_reason_codes(shap_values, feature_names, top_n=5, min_contribution=0.1):
    # 正の寄与のみを取得
    positive_shap = shap_values[shap_values > 0]
    positive_features = [feature_names[i] for i in range(len(shap_values)) if shap_values[i] > 0]
    
    if len(positive_shap) == 0:
        return ["リスク要因が検出されませんでした"]
    
    # 寄与度でソート
    sorted_indices = np.argsort(positive_shap)[::-1]
    total_contribution = np.sum(positive_shap)
    
    reason_codes = []
    cumulative_contribution = 0
    
    # 理由コードのマッピング辞書を作成
    reason_mapping = dict(zip(mapping_df['feature_name'], mapping_df['reason_code']))
    
    for i, idx in enumerate(sorted_indices[:top_n]):
        contribution = positive_shap[idx]
        contribution_ratio = contribution / total_contribution
        
        if contribution_ratio >= min_contribution:
            feature_name = positive_features[idx]
            reason_code = reason_mapping.get(feature_name, feature_name)
            # SHAP値をそのまま表示（必要なら小数点以下調整）
            reason_codes.append(f"{reason_code} (AIの判断根拠スコア: {contribution:.2f}ポイント)")
            cumulative_contribution += contribution_ratio
    
    # 残りの寄与度が10%以上ある場合は「その他」として追加
    remaining = 1 - cumulative_contribution
    if remaining >= min_contribution:
        reason_codes.append("その他")
    
    return reason_codes if reason_codes else ["リスク要因が検出されませんでした"]


# サイドバーにファイル入力を配置
st.sidebar.header("口座申請情報を入力")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

@st.cache_resource
def get_explainer_and_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values



def evaluate(predictions, y_true, threshold):
    # ROC曲線描画（plot_roc関数はユーザー定義と仮定）
    fprs, tprs, _ = roc_curve(y_true, predictions)

    # ROC-AUCスコア計算
    auc = roc_auc_score(y_true, predictions)

    # 閾値によるラベル予測
    pred_labels = (predictions >= threshold).astype(int)

    # 各種評価指標計算
    acc = accuracy_score(y_true, pred_labels)
    rec = recall_score(y_true, pred_labels)
    prec = precision_score(y_true, pred_labels)

    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels).ravel()
    fpr = fp / (fp + tn)

    total = len(pred_labels)
    predicted_frauds = pred_labels.sum()
    predicted_fraud_rate = predicted_frauds / total * 100

    # 担当者がチェックする口座数の削減率計算（モデル適用前は全数チェック想定）
    pre_model_check_count = total
    post_model_check_count = predicted_frauds
    reduction_rate = 1 - (post_model_check_count / pre_model_check_count)

    # モデル適用前後の1の割合
    pre_ratio = y_true.mean()
    if post_model_check_count > 0:
        post_ratio = y_true[pred_labels == 1].mean()
    else:
        post_ratio = 0

    # 結果出力（printとstreamlit両方）
    print(f"AUC（ROC曲線の下の面積）: {auc:.4f}")
    st.write(f"AUC（ROC曲線の下の面積）: {auc:.4f}")

    print(f"再現率（fraud=1 のうち正しく検出した割合）: {rec:.4f}")
    st.write(f"再現率（fraud=1 のうち正しく検出した割合）: {rec:.4f}")

    print(f"適合率（fraud=1 と予測したうち正解した割合）: {prec:.4f}")
    st.write(f"適合率（fraud=1 と予測したうち正解した割合）: {prec:.4f}")

    print(f"偽陽性率（fraud=0 を誤って1と予測した割合）: {fpr:.4f}")
    st.write(f"偽陽性率（fraud=0 を誤って1と予測した割合）: {fpr:.4f}")

    print(f"しきい値（threshold）: {threshold}")
    st.write(f"しきい値（threshold）: {threshold}")

    print(f"正解率（Accuracy）: {acc:.4f}")
    st.write(f"正解率（Accuracy）: {acc:.4f}")

    print(f"fraudと予測した件数: {predicted_frauds} 件 / 全体 {total} 件 "
          f"（{predicted_fraud_rate:.2f}%）")
    st.write(f"fraudと予測した件数: {predicted_frauds} 件 / 全体 {total} 件 "
          f"（{predicted_fraud_rate:.2f}%）")

    print(f"担当者がチェックする口座数の削減率: {reduction_rate:.4f}")
    st.write(f"担当者がチェックする口座数の削減率: {reduction_rate:.4f}")

    print(f"モデル適用前の1の割合: {pre_ratio:.4f}")
    st.write(f"モデル適用前の1の割合: {pre_ratio:.4f}")

    print(f"モデル適用後の1の割合（予測が1のグループ内）: {post_ratio:.4f}")
    st.write(f"モデル適用後の1の割合（予測が1のグループ内）: {post_ratio:.4f}")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 推論対象を month >= 6 に絞る
    df_inference = df[df["month"] >= 6].copy()
    X_test = df_inference.drop(['fraud_bool'], axis=1)
    y_test = df_inference['fraud_bool']

    predictions = model.predict_proba(X_test)[:, 1]

    result_df = X_test.copy()

    threshold = 0.92
    
    result_df["予測確率"] = predictions
    result_df["予測ラベル"] = (predictions >= threshold).astype(int)
    result_df["正解ラベル"] = y_test.values

    #evaluate(predictions, y_test, threshold)
    
    # カラム名を日本語に変換
    result_df = rename_columns(result_df)

    st.subheader("凍結対象候補リスト")

    # サイドバーに表示件数選択を配置
    max_rows = st.sidebar.selectbox(
        "表示する件数を選択",
        options=[100, 200, 500, 1000, 5000, 100000],
        index=3
    )
    filtered_df = result_df[result_df["予測ラベル"] == 1]

    # SHAP値を計算してスタイリング用のデータフレームを作成
    explainer, shap_values = get_explainer_and_shap_values(model, X_test)
    shap_val_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # フィルタされたデータのSHAP値を取得
    filtered_indices = filtered_df.index.tolist()
    filtered_shap_indices = [X_test.index.get_loc(idx) for idx in filtered_indices]
    filtered_shap_values = shap_val_pos[filtered_shap_indices]
    
    # SHAP値に基づく色分け関数
    def style_dataframe_by_shap(df, shap_values):
        def apply_shap_colors(row):
            colors = []
            feature_columns = [col for col in df.columns if col not in ['予測確率', '予測ラベル', '正解ラベル']]
            
            for col in df.columns:
                if col in feature_columns:
                    col_idx = feature_columns.index(col)
                    row_idx = row.name
                    
                    if row_idx < len(shap_values) and col_idx < shap_values.shape[1]:
                        shap_val = shap_values[row_idx, col_idx]
                        
                        # 正規化（全体の最大絶対値で割る）
                        max_abs = np.abs(shap_values).max()
                        if max_abs > 0:
                            normalized_shap = shap_val / max_abs
                        else:
                            normalized_shap = 0
                        
                        # 色の強度を計算
                        intensity = min(abs(normalized_shap), 1) * 0.6
                        
                        if shap_val > 0:
                            # 赤系（リスク上昇）
                            colors.append(f'background-color: rgba(255, 100, 100, {intensity})')
                        elif shap_val < 0:
                            # 青系（リスク下降）
                            colors.append(f'background-color: rgba(100, 150, 255, {intensity})')
                        else:
                            colors.append('')
                    else:
                        colors.append('')
                else:
                    colors.append('')
            
            return colors
        
        return df.style.apply(apply_shap_colors, axis=1)
    
    # 表示用データとSHAP値
    display_df = filtered_df.head(max_rows).reset_index(drop=True)
    display_shap = filtered_shap_values[:len(display_df)]
    
    # セルクリック機能付きの表を実装
    if len(display_shap) > 0:
        # 凡例を表示
        st.markdown("🔴 赤系 = リスク上昇要因  |  🔵 青系 = リスク下降要因")
        
        # 色分けされた表を表示（クリック選択機能付き）
        styled_df = style_dataframe_by_shap(display_df, display_shap)
        
        # セッションステートでクリックされた行を管理
        if 'clicked_row' not in st.session_state:
            st.session_state.clicked_row = None
    
        
        # 選択用チェックボックス列を追加
        display_df_with_select = display_df.copy()
        display_df_with_select.insert(0, '選択', False)
        
        # サイドバーに行選択を配置
        st.sidebar.header("AIの判断根拠を表示")
        
        # セレクトボックスで行選択（予測確率も表示）
        row_options = []
        for i in range(len(display_df)):
            prob = display_df.iloc[i]['予測確率']
            row_options.append(f"口座番号 {i+1} ")
        
        selected_option = st.sidebar.selectbox(
            "口座を選択してください",
            options=row_options,
            index=0,  # デフォルトは1行目
            key="row_selector"
        )
        # 選択された行番号を抽出
        selected_row_number = row_options.index(selected_option)
        
        # 選択された行の情報をサイドバーに表示
        if 0 <= selected_row_number < len(display_df):
            prob = display_df.iloc[selected_row_number]['予測確率']
            pred_label = display_df.iloc[selected_row_number]['予測ラベル']
            true_label = display_df.iloc[selected_row_number]['正解ラベル']
            selected_rows_data = True
        else:
            selected_rows_data = False
        
        styled_df = style_dataframe_by_shap(display_df, display_shap)
        st.dataframe(styled_df, use_container_width=True)
        
        # 選択された行の処理
        if selected_rows_data and 0 <= selected_row_number < len(filtered_df):
            original_index = filtered_df.index[selected_row_number]
            sample_index = X_test.index.get_loc(original_index)
    else:
        st.dataframe(display_df, use_container_width=True)
        selected_rows_data = False
    
    # sample_indexのデフォルト値（選択されていない場合）
    if 'sample_index' not in locals():
        sample_index = 0
    
    if selected_rows_data:
        
        # SHAP値を計算（キャッシュされているものを使用）
        explainer, shap_values = get_explainer_and_shap_values(model, X_test)
        shap_val_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        
        # 理由コードを生成して表示
        reason_codes = get_reason_codes(shap_val_pos[sample_index], X_test.columns.tolist())
        st.subheader("AIの判断根拠")
        for i, reason in enumerate(reason_codes, 1):
            st.write(f"{i}. {reason}")
        
        # ウォーターフォール図を作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_val_pos[sample_index],
                base_values=base_value,
                data=X_test.iloc[sample_index],
                feature_names=X_test.columns
            ),
            show=False
        )
        
        #st.pyplot(fig)
        #plt.close(fig)  # メモリリークを防ぐ
        
    else:
        st.info("詳細分析を表示するには、上のチェックボックスから行を選択してください")
