import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import shap
import matplotlib.pyplot as plt
from st_aggrid import GridUpdateMode

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
            reason_codes.append(f"{reason_code} (寄与度: {contribution_ratio:.1%})")
            cumulative_contribution += contribution_ratio
    
    # 残りの寄与度が10%以上ある場合は「その他」として追加
    remaining = 1 - cumulative_contribution
    if remaining >= min_contribution:
        reason_codes.append(f"その他 (寄与度: {remaining:.1%})")
    
    return reason_codes if reason_codes else ["リスク要因が検出されませんでした"]

# CSV アップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

@st.cache_resource
def get_explainer_and_shap_values(_model, X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values



def evaluate(predictions, y_true, threshold=0.55):
    fprs, tprs, _ = roc_curve(y_true, predictions)
    plot_roc(fprs, tprs)

    auc = roc_auc_score(y_true, predictions)
    pred_labels = (predictions >= threshold).astype(int)
    acc = accuracy_score(y_true, pred_labels)

    from sklearn.metrics import recall_score, confusion_matrix
    rec = recall_score(y_true, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_labels).ravel()
    fpr = fp / (fp + tn)

    total = len(pred_labels)
    predicted_frauds = pred_labels.sum()
    predicted_fraud_rate = predicted_frauds / total * 100

    print(f"AUC（ROC曲線の下の面積）: {auc:.4f}")
    st.write(f"AUC（ROC曲線の下の面積）: {auc:.4f}")
    print(f"再現率（fraud=1 のうち正しく検出した割合）: {rec:.4f}")
    st.write(f"再現率（fraud=1 のうち正しく検出した割合）: {rec:.4f}")
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


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 推論対象を month >= 6 に絞る
    df_inference = df[df["month"] >= 6].copy()
    X_test = df_inference.drop(['fraud_bool'], axis=1)
    y_test = df_inference['fraud_bool']

    predictions = model.predict_proba(X_test)[:, 1]

    result_df = X_test.copy()
    result_df["予測確率"] = predictions
    result_df["予測ラベル"] = (predictions >= 0.5).astype(int)
    result_df["正解ラベル"] = y_test.values
    
    # カラム名を日本語に変換
    result_df = rename_columns(result_df)

    st.subheader("凍結対象候補リスト")

    max_rows = st.selectbox(
    "表示する件数を選択",
    options=[100, 200, 500, 1000, 5000, 100000],
    index=5
)
    filtered_df = result_df[result_df["予測ラベル"] == 1]

    # AgGridの設定
    filtered_df_reset = filtered_df.reset_index()

    gb = GridOptionsBuilder.from_dataframe(filtered_df_reset.head(max_rows))
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_column("予測確率", sortable=True, sort="desc")  # ← ここでソート可能＆初期降順指定
    
    grid_options = gb.build()
    
    grid_response = AgGrid(
        filtered_df_reset.head(max_rows),
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        enable_enterprise_modules=False,
        theme='streamlit',
        height=300,
    )
    
    selected_rows = grid_response.selected_rows
    
    # デフォルトの sample_index
    sample_index = 0
    
    if selected_rows is not None and not selected_rows.empty:
        selected_row = selected_rows.iloc[0]
        original_index = selected_row['index']  # 元のDataFrameのインデックス
    
        # 元のインデックスから連番に変換
        sample_index = X_test.index.get_loc(original_index)
    
        # st.write(f"選択された行番号: {original_index}")

        st.markdown("""
        - 予測値が **0.5以上** なら「怪しい」です。
        - 予測値が **0.5未満** なら「安全」です。
        - 下のプロットは特徴量ごとの影響を示しており、どの要素が判定に大きく影響したかが分かります。
        """)
    
        # 選択された場合のみウォーターフォールプロット描画
        explainer, shap_values = get_explainer_and_shap_values(model, X_test)
        shap_val_pos = shap_values[1] if isinstance(shap_values, list) else shap_values
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
        # 理由コードを生成して表示
        reason_codes = get_reason_codes(shap_val_pos[sample_index], X_test.columns.tolist())
        st.subheader("検出理由")
        for i, reason in enumerate(reason_codes, 1):
            st.write(f"{i}. {reason}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_val_pos[sample_index],
                base_values=base_value,
                data=X_test.iloc[sample_index],
                feature_names=X_test.columns
            ),
            show=False
        )
        st.pyplot(fig)
    
    else:
        st.write("行が選択してください")
        # 何も表示しない（図も表示しない）
