import gradio as gr 
import numpy as np 
import pandas as pd 
from sklearn.linear_model import Logis cRegression 
from sklearn.svm import LinearSVC 
import traceback 

model = best_model                
encoders = label_encoders         
scaler_obj = scaler               

feature_names = X.columns.tolist() 
categorical_cols = list(encoders.keys()) 
numeric_cols = [c for c in feature_names if c not in categorical_cols] 
#  
DEFINE PREDICTION THRESHOLD AND MODEL NAME 
PREDICTION_THRESHOLD = 0.35 
best_model_name = "Best Model"  

def safe_encode(le, value): 
"""Encode safely; handle unseen categories gracefully.""" 
value = str(value) 
    if value not in le.classes_: 
        le.classes_ = np.append(le.classes_, value) 
    return le.transform([value])[0] 
 
def preprocess_input(input_dict): 
    """Prepare a single-row DataFrame for predic on.""" 
    df_input = pd.DataFrame([input_dict]) 

    for col in categorical_cols: 
        le = encoders[col] 
        val = str(df_input.loc[0, col]) 
        df_input[col] = safe_encode(le, val) 
 
    for col in numeric_cols: 
        try: 
            df_input[col] = pd.to_numeric(df_input[col]) 
        except Excep on: 
            df_input[col] = 0.0 
          
    df_input = df_input[feature_names] 

    if isinstance(model, (Logis cRegression, LinearSVC)): 
        df_input[numeric_cols] = scaler_obj.transform(df_input[numeric_cols]) 
 
    return df_input 

def predict_cancer(*args): 
    try: 
        input_dict = dict(zip(feature_names, args)) 
        data_prepared = preprocess_input(input_dict) 
        if hasa r(model, "predict_proba"): 
            prob = float(model.predict_proba(data_prepared)[:, 1][0]) 
        else: 
            decision = float(model.decision_func on(data_prepared)) 
            prob = 1 / (1 + np.exp(-decision))  

        pred_class = int(prob > PREDICTION_THRESHOLD) 
        result = " No Cancer Predicted" if pred_class == 1 else " Cancer Predicted" 
        confidence = f"{prob:.1%}"
  
        if prob > 0.5: 
            risk = "⚠ Low RISK - Immediate medical consulta on recommended" 
        elif prob > 0.3: 
            risk = " MODERATE RISK - Medical screening advised" 
        elif prob > 0.15: 
            risk = "⚠ MODERATE-High RISK - Consider medical consulta on" 
        else:  
            risk = " High RISK - Regular checkups recommended" 

        threshold_info = f"(Detec on threshold: {PREDICTION_THRESHOLD:.0%})" 
 
        return result, confidence, risk, threshold_info 
 
    except Excep on as e: 
        print("\n--- ERROR IN PREDICTION ---") 
        print(traceback.format_exc()) 
        return f"Error: {str(e)}", "N/A", "N/A", "N/A" 

#   GRADIO INTERFACE SETUP 
inputs = [] 
for col in feature_names: 
    if col in categorical_cols: 
        op ons = sorted(X[col].astype(str).unique().tolist()) 
        inputs.append(gr.Dropdown(label=col, choices=op ons, value=op ons[0] if op ons else None)) 
    else: 
        val = float(X[col].median()) 
        inputs.append(gr.Number(label=col, value=val)) 

custom_css = """ 
.gradio-container { 
    font-family: 'Arial', sans-serif; 
} 
.output-class { 
    font-size: 1.2em;
    font-weight: bold; 
} 
""" 
 
app = gr.Interface( 
    fn=predict_cancer, 
    inputs=inputs, 
    outputs=[ 
        gr.Textbox(label=" Predic on Result", elem_classes="output-class"), 
        gr.Textbox(label="ℹ Threshold Info", elem_classes="output-class"), 
    ], 
    title=" Lung Cancer Predic on System", 
    description=( 
        f"<div style='padding: 15px; background-color: #f0f8ff; border-radius: 10px;'>" 
        f"<h3> Pa ent Risk Assessment Tool</h3>" 
        f"<p><b>Detec on Threshold:</b> {PREDICTION_THRESHOLD:.0%} (Opmized for high 
sensi vity)</p>" 
        f"<p><b>Note:</b> This AI model uses a <b>lower threshold 
({PREDICTION_THRESHOLD:.0%})</b> " 
        f"to maximize cancer detec on. Standard threshold is 50%.</p>" 
        f"<p><b>⚠ Important:</b> This is a screening tool, not a diagnos c device. " 
        f"Always consult healthcare professionals for medical decisions.</p>" 
        f"</div>" 
    ), 
    theme="so ", 
    flagging_mode="never", 
    css=custom_css, 
    allow_flagging="never" 
) 
 
print(f" Launching Gradio Interface...") 
print(f" 
Model: {best_model_name}") 
print(f" 
print(f" 
Detection Threshold: {PREDICTION_THRESHOLD:.0%}") 
Tip: Threshold lowered to {PREDICTION_THRESHOLD:.0%} for be er cancer detec on") 
app.launch(share=True, debug=False)
