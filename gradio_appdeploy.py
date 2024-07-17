import gradio as gr
import os
from transformers import TextClassificationPipeline, BertTokenizerFast, TFBertForSequenceClassification

HF_TOKEN = os.getenv('HF_TOKEN')
hf_writer = gr.HuggingFaceDatasetSaver(HF_TOKEN, "Tolerblanc/Demo_Kurse_detection")


loaded_tokenizer = BertTokenizerFast.from_pretrained('Tolerblanc/klue-bert-finetuned')
loaded_model = TFBertForSequenceClassification.from_pretrained('Tolerblanc/klue-bert-finetuned', output_attentions=True)

text_classifier = TextClassificationPipeline(
    tokenizer=loaded_tokenizer, 
    model=loaded_model, 
    framework='tf',
    device=0
)

def inference(text):
    output = text_classifier(text)[0]
    if output['label'] == 'LABEL_1':
        return "curse", output['score'] * 100 
    else:
        return "clean", output['score'] * 100

demo = gr.Interface(
    fn=inference, 
    inputs=gr.Textbox(lines=1, placeholder="이곳에 혐오표현을 탐지할 문장을 넣어보세요."), 
    outputs=[gr.Textbox(label="Label"), gr.Textbox(label="score")], 
    allow_flagging="manual", 
    flagging_options=["Wrong Label", "Too Low Score", "Nice Label"],
    flagging_callback=hf_writer
)

demo.launch()