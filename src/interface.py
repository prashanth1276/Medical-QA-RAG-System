import gradio as gr
from typing import Optional

def create_gradio_interface(rag_pipeline):
    """Medical QA interface with session memory"""
    with gr.Blocks(title="Clinical Guidelines QA") as interface:
        gr.Markdown("## Ask About Medical Guidelines")
        
        with gr.Row():
            query = gr.Textbox(label="Your Question", placeholder="e.g. What's the ICD code for...")
            submit = gr.Button("Ask")
        
        output = gr.Textbox(label="Answer", interactive=False)
        
        examples = gr.Examples(
            examples=[
                ["ICD-10 code for major depression"],
                ["Treatment guidelines for bipolar disorder"],
                ["Diagnostic criteria for PTSD"]
            ],
            inputs=query
        )
        
        submit.click(
            fn=rag_pipeline.answer_query,
            inputs=query,
            outputs=output
        )
    
    return interface