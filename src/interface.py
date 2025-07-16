import gradio as gr
from typing import Optional

def create_gradio_interface(rag_pipeline):
    """Medical QA interface with session memory"""
    with gr.Blocks(title="Clinical Guidelines QA") as interface:
        gr.Markdown("## 🩺 Medical QnA RAG System")

        with gr.Row(equal_height=True):
            query = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What's the ICD code for...",
                lines=2,
                scale=5
            )
            submit = gr.Button("🔍 Ask", size="sm", variant="primary", scale=1)
        
        output = gr.Textbox(
            label="Answer",
            interactive=False,
            lines=4,
            show_copy_button=True
        )
        
        gr.Markdown("### 💡 Example Questions:")
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
