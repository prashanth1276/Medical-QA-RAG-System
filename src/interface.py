import gradio as gr
from typing import Optional

def create_gradio_interface(rag_pipeline):
    """Medical QA interface with session memory"""
    with gr.Blocks(title="Clinical Guidelines QnA System") as interface:
        gr.Markdown("## 🩺 Medical QnA RAG System")

        with gr.Row(equal_height=True):
            query = gr.Textbox(
                label="Your Question",
                placeholder="What's your Question...",
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
                ["Give me the correct coded classification for the following diagnosis: 'Recurrent depressive disorder, currently in remission' "],
                ["Give me the correct coded classification for the following diagnosis: 'Severe depressive episode with psychotic symptoms'"],
                ["Give me the correct coded classification for the following diagnosis: 'Generalized anxiety disorder with panic attacks'"]
            ],
            inputs=query
        )
        
        submit.click(
            fn=rag_pipeline.answer_query,
            inputs=query,
            outputs=output
        )

    return interface
