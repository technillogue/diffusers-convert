import csv
from datetime import datetime
import os
from typing import Optional
import gradio as gr

from convert import convert
from huggingface_hub import HfApi, Repository


DATASET_REPO_URL = "https://huggingface.co/datasets/safetensors/conversions"
DATA_FILENAME = "data.csv"
DATA_FILE = os.path.join("data", DATA_FILENAME)

HF_TOKEN = os.environ.get("HF_TOKEN")

repo: Optional[Repository] = None
if HF_TOKEN:
    repo = Repository(local_dir="data", clone_from=DATASET_REPO_URL, token=HF_TOKEN)


def run(token: str, model_id: str) -> str:
    if token == "" or model_id == "":
        return """
        ### Invalid input 🐞
        
        Please fill a token and model_id.
        """
    try:
        api = HfApi(token=token)
        is_private = api.model_info(repo_id=model_id).private
        print("is_private", is_private)

        commit_info = convert(api=api, model_id=model_id, force=True)
        print("[commit_info]", commit_info)

        # save in a (public) dataset:
        if repo is not None and not is_private:
            repo.git_pull(rebase=True)
            print("pulled")
            with open(DATA_FILE, "a") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=["model_id", "pr_url", "time"]
                )
                writer.writerow(
                    {
                        "model_id": model_id,
                        "pr_url": commit_info.pr_url,
                        "time": str(datetime.now()),
                    }
                )
            commit_url = repo.push_to_hub()
            print("[dataset]", commit_url)

        return f"""
        ### Success 🔥

        Yay! This model was successfully converted and a PR was open using your token, here:

        [{commit_info.pr_url}]({commit_info.pr_url})
        """
    except Exception as e:
        return f"""
        ### Error 😢😢😢
        
        {e}
        """


DESCRIPTION = """
The steps are the following:

- Paste a read-access token from hf.co/settings/tokens. Read access is enough given that we will open a PR against the source repo.
- Input a model id from the Hub
- Click "Submit"
- That's it! You'll get feedback if it works or not, and if it worked, you'll get the URL of the opened PR 🔥

⚠️ For now only `pytorch_model.bin` files are supported but we'll extend in the future.
"""

demo = gr.Interface(
    title="Convert any model to Safetensors and open a PR",
    description=DESCRIPTION,
    allow_flagging="never",
    article="Check out the [Safetensors repo on GitHub](https://github.com/huggingface/safetensors)",
    inputs=[
        gr.Text(max_lines=1, label="your_hf_token"),
        gr.Text(max_lines=1, label="model_id"),
    ],
    outputs=[gr.Markdown(label="output")],
    fn=run,
)

demo.launch()
