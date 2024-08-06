import os
os.environ["TMPDIR"] = os.path.expanduser('~/libs/llm_finetuning/LangGraph/')

import warnings
warnings.filterwarnings("ignore")

from essay_assistant import ewriter, writer_gui

MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()
