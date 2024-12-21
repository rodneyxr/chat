from langchain.prompts import PromptTemplate
from commands import CommandInput
from config import JARVIS_PROMPT
from stt import llm

args = ["hey jarvis"]

def action(cmd: CommandInput):
    jarvis_prompt = PromptTemplate(template=JARVIS_PROMPT, input_variables=["text"])
    result = llm.invoke(jarvis_prompt.format(text=cmd.text))
    return result
