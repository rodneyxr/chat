from langchain.prompts import PromptTemplate

from stt.stt import llm
from stt.commands import CommandInput
from stt.config import JARVIS_PROMPT

args = ["hey jarvis"]


def action(cmd: CommandInput):
    jarvis_prompt = PromptTemplate(template=JARVIS_PROMPT, input_variables=["text"])
    result = llm.invoke(jarvis_prompt.format(text=cmd.text))
    return result
