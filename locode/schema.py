from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

class AgentAction(BaseModel):
    thought: str = Field(..., description="Your thought process and reasoning for the next step.")
    action: Literal["tool", "finish"] = Field(..., description="The type of action to take. Use 'tool' to execute a command or file operation. Use 'finish' when the task is complete or to answer a question.")
    tool_name: Optional[str] = Field(None, description="The name of the tool to use. Required if action is 'tool'.")
    tool_args: Optional[Dict[str, Any]] = Field(None, description="The arguments for the tool. Required if action is 'tool'.")
    final_answer: Optional[str] = Field(None, description="The final response to the user. Required if action is 'finish'.")
