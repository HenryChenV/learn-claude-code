""" Reminder
"""


from typing import Any

from ..core.tool import Tool


class TaskTracker(Tool):

    def __init__(self, name: str, description: str, input_schema: dict) -> None:
        super().__init__(
            name="todo", 
            description="Update task list. Track progress on multi-step tasks.", 
            input_schema=input_schema
        )
