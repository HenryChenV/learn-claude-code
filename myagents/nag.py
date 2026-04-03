""" Nag System
1. Trace the tasks in session
2. Remind agent after some idle turns to avoid agent running in a wrong way
"""


from dataclasses import dataclass
from enum import Enum
from multiprocessing import Value
from multiprocessing.spawn import prepare
import stat

from attr import frozen

from .session import Session, SessionMiddleware
from .tools.core import Tool, FunctionTool


class Status(Enum):
    TODO = "todo"
    DOING = "doing"
    DONE = "done"


class Step:
    _desc: str
    _status: Status

    def __init__(self, desc: str):
        self._desc = desc
        self._status = Status.TODO

    def start(self):
        self._status = Status.DOING

    def complete(self):
        self._status = Status.DONE

    @property
    def desc(self):
        return f"{self._desc} [{self._status.name}]"


class Task:

    _name: str
    _steps: list[Step]
    _current_step_index: int
    _status: Status
    
    def __init__(self, name: str, steps: list[str]):
        if not steps:
            raise ValueError("steps must not be empty.")

        self._name = name
        self._steps = [Step(s) for s in steps]
        self._current_step_index = 0
        self._status = Status.TODO

    def start(self) -> str:
        self._steps[self._current_step_index].start()
        self._status = Status.DOING
        return self.progress

    def complete(self, step_no: int) -> str:
        if self.is_done():
            raise RuntimeError(f"Task{self._name} is already done!!!")
        if (step_no - 1) != self._current_step_index:
            raise RuntimeError(f"Wrong step no {step_no}. Current progress is {self.progress}")

        self._steps[self._current_step_index].complete()

        self._current_step_index += 1
        if self._current_step_index >= len(self._steps):
            self._status = Status.DONE
        elif self._status is Status.TODO:
            self._status = Status.DOING
        
        return self.progress

    @property
    def progress(self) -> str:
        if self._status is Status.DONE:
            return f"Task{self._name}[{len(self._steps)}/{len(self._steps)}] is Done."
        if self._status is Status.TODO:
            return f"Task{self._name}[0/{len(self._steps)}] is not started. The first step is f{self._steps[0].desc}"
        if self._status is Status.DOING:
            return f"Task{self._name}[{self._current_step_index+1}/{len(self._steps)}] is Doing. Current Step is f{self._steps[self._current_step_index]}"
        raise ValueError(f"Unknown status {self._status}")

    @property
    def detail(self) -> str:
        return "\n".join([self._name + ":"] + [f"Step{i+1}: {s.desc}" for i, s in enumerate(self._steps)])

    def is_done(self):
        return self._status is Status.DONE


class TaskManager:
    """ trace the tasks
    """

    # At most one task is allowed. Maybe support multi-task in the future
    _current_task: Task | None
    _task_tools: list[Tool] | None

    def __init__(self):
        self._current_task = None
        self._task_tools = None

    def create_task(self, task_name: str, steps: list[str]):
        if self._current_task is not None and not self._current_task.is_done():
            raise RuntimeError(f"Current Task is not Done. Progress: {self._current_task.progress}")
        self._current_task = Task(task_name, steps)
        return self._current_task.detail

    def start_task(self) -> str:
        if self._current_task is None:
            raise RuntimeError("no task yet")
        return self._current_task.start()

    def complete(self, step_no: int):
        if self._current_task is None:
            raise RuntimeError("no task yet")
        return self._current_task.complete(step_no)

    @property
    def current_task_progress(self) -> str:
        if self._current_task is None:
            return "no task yet"
        return self._current_task.progress

    @property
    def current_task_details(self) -> str:
        if self._current_task is None:
            return "no task yet"
        return self._current_task.detail

    def get_tools(self) -> list[Tool]:
        if self._task_tools is None:

            @FunctionTool.wrapper
            def create_task(task_name: str, steps: list[str]) -> str:
                """ Create task with task name and steps. Task detail will be returned.
                    Use complate_task to complete each step when it is completed.
                """
                return self.create_task(task_name, steps)

            @FunctionTool.wrapper
            def start_task() -> str:
                """start the task which means you will start the first step of the task.
                """
                return self.start_task()

            @FunctionTool.wrapper
            def complete_task(step_no: int):
                """ complete one of step of task with no of step.
                """
                return self.complete(step_no)

            @FunctionTool.wrapper
            def get_task_progress():
                """get progress of current task
                """
                return self.current_task_progress

            @FunctionTool.wrapper
            def get_task_details():
                """ get the details of task
                """
                return self.current_task_details

            self._task_tools = [
                create_task, 
                start_task,
                complete_task, 
                get_task_progress, 
                get_task_details
            ]

        return self._task_tools


class NagSystem(SessionMiddleware):

    _task_manager: TaskManager

    def __init__(self):
        self._task_manager = TaskManager()

    def post_init(self, session: Session) -> None:
        session.add_tool_provider(self._task_manager)
