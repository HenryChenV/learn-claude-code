""" Nag System
1. Trace the tasks in session
2. Remind agent after some idle turns to avoid agent running in a wrong way
"""


from enum import Enum
from mailbox import Message
from typing import Generator, Iterable, Optional
from typing_extensions import override

from anthropic.types import Message

from myagents.capability import Capability
from myagents.common import HumanInput

from .session import Session, SessionMiddleware, SessionMiddlewareFactory
from .tools.core import Tool, FunctionTool
from .events import Event, SystemWarnEvent


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

    def __str__(self):
        return f"{self._desc} [{self._status.name}]"

    __repr__ = __str__


class Task:

    _name: str
    _steps: list[Step]
    _current_step_index: int
    _status: Status

    _idle_steps: int
    
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
            return f"Task{self._name}[0/{len(self._steps)}] is not started. The first step is f{self._steps[0]}"
        if self._status is Status.DOING:
            return (f"Task{self._name}[{self._current_step_index+1}/{len(self._steps)}] is in progress." 
                    f"Current Step is {self._steps[self._current_step_index]}")
        raise ValueError(f"Unknown status {self._status}")

    @property
    def detail(self) -> str:
        return "\n".join([self._name + ":"] + [f"Step{i+1}: {s}" for i, s in enumerate(self._steps)])

    def is_done(self):
        return self._status is Status.DONE


class CreateTaskFailure(RuntimeError):

    _comment: str

    def __init__(self, comment: str):
        self._comment= comment

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._comment}"


class TaskManager:
    """ trace the tasks
    """

    def __init__(self, human_input: HumanInput):
        # At most one task is allowed. Maybe support multi-task in the future
        self._current_task:Optional[Task] = None
        self._task_tools: Optional[Iterable[Tool]] = None
        self._task_tool_names: Optional[list[str]] = None
        self._human_input: HumanInput = human_input

    def create_task(self, task_name: str, steps: list[str]):
        if self._current_task is not None and not self._current_task.is_done():
            raise RuntimeError(f"Current Task is not Done. Progress: {self._current_task.progress}")
        task = Task(task_name, steps)
        comment = self._human_input.input(
            f"Task: {task.detail}\n\n"
            "如果同意，请输入: 'yes/y/ok/approved' ,\n" 
            "其他输入将拒绝任务创建，并将评论发送给Agent\n "
            "Comment: "
        )

        if comment not in ("yes", "y", "ok", "approved"):
            raise CreateTaskFailure(f"用户拒绝创建. 理由: {comment}. 修复问题，然后重新创建任务.")

        self._current_task = task
        return self._current_task.detail

    def start_task(self) -> str:
        if self._current_task is None:
            raise RuntimeError("no task yet")
        return self._current_task.start()

    def complete_step(self, step_no: int):
        if self._current_task is None:
            raise RuntimeError("no task yet")
        return self._current_task.complete(step_no)

    def has_uncompleted_task(self) -> bool:
        return self._current_task is not None and not self._current_task.is_done()

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

    def get_tools(self) -> Iterable[Tool]:
        if self._task_tools is None:

            @FunctionTool.wrapper(required_capabilities=Capability.TASK_CREATE.value)
            def create_task(task_name: str, steps: list[str]) -> str:
                """ Create task with task name and steps. 
                    You have to confirme with user before creating the task.
                    Use complete_task to complete each step when it is completed.
                    If the task status is not updated for more than 3 rounds,
                    you will receive a notification from TaskTracker.
                """
                return self.create_task(task_name, steps)

            @FunctionTool.wrapper(required_capabilities=Capability.TASK_START.value)
            def start_task() -> str:
                """start the task which means you will start the first step of the task.
                """
                return self.start_task()

            @FunctionTool.wrapper(required_capabilities="task.complete")
            def complete_task(step_no: int):
                """ complete one of step of task with no of step.
                """
                return self.complete_step(step_no)

            @FunctionTool.wrapper(required_capabilities=Capability.TASK_PROGRESS_GET.value)
            def get_task_progress():
                """get progress of current task
                """
                return self.current_task_progress

            @FunctionTool.wrapper(required_capabilities=Capability.TASK_DETAILS_GET.value)
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
            self._task_tool_names = [t.meta.name for t in self._task_tools]

        return self._task_tools

    @property
    def tool_names(self) -> list[str]:
        if self._task_tool_names is None:
            self.get_tools()
        return self._task_tool_names or []


class TaskTracker(SessionMiddleware):
    """ TaskTracker to trace the prcessing of task and send remind if necessary.
    """

    _task_manager: TaskManager
    _idle_steps: int
    _max_idle_steps: int

    def __init__(self, human_input: HumanInput, max_idle_steps=3):
        self._task_manager = TaskManager(human_input)
        self._idle_steps = 0
        self._max_idle_steps = max_idle_steps

    @override
    def post_session_init(self, session: Session) -> None:
        session.add_tool_provider(self._task_manager)

    @override
    def post_agent_step(self, session: Session, resp: Message) -> bool:
        if not self._task_manager.has_uncompleted_task() or not self._task_manager.tool_names:
            # no uncompleted task, nothing to trace
            return False

        idle = True
        if resp.stop_reason == "tool_use":
            for block in resp.content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                if tool_name in self._task_manager.tool_names:
                    idle = False
                    break

        if not idle:
            self._reset_idle_steps()
            return False

        self._idle_steps += 1
        if self._idle_steps < self._max_idle_steps:
            return False

        alert = (f"TaskTracker(I'm not user, just a task tracker): "
                 f"You have uncompleted task "
                 f"and don't update the status for at least {self._max_idle_steps} rounds." 
                 f"The progress is {self._task_manager.current_task_progress}. " 
                 f"Please update the task status or explain why you cannot.")

        session.publish(SystemWarnEvent(
            paths=session.extend_paths(f"{self.__class__.__name__}", "idle_warn"), 
            source_name="TaskTracker", 
            content=alert,
            extra=session.event_extra
        ))
        session.append_user_prompt(prompt=alert, new_round=False)

        self._reset_idle_steps()

        return True

    def _reset_idle_steps(self):
        self._idle_steps = 0


class TaskTrackerFactory(SessionMiddlewareFactory):

    def __init__(self, human_input:HumanInput, max_idle_steps: int = 3):
        self._human_input: HumanInput = human_input
        self._max_idle_steps: int = max_idle_steps

    @override
    def create(self, session: Session) -> TaskTracker:
        return TaskTracker(
            self._human_input, 
            max_idle_steps=self._max_idle_steps
        )