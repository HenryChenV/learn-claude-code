"""
Context Compaction Module

上下文压缩模块，负责管理对话历史的大小，防止 token 超出限制。

核心设计：
1. 分层压缩：
   - L1: 大输出持久化（超过阈值时写入磁盘，返回预览）
   - L2: 微压缩（保留最近 N 个 tool_result，其余写入文件并替换为占位符）
   - L3: 完整压缩（生成摘要，保存 transcript，重置对话）

2. 职责分离：
   - ContextManager: 负责统计、决策、执行压缩
   - ContextCompactorMiddleware: 负责与 Session 交互，触发压缩

3. 统计与决策：
   - CompactStats: 实时统计（context 百分比、未压缩的 tool_result 数量）
   - CompactState: 历史记录（压缩次数、压缩收益）
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from anthropic.types import MessageParam, Usage

from myagents.chat_model import ChatModelManager
from myagents.session import Session, SessionMiddleware, SessionMiddlewareFactory
from myagents.tools.core import Tool, FunctionTool


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONTEXT_PERCENTAGE_THRESHOLD = 0.80  # 触发压缩的 context 百分比阈值
DEFAULT_KEEP_RECENT_TOOL_RESULTS = 3          # micro_compact 时保留最近的 tool_result 数量
DEFAULT_LARGE_OUTPUT_THRESHOLD = 3000         # 大输出阈值，超过此值才持久化
DEFAULT_PREVIEW_CHARS = 500                   # 预览字符数（头尾各取 preview_chars）
DEFAULT_MIN_OUTPUT_CHARS_FOR_COMPACT = 100    # 只有超过此值的 output 才计入 uncaptured_tool_results
DEFAULT_SUMMARY_MODEL = "MiniMax/MiniMax-M2.7"
DEFAULT_SUMMARY_MAX_TOKENS_RATIO = 0.20       # summary_max_tokens = context_window * ratio


# =============================================================================
# State Classes
# =============================================================================

@dataclass
class CompactState:
    """
    压缩状态，历史记录。
    
    Attributes:
        last_compact_time: 上次压缩的时间
        compact_count: 累计压缩次数
        last_compact_ratio: 上次压缩的收益（context 减少的百分比）
    """
    last_compact_time: Optional[datetime] = None
    compact_count: int = 0
    last_compact_ratio: float = 0.0


@dataclass
class CompactStats:
    """
    实时统计信息，用于压缩决策。
    
    Attributes:
        uncaptured_tool_results: 累计未压缩的 tool_result 数量（每次 micro_compact 后重置）
        context_percentage: 当前 context 占 context_window 的百分比
        recent_tool_names: 最近使用的工具名称列表（用于调试和决策）
    """
    uncaptured_tool_results: int = 0
    context_percentage: float = 0.0
    recent_tool_names: list[str] = field(default_factory=list)


# =============================================================================
# ContextManager
# =============================================================================

class ContextManager:
    """
    上下文压缩管理器。
    
    职责：
    1. 大输出持久化 + 预览生成（may_persist_large_output）
    2. 实时统计（CompactStats）：通过 track_tool_result 和 refresh_stats 更新
    3. 压缩决策：should_full_compact 用于外部判断
    4. 执行压缩：
       - micro_compact: 保留最近 N 个 tool_result，其余写入文件并替换为占位符
       - full_compact: 生成对话摘要，保存 transcript，返回压缩后的 messages
    
    Attributes:
        context_percentage_threshold: 触发 full_compact 的 context 百分比阈值
        keep_recent_tool_results: micro_compact 时保留最近的 tool_result 数量
        large_output_threshold: 大输出阈值，超过此值才持久化
        preview_chars: 预览字符数（头尾各取 preview_chars）
        min_output_chars_for_compact: 只有超过此值的 output 才计入 uncaptured_tool_results
        large_output_dir: 大输出文件存储目录
        transcript_dir: transcript 文件存储目录
        summary_model: 摘要生成使用的模型
        summary_max_tokens_ratio: summary_max_tokens = context_window * ratio
    """

    def __init__(
        self,
        context_percentage_threshold: float = DEFAULT_CONTEXT_PERCENTAGE_THRESHOLD,
        keep_recent_tool_results: int = DEFAULT_KEEP_RECENT_TOOL_RESULTS,
        large_output_threshold: int = DEFAULT_LARGE_OUTPUT_THRESHOLD,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
        min_output_chars_for_compact: int = DEFAULT_MIN_OUTPUT_CHARS_FOR_COMPACT,
        large_output_dir: Path = None,
        transcript_dir: Path = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        summary_max_tokens_ratio: float = DEFAULT_SUMMARY_MAX_TOKENS_RATIO,
    ):
        self._context_percentage_threshold: float = context_percentage_threshold
        self._keep_recent_tool_results: int = keep_recent_tool_results
        self._large_output_threshold: int = large_output_threshold
        self._preview_chars: int = preview_chars
        self._min_output_chars_for_compact: int = min_output_chars_for_compact
        self._large_output_dir: Path = large_output_dir
        self._transcript_dir: Path = transcript_dir
        self._summary_model: str = summary_model
        self._summary_max_tokens_ratio: float = summary_max_tokens_ratio

        self._state = CompactState()
        self._stats = CompactStats()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> CompactState:
        """压缩状态（历史记录）"""
        return self._state

    @property
    def stats(self) -> CompactStats:
        """实时统计（当前状态）"""
        return self._stats

    # =========================================================================
    # Persistence
    # =========================================================================

    def may_persist_output(self, tool_use_id: str, output: str) -> tuple[str, bool]:
        """
        大输出持久化。
        
        如果 output 长度超过 large_output_threshold：
          1. 将完整内容写入 large_output_dir/{tool_use_id}.txt
          2. 返回格式化的预览字符串（头 + 尾各 preview_chars 字符）
          3. 返回 (预览内容, True)
        
        否则：
          1. 返回 (原始内容, False)
        
        Args:
            tool_use_id: 工具调用的唯一 ID，用于命名文件
            output: 原始输出内容
        
        Returns:
            (content, was_persisted): 
              - was_persisted=True 时 content 是预览，原始已写入文件
              - was_persisted=False 时 content 是原始内容
        """
        if len(output) <= self._large_output_threshold:
            return (output, False)
        return (self._persist_output(tool_use_id, output), True)

    def _persist_output(self, tool_use_id: str, output: str) -> str:
        """持久化输出，返回预览
        """
        # 确保目录存在
        self._large_output_dir.mkdir(parents=True, exist_ok=True)

        # 写入完整内容到文件
        file_path = self._large_output_dir / f"{tool_use_id}.txt"
        file_path.write_text(output)

        # 生成预览（头 + 尾）
        head = output[:self._preview_chars]
        tail = output[-self._preview_chars:] if len(output) > self._preview_chars else ""
        
        return (
            f"<output persisted to: {file_path.name}>\n"
            f"Head ({self._preview_chars} chars):\n"
            f"{head}\n"
            f"...\n"
            f"Tail ({self._preview_chars} chars):\n"
            f"{tail}"
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def track_tool_result(self, tool_name: str, output: str) -> None:
        """
        追踪 tool_result，用于 micro_compact 决策。
        
        规则：
        - 如果 output 长度超过 min_output_chars_for_compact，则计入 uncaptured_tool_results
        - 始终将 tool_name 加入 recent_tool_names 列表（保持长度不超过 keep_recent_tool_results）
        
        Args:
            tool_name: 工具名称
            output: 工具输出内容
        """
        # 只有超过阈值的 output 才计入统计
        if len(output) > self._min_output_chars_for_compact:
            self._stats.uncaptured_tool_results += 1

        # 记录最近使用的工具名称
        self._stats.recent_tool_names.append(tool_name)
        if len(self._stats.recent_tool_names) > self._keep_recent_tool_results:
            self._stats.recent_tool_names = self._stats.recent_tool_names[-self._keep_recent_tool_results:]

    def refresh_stats(self, model: str, usage: Usage) -> None:
        """
        根据 LLM 响应刷新 context 统计。
        
        使用 estimate_next_context 计算预期消耗，然后更新 context_percentage。
        计算方式参考 engine.py 中的 ExecutionEngine._evaluate_usage。
        
        Args:
            model: 模型 ID
            usage: LLM 响应中的 usage 信息（包含 input_tokens, output_tokens 等）
        """
        if not usage:
            return

        # 获取 context_window
        model_manager = ChatModelManager.get_default()
        chat_model = model_manager.get_model(model)
        context_window = chat_model.context_window if chat_model else 200000

        # 计算当前 context tokens
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read_input_tokens = usage.cache_read_input_tokens or 0

        # 计算预估的下一个 context 大小（包含本次输出）
        total_tokens = input_tokens + output_tokens + cache_read_input_tokens

        # 更新 context percentage
        self._stats.context_percentage = total_tokens / context_window

    # =========================================================================
    # Decision
    # =========================================================================

    def should_full_compact(self) -> bool:
        """
        判断是否需要执行 full_compact。
        
        条件（同时满足）：
        1. context_percentage >= context_percentage_threshold
        2. uncaptured_tool_results > 0
        
        Returns:
            True: 需要 full_compact
            False: 暂不需要
        """
        return (
            self._stats.context_percentage >= self._context_percentage_threshold
            and self._stats.uncaptured_tool_results > 0
        )

    # =========================================================================
    # Compression
    # =========================================================================

    def micro_compact(self, messages: list[MessageParam]) -> list[MessageParam]:
        """
        微压缩：保留最近 N 个 tool_result，其余写入文件并替换为占位符。
        
        内部判断：如果 uncaptured_tool_results <= keep_recent_tool_results，不执行压缩。
        压缩完成后重置 uncaptured_tool_results 为 0。
        
        对于每个被压缩的 tool_result：
        1. 将内容写入 large_output_dir/{tool_use_id}.txt
        2. 替换原始内容为占位符文本，包含读取提示
        
        Args:
            messages: 原始消息列表
        
        Returns:
            压缩后的消息列表
        """
        # 内部判断：只有当未压缩数量超过保留数量时才执行
        if self._stats.uncaptured_tool_results <= self._keep_recent_tool_results:
            return messages

        # 收集所有 tool_result 块
        tool_result_blocks = self._collect_tool_result_blocks(messages)
        if len(tool_result_blocks) <= self._keep_recent_tool_results:
            return messages

        # 压缩较早的 tool_result（保留最近的）
        for msg_idx, block_idx, block in tool_result_blocks[:-self._keep_recent_tool_results]:
            content = block.get("content", "")
            if not isinstance(content, str) or len(content) <= self._min_output_chars_for_compact:
                continue

            tool_use_id = block.get("tool_use_id", "unknown")

            # 写入文件
            self._large_output_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._large_output_dir / f"{tool_use_id}.txt"
            file_path.write_text(content)

            # 替换为占位符
            block["content"] = (
                f"[Output persisted to {file_path.name}. "
                f"If needed, use read_file to read the full content. "
                f"Size: {len(content)} chars]"
            )

        # 重置计数器
        self._stats.uncaptured_tool_results = 0

        return messages

    def full_compact(self, messages: list[MessageParam]) -> list[MessageParam]:
        """
        完整压缩：生成对话摘要，返回压缩后的消息。
        
        流程：
        1. 将原始 messages 写入 transcript_dir/{timestamp}.jsonl
        2. 调用 LLM 生成摘要
        3. 更新 CompactState（last_compact_time, compact_count, last_compact_ratio）
        4. 返回压缩后的单条消息（role=user，包含压缩说明和摘要）
        
        注意：调用方应先检查 should_full_compact() 再决定是否调用。
        
        Args:
            messages: 原始消息列表
        
        Returns:
            压缩后的消息列表，通常只有一条用户消息
        """
        if not messages:
            return messages

        # 计算压缩前的 context 大小
        before_size = len(json.dumps(messages, default=str))

        # 1. 保存 transcript
        transcript_path = self._write_transcript(messages)

        # 2. 生成摘要
        summary = self._summarize(messages)

        # 3. 计算压缩后的估计大小（用于计算收益）
        compact_message = self._build_compact_message(summary, transcript_path)
        after_size = len(json.dumps(compact_message, default=str))

        # 4. 更新状态
        self._state.last_compact_time = datetime.now()
        self._state.compact_count += 1
        self._state.last_compact_ratio = max(0, 1 - after_size / before_size) if before_size > 0 else 0

        # 5. 重置 stats
        self._stats.uncaptured_tool_results = 0

        return compact_message

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _collect_tool_result_blocks(
        self, 
        messages: list[MessageParam]
    ) -> list[tuple[int, int, dict]]:
        """
        收集所有 tool_result 块。
        
        Returns:
            list of (message_index, block_index, block)
        """
        blocks = []
        for msg_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block_idx, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    blocks.append((msg_idx, block_idx, block))
        return blocks

    def _write_transcript(self, messages: list[MessageParam]) -> Path:
        """
        保存完整 transcript 到磁盘。
        
        Args:
            messages: 消息列表
        
        Returns:
            保存的文件路径
        """
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        path = self._transcript_dir / f"transcript_{timestamp}.jsonl"
        
        with path.open("w") as handle:
            for message in messages:
                handle.write(json.dumps(message, default=str) + "\n")
        
        return path

    def _summarize(self, messages: list[MessageParam]) -> str:
        """
        调用 LLM 生成对话摘要。
        
        摘要保留：
        1. 当前任务目标
        2. 已完成的关键动作
        3. 已修改或重点查看过的文件
        4. 关键决定与约束
        5. 下一步应该做什么
        
        Args:
            messages: 原始消息列表
        
        Returns:
            生成的摘要文本
        """
        # 限制输入长度（取最后 80k 字符）
        conversation = json.dumps(messages, default=str)[-80000:]

        prompt = f"""Summarize this coding-agent conversation so work can continue.
Preserve the following information:
1. Current task/goal
2. Completed key actions
3. Modified or importantly viewed files
4. Key decisions and constraints
5. What should be done next

Be compact but concrete. Include specific file names and decisions.

---

CONVERSATION:
{conversation}"""

        # 调用 LLM 生成摘要
        client = ChatModelManager.get_default().get_model(self._summary_model)
        if not client:
            return "[Summary unavailable]"

        # 获取 context_window 计算 max_tokens
        context_window = getattr(client, 'context_window', 200000)
        max_tokens = int(context_window * self._summary_max_tokens_ratio)

        try:
            response = client.chat(
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                system_prompt=[],
            )
            return response.content[0].text.strip() if response.content else "[No summary generated]"
        except Exception as e:
            return f"[Summary error: {e}]"

    def _build_compact_message(
        self, 
        summary: str, 
        transcript_path: Path
    ) -> list[MessageParam]:
        """
        构建压缩后的消息。
        
        Args:
            summary: 摘要文本
            transcript_path: transcript 文件路径
        
        Returns:
            包含压缩说明和摘要的消息列表
        """
        content = (
            "## Conversation Compacted\n\n"
            "This conversation was compacted due to context length limits. "
            "The full history has been saved and can be referenced if needed.\n\n"
            "## Summary\n\n"
            f"{summary}\n\n"
            "---\n"
            f"Full transcript: {transcript_path}\n"
            f"Large outputs: {self._large_output_dir}/"
        )
        return [{"role": "user", "content": content}]


# =============================================================================
# ContextCompactorMiddleware
# =============================================================================

class ContextCompactorMiddleware(SessionMiddleware):
    """
    上下文压缩中间件。
    
    职责：
    1. 在 Session 生命周期中注册压缩相关的 hooks
    2. 在每个 step 结束后检查并执行压缩
    3. 提供 compact 工具供 Agent 主动调用
    
    流程：
      Session.append_tool_use_result():
            │
            ├── pre_tool_result(kwargs)  → 持久化大输出
            │
            ├── 追加 tool_result 到 messages
            │
            └── post_tool_result(tool_use_id, output)  → track_tool_result
    
      Session.post_agent_step():
            │
            └── post_agent_step(resp)
                    │
                    ├── refresh_stats(model, usage)  → 更新 context_percentage
                    │
                    ├── micro_compact() if uncaptured > keep_recent
                    │
                    └── full_compact() if should_full_compact()
    
    Attributes:
        context_manager: 上下文管理器实例
        summary_focus_hint: Agent 调用 compact 时可传入的焦点提示
    """

    def __init__(
        self,
        context_manager: ContextManager,
        summary_focus_hint: Optional[str] = None,
    ):
        self._context_manager = context_manager
        self._summary_focus_hint = summary_focus_hint
        self._session: Optional[Session] = None
        self._pending_intercepted_output: Optional[str] = None  # pre_tool_result 拦截的输出

    # =========================================================================
    # SessionMiddleware Implementation
    # =========================================================================

    def post_session_init(self, session: Session) -> None:
        """
        Session 初始化后的回调。
        
        注册 hooks 和添加 tool provider。
        """
        self._session = session

    def post_agent_step(self, session: Session, resp: Message) -> bool:
        """
        在每个 agent step 结束后执行压缩检查。
        
        1. 从 resp 获取 usage，调用 refresh_stats
        2. micro_compact() 内部判断并执行（如果 uncaptured > keep_recent）
        3. 如果 should_full_compact()，执行 full_compact()
        4. 如果执行了压缩，追加提示消息到 session
        
        Returns:
            True: 继续循环
            False: 结束循环
        """
        # 1. 刷新统计
        if resp.usage:
            model = resp.model
            self._context_manager.refresh_stats(model, resp.usage)

        # 2. 微压缩（内部判断是否执行）
        current_messages = list(session.messages)
        compacted_messages = self._context_manager.micro_compact(current_messages)
        if compacted_messages != current_messages:
            session.overwrite_messages(compacted_messages)
            session.append_user_prompt(
                prompt="[System] Some earlier tool outputs were compacted to save context space.",
                new_round=False
            )

        # 3. 完整压缩（外部判断）
        if self._context_manager.should_full_compact():
            messages = list(session.messages)
            compacted = self._context_manager.full_compact(messages)
            session.overwrite_messages(compacted)
            session.append_user_prompt(
                prompt="[System] Context was fully compacted due to length limits. Full history saved.",
                new_round=False
            )
            return True

        return False

    # =========================================================================
    # Hooks
    # =========================================================================

    def pre_tool_result(self, tool_name: str, kwargs: dict) -> None:
        """
        在工具执行前的 hook。
        
        目前用于记录可能的 large output 信息。
        实际持久化在 post_tool_result 中根据实际输出大小判断。
        """
        # 预留扩展点：可以在这里根据 tool_name 和 kwargs 做一些预处理
        pass

    def post_tool_result(
        self, 
        tool_use_id: str, 
        tool_name: str, 
        output: str
    ) -> None:
        """
        在工具结果追加到 messages 后的 hook。
        
        1. 调用 track_tool_result 统计
        2. 如果输出超过阈值，持久化并替换
        """
        # 追踪 tool_result
        self._context_manager.track_tool_result(tool_name, output)

        # 检查是否需要持久化
        persisted_output, was_persisted = self._context_manager.may_persist_output(
            tool_use_id, output
        )

        # 如果持久化了，需要更新 messages 中的内容
        if was_persisted and self._session:
            self._update_tool_result_in_messages(tool_use_id, persisted_output)

    # =========================================================================
    # Tool
    # =========================================================================

    def compact_tool(self, focus: Optional[str] = None) -> str:
        """
        Agent 调用的 compact 工具。
        
        手动触发完整压缩，Agent 可以传入 focus 指定保留重点。
        
        Args:
            focus: Agent 希望保留的重点
        
        Returns:
            压缩执行结果描述
        """
        if not self._session:
            return "[Error] Session not initialized"

        messages = list(self._session.messages)
        compacted = self._context_manager.full_compact(messages)
        self._session.overwrite_messages(compacted)

        return (
            f"Context compacted. "
            f"History saved to {self._context_manager._transcript_dir}. "
            f"Large outputs saved to {self._context_manager._large_output_dir}. "
            f"Compact count: {self._context_manager.state.compact_count}"
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _update_tool_result_in_messages(
        self, 
        tool_use_id: str, 
        new_content: str
    ) -> None:
        """
        更新 messages 中指定 tool_result 的内容。
        
        Args:
            tool_use_id: 工具调用 ID
            new_content: 新的内容（通常是预览）
        """
        if not self._session:
            return

        messages = list(self._session.messages)
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict) 
                    and block.get("type") == "tool_result"
                    and block.get("tool_use_id") == tool_use_id
                ):
                    block["content"] = new_content
                    break
        
        self._session.overwrite_messages(messages)


# =============================================================================
# ContextCompactorFactory
# =============================================================================

class ContextCompactorFactory(SessionMiddlewareFactory):
    """
    ContextCompactor 的工厂类。
    
    使用方式：
        middleware_factories=[
            ContextCompactorFactory(
                session_id="main",
                base_config_dir=Path(".config/sessions")
            )
        ]
    
    Attributes:
        session_id: Session ID，用于生成目录路径
        base_config_dir: 配置文件根目录，默认 .config/sessions/{session_id}
        其他参数传递给 ContextManager
    """

    def __init__(
        self,
        session_id: str,
        base_config_dir: Path = Path(".config/sessions"),
        context_percentage_threshold: float = DEFAULT_CONTEXT_PERCENTAGE_THRESHOLD,
        keep_recent_tool_results: int = DEFAULT_KEEP_RECENT_TOOL_RESULTS,
        large_output_threshold: int = DEFAULT_LARGE_OUTPUT_THRESHOLD,
        preview_chars: int = DEFAULT_PREVIEW_CHARS,
        min_output_chars_for_compact: int = DEFAULT_MIN_OUTPUT_CHARS_FOR_COMPACT,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        summary_max_tokens_ratio: float = DEFAULT_SUMMARY_MAX_TOKENS_RATIO,
    ):
        self._session_id = session_id
        self._base_config_dir = base_config_dir
        self._context_percentage_threshold = context_percentage_threshold
        self._keep_recent_tool_results = keep_recent_tool_results
        self._large_output_threshold = large_output_threshold
        self._preview_chars = preview_chars
        self._min_output_chars_for_compact = min_output_chars_for_compact
        self._summary_model = summary_model
        self._summary_max_tokens_ratio = summary_max_tokens_ratio

    def create(self, session: Session) -> ContextCompactorMiddleware:
        """
        创建 ContextCompactorMiddleware 实例。
        
        Args:
            session: Session 实例
        
        Returns:
            ContextCompactorMiddleware 实例
        """
        # 生成目录路径
        session_dir = self._base_config_dir / self._session_id
        large_output_dir = session_dir / "large_outputs"
        transcript_dir = session_dir / "transcripts"

        # 创建 ContextManager
        context_manager = ContextManager(
            context_percentage_threshold=self._context_percentage_threshold,
            keep_recent_tool_results=self._keep_recent_tool_results,
            large_output_threshold=self._large_output_threshold,
            preview_chars=self._preview_chars,
            min_output_chars_for_compact=self._min_output_chars_for_compact,
            large_output_dir=large_output_dir,
            transcript_dir=transcript_dir,
            summary_model=self._summary_model,
            summary_max_tokens_ratio=self._summary_max_tokens_ratio,
        )

        return ContextCompactorMiddleware(
            context_manager=context_manager,
            summary_focus_hint=None,
        )
