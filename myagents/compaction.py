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
from typing import Optional

from anthropic.types import MessageParam, Usage

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
        uncaptured_tool_results: 达到 min_output_chars_for_compact 但未处理（被 micro_compact 保留）的 tool_result 数量
        unhandled_tool_results: 未判断过是否要压缩的 tool_result 数量（每次 post_tool_result 后增加）
        context_percentage: 当前 context 占 context_window 的百分比
    """
    uncaptured_tool_results: int = 0
    unhandled_tool_results: int = 0
    context_percentage: float = 0.0


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
       - micro_compact: 保留最近 N 个，达到阈值的其余写入文件并替换为占位符
       - full_compact: 生成对话摘要，保存 transcript，返回压缩后的 messages
    
    Attributes:
        context_percentage_threshold: 触发 full_compact 的 context 百分比阈值
        keep_recent_tool_results: micro_compact 时保留最近的 tool_result 数量
        large_output_threshold: 大输出阈值，超过此值才持久化
        preview_chars: 预览字符数（头尾各取 preview_chars）
        min_output_chars_for_compact: 只有超过此值的 output 才计入 uncaptured_tool_results
        tool_result_dump_dir: tool_result 持久化文件存储目录（包含 large_output 和 micro_compact 的文件）
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
        tool_result_dump_dir: Optional[Path] = None,
        transcript_dir: Optional[Path] = None,
        summary_model: str = DEFAULT_SUMMARY_MODEL,
        summary_max_tokens_ratio: float = DEFAULT_SUMMARY_MAX_TOKENS_RATIO,
    ):
        self._context_percentage_threshold: float = context_percentage_threshold
        self._keep_recent_tool_results: int = keep_recent_tool_results
        self._large_output_threshold: int = large_output_threshold
        self._preview_chars: int = preview_chars
        self._min_output_chars_for_compact: int = min_output_chars_for_compact
        self._tool_result_dump_dir: Path = tool_result_dump_dir or Path(".tool_result_dump")
        self._transcript_dir: Path = transcript_dir or Path(".transcripts")
        self._summary_model: str = summary_model
        self._summary_max_tokens_ratio: float = summary_max_tokens_ratio

        self._state: CompactState = CompactState()
        self._stats: CompactStats = CompactStats()

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

    def may_persist_large_output(self, tool_use_id: str, output: str) -> tuple[str, bool]:
        """
        大输出持久化。
        
        如果 output 长度超过 large_output_threshold：
          1. 将完整内容写入 tool_result_dump_dir/{tool_use_id}.txt
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

        # 确保目录存在
        self._tool_result_dump_dir.mkdir(parents=True, exist_ok=True)

        # 写入完整内容到文件
        file_path = self._tool_result_dump_dir / f"{tool_use_id}.txt"
        file_path.write_text(output)

        # 生成预览（头 + 尾）
        head = output[:self._preview_chars]
        tail = output[-self._preview_chars:] if len(output) > self._preview_chars else ""
        
        preview = (
            f"<output persisted to: {file_path.name}>\n"
            f"Head ({self._preview_chars} chars):\n"
            f"{head}\n"
            f"...\n"
            f"Tail ({self._preview_chars} chars):\n"
            f"{tail}"
        )

        return (preview, True)

    # =========================================================================
    # Statistics
    # =========================================================================

    def track_tool_result(self, tool_name: str, output: str) -> None:
        """
        追踪 tool_result，用于 micro_compact 决策。
        
        规则：
        - 所有 tool_result 都计入 unhandled_tool_results（表示待处理）
        - 如果 output 长度超过 min_output_chars_for_compact，则同时计入 uncaptured_tool_results
        
        Args:
            tool_name: 工具名称
            output: 工具输出内容
        """
        # 计入未处理计数
        self._stats.unhandled_tool_results += 1

        # 只有超过阈值的才计入待压缩计数
        if len(output) > self._min_output_chars_for_compact:
            self._stats.uncaptured_tool_results += 1

    def refresh_stats(self, context_window: int, usage: Usage) -> None:
        """
        根据 LLM 响应刷新 context 统计。
        
        计算方式参考 engine.py 中的 ExecutionEngine._evaluate_usage：
        - total_tokens = input_tokens + output_tokens + cache_read_input_tokens
        - context_percentage = total_tokens / context_window
        
        Args:
            context_window: 模型的 context window 大小
            usage: LLM 响应中的 usage 信息
        """
        if not usage:
            return

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read_input_tokens = usage.cache_read_input_tokens or 0

        total_tokens = input_tokens + output_tokens + cache_read_input_tokens
        self._stats.context_percentage = total_tokens / context_window

    # =========================================================================
    # Decision
    # =========================================================================

    def should_full_compact(self) -> bool:
        """
        判断是否需要执行 full_compact。
        
        条件：context_percentage >= context_percentage_threshold
        
        注意：micro_compact 由内部判断执行，不在此决策。
        
        Returns:
            True: 需要 full_compact
            False: 暂不需要
        """
        return self._stats.context_percentage >= self._context_percentage_threshold

    # =========================================================================
    # Compression
    # =========================================================================

    def micro_compact(self, messages: list[MessageParam]) -> list[MessageParam]:
        """
        微压缩：保留最近 N 个达到阈值的 tool_result，其余写入文件并替换为占位符。
        
        内部判断：只有当 unhandled_tool_results > 0 时才执行扫描。
        扫描最近 unhandled_tool_results 个 tool_result block：
        - 对于达到 min_output_chars_for_compact 且不在保留范围内的，压缩
        - 压缩后重置 unhandled_tool_results 为 0
        
        Args:
            messages: 原始消息列表
        
        Returns:
            压缩后的消息列表
        """
        if self._stats.unhandled_tool_results <= 0:
            return messages

        # 收集所有 tool_result 块（按出现顺序）
        tool_result_blocks = self._collect_tool_result_blocks(messages)
        
        # 只扫描未处理的部分
        # 计算需要处理的 block 范围：全部 block 中，最后 unhandled_tool_results 个
        total_tr_count = len(tool_result_blocks)
        if total_tr_count < self._stats.unhandled_tool_results:
            # 理论上不应该发生，但如果发生了，以实际数量为准
            self._stats.unhandled_tool_results = total_tr_count
        
        # 需要处理的起始索引
        start_idx = total_tr_count - self._stats.unhandled_tool_results
        
        # 统计在需要处理的范围内，达到阈值的 block 数量
        candidates_for_compact = []
        for i in range(start_idx, total_tr_count):
            msg_idx, block_idx, block = tool_result_blocks[i]
            content = block.get("content", "")
            if isinstance(content, str) and len(content) > self._min_output_chars_for_compact:
                candidates_for_compact.append((msg_idx, block_idx, block, content))

        # 如果需要压缩的多于保留数量，则压缩较早的
        # 保留范围是最后 keep_recent_tool_results 个
        compact_count = len(candidates_for_compact) - self._keep_recent_tool_results
        
        if compact_count <= 0:
            # 不需要压缩，但标记为已处理
            self._stats.unhandled_tool_results = 0
            return messages

        # 压缩较早的（排在前面的）候选者
        for i in range(compact_count):
            msg_idx, block_idx, block, content = candidates_for_compact[i]
            tool_use_id = block.get("tool_use_id", f"unknown_{msg_idx}_{block_idx}")

            # 写入文件
            self._tool_result_dump_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._tool_result_dump_dir / f"{tool_use_id}.txt"
            file_path.write_text(content)

            # 替换为占位符
            block["content"] = (
                f"[Output persisted to {file_path.name}. "
                f"If needed, use read_file to read the full content. "
                f"Size: {len(content)} chars]"
            )
            
            # 减少 uncaptured 计数
            self._stats.uncaptured_tool_results -= 1

        # 重置未处理计数
        self._stats.unhandled_tool_results = 0

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
        self._stats.unhandled_tool_results = 0

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
            list of (message_index, block_index, block)，按出现顺序排列
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
            保存的文件路径，格式为 {timestamp}.jsonl
        """
        self._transcript_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        path = self._transcript_dir / f"{timestamp}.jsonl"
        
        with path.open("w") as handle:
            for message in messages:
                handle.write(json.dumps(message, default=str) + "\n")
        
        return path

    def _summarize(self, messages: list[MessageParam], summary_model_object) -> str:
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
            summary_model_object: 用于生成摘要的模型对象（需有 chat() 方法和 context_window 属性）
        
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

        if not summary_model_object:
            return "[Summary unavailable: no model]"

        # 计算 max_tokens
        context_window = getattr(summary_model_object, 'context_window', 200000)
        max_tokens = int(context_window * self._summary_max_tokens_ratio)

        try:
            response = summary_model_object.chat(
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                system_prompt=[],
            )
            
            # 过滤出 text block
            text_blocks = [
                block for block in response.content 
                if hasattr(block, 'type') and block.type == 'text'
            ]
            if text_blocks:
                return text_blocks[0].text.strip()
            return "[No summary generated]"
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
            f"Full transcript: {transcript_path}"
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
      pre_tool_result(tool_name, kwargs):
            │
            └── 执行 tool，获得 output
            └── track_tool_result + may_persist_large_output
            └── 如果 was_persisted，替换 output 为预览
    
      post_tool_result(tool_use_id, tool_name, output):
            └── 仅用于扩展点（当前无额外操作）
    
      post_agent_step():
            ├── refresh_stats(context_window, usage)
            ├── micro_compact() if unhandled > 0  # 内部判断
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
        # 存储 pre_tool_result 中可能需要替换的 output
        self._pending_tool_result_content: Optional[tuple[str, str]] = None

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
        
        1. 获取 model 的 context_window，调用 refresh_stats
        2. micro_compact() 内部判断并执行（如果 unhandled > 0）
        3. 如果 should_full_compact()，执行 full_compact()
        
        Returns:
            True: 继续循环
            False: 结束循环
        """
        # 1. 刷新统计
        if resp.usage:
            model_object = session.resolve_model([resp.model])
            if model_object:
                context_window = getattr(model_object, 'context_window', 200000)
                self._context_manager.refresh_stats(context_window, resp.usage)

        # 2. 微压缩（内部判断是否执行）
        current_messages = list(session.messages)
        compacted_messages = self._context_manager.micro_compact(current_messages)
        if compacted_messages != current_messages:
            session.overwrite_messages(compacted_messages)

        # 3. 完整压缩（外部判断）
        if self._context_manager.should_full_compact():
            messages = list(session.messages)
            compacted = self._context_manager.full_compact(messages)
            session.overwrite_messages(compacted)
            return True

        return False

    # =========================================================================
    # Hooks
    # =========================================================================

    def pre_tool_result(self, tool_name: str, kwargs: dict) -> None:
        """
        在工具执行前的 hook。
        
        注意：这个 hook 在 tool 执行之前调用，用于预处理。
        实际 tool 执行由 caller 完成，结果通过 post_tool_result 回调。
        
        Args:
            tool_name: 工具名称
            kwargs: 工具参数
        """
        # 预留扩展点
        pass

    def post_tool_result(
        self, 
        tool_use_id: str, 
        tool_name: str, 
        output: str
    ) -> None:
        """
        在工具结果追加到 messages 后的 hook。
        
        注意：由于 track_tool_result 需要在 tool_result append 到 messages 之前调用，
        实际统计逻辑在 ToolProvider 层面处理，此处仅作扩展点。
        
        Args:
            tool_use_id: 工具调用 ID
            tool_name: 工具名称
            output: 工具输出内容
        """
        # 扩展点，当前无额外操作
        pass

    # =========================================================================
    # Tool
    # =========================================================================

    def compact_tool(self, session: Session, focus: Optional[str] = None) -> str:
        """
        Agent 调用的 compact 工具。
        
        手动触发完整压缩，Agent 可以传入 focus 指定保留重点。
        
        Args:
            session: Session 实例
            focus: Agent 希望保留的重点
        
        Returns:
            压缩执行结果描述
        """
        messages = list(session.messages)
        compacted = self._context_manager.full_compact(messages)
        session.overwrite_messages(compacted)

        return (
            f"Context compacted. "
            f"History saved to {self._context_manager._transcript_dir}. "
            f"Compact count: {self._context_manager.state.compact_count}"
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _get_summary_model(self, session: Session):
        """
        获取用于摘要的模型对象。
        
        Args:
            session: Session 实例
        
        Returns:
            模型对象，如果不可用返回 None
        """
        return session.resolve_model([self._context_manager._summary_model])


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
        tool_result_dump_dir = session_dir / "tool_results"
        transcript_dir = session_dir / "transcripts"

        # 创建 ContextManager
        context_manager = ContextManager(
            context_percentage_threshold=self._context_percentage_threshold,
            keep_recent_tool_results=self._keep_recent_tool_results,
            large_output_threshold=self._large_output_threshold,
            preview_chars=self._preview_chars,
            min_output_chars_for_compact=self._min_output_chars_for_compact,
            tool_result_dump_dir=tool_result_dump_dir,
            transcript_dir=transcript_dir,
            summary_model=self._summary_model,
            summary_max_tokens_ratio=self._summary_max_tokens_ratio,
        )

        return ContextCompactorMiddleware(
            context_manager=context_manager,
            summary_focus_hint=None,
        )
