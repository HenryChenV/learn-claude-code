```shell
Session（状态容器）
 ├── Conversation
 ├── ToolManager
 ├── SkillManager
 ├── EventBus ⭐
 └── ContextManager  ⭐

Agent（声明）
 ├── Capabilities（tools / model）
 ├── Persona（soul）
 ├── ModelProvider
 └── Strategy（行为策略） ⭐

Strategy（可插拔）
 └── step(agent, session)

ExecutionEngine（无状态）
 └── run(session, agent)
        ├── selector.select() ⭐
        ├── context_manager.maybe_compact()
        ├── strategy.step()
        ├── session.append()
        └── loop control

StrategySelector (选策略)
    select(agent, session, state):
```

---

你可以问自己三句话：

⸻

1️⃣ 我能不能在不改 Session 的情况下换掉 ReAct → ToT？

👉 不行 = loop 放错地方

⸻

2️⃣ 我能不能在不改 Agent 的情况下加一个 compact 策略？

👉 不行 = context 管理耦合错

⸻

3️⃣ 我能不能并发跑多个 session 共用一个 executor？

👉 不行 = executor 有状态了

⸻

如果三条都满足：

👉 你的架构基本已经到“可扩展级别”了

---


事件机制更新会话元信息 (ContextManager接收时间更新会话元信息，用于压缩策略决定)

```python
class Session:

    def append_message(self, msg):
        self.conversation.append(msg)

        # ⭐ 触发事件
        self.context_manager.on_message_added(msg)

class ContextManager:

    def on_message_added(self, msg):
        if msg.type == "tool":
            self.meta["tool_token_count"] += estimate(msg)
```