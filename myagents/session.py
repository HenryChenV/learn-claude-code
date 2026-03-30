"""
Session management for myagents.
"""


from agent import AnthropicAgent
from tools.base import ToolManager
from utils import truncate


class Session:

    _main_agent: AnthropicAgent
    _sub_agents: dict[str, AnthropicAgent]
    _tool_manager: ToolManager

    def __init__(self, agent: AnthropicAgent) -> None:
        self._history = []
        self._main_agent = agent
        self._tool_manager = ToolManager()

        # register main agent's tools
        self._tool_manager.register_tools(*agent.tools)

    def handle(self, prompt: str):
        # Append user turn
        input_message = {"role": "user", "content": prompt}
        self._history.append(input_message)

        # Run the agent loop until it stops
        self._agent_loop()

        # Return final response 
        final_message = self._history[-1]
        content = final_message["content"]
        return content

    def _agent_loop(self):
        while True:
            # Agent takes a step
            response = self._main_agent.step(self._history)
            # Append assistant turn
            self._history.append({"role": "assistant", "content": response.content})

            # If the model didn't call a tool, we're done
            if response.stop_reason != "tool_use":
                return

            # Execute each tool call, collect results, or call sub-agents as needed, and append results to history for next step
            results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_kwargs = block.input

                    print(f"\033[33mTool {tool_name} {tool_kwargs}\033[0m")

                    # Tool call
                    output = self._tool_manager.execute(
                        allowed_tools=self._main_agent.allowed_tools, 
                        tool_name=tool_name, 
                        **tool_kwargs
                    )

                    print(truncate(output))

                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

            self._history.append({"role": "user", "content": results})

    def close(self):
        print("Exiting.")
        self._main_agent.close()

