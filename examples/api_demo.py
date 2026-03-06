"""
API Usage Example for LoopAgent

Demonstrates the high-level API for programmatic use.
"""

from loop_agent.api import create_agent, run_goal, AgentConfig, LoopAgentAPI


def example_quick_start():
    """Quick start with run_goal()"""
    # Simplest usage - just pass a goal
    result = run_goal("Say hello in JSON format")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Steps: {result.steps}")


def example_create_agent():
    """Create a configured agent"""
    agent = create_agent(
        provider="mock",
        model="mock-v3",
        max_steps=10,
        temperature=0.5,
    )
    result = agent.run("Your goal here")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")


def example_custom_provider():
    """Use a custom LLM provider"""
    def my_provider(prompt: str) -> str:
        """Your LLM call logic here"""
        # Example: call OpenAI, Anthropic, Gemini, etc.
        # return openai.ChatCompletion.create(...)
        return '{"answer": "response", "done": true}'
    
    agent = create_agent(max_steps=5)
    agent.set_provider(my_provider)
    
    result = agent.run("Analyze this data")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")


def example_coding_agent():
    """Use the coding agent with tools"""
    def openai_provider(prompt: str) -> str:
        """Your OpenAI API call"""
        # import openai
        # response = openai.chat.completions.create(...)
        # return response.choices[0].message.content
        return '{"thought": "analyzing", "plan": ["step 1"], "tool_calls": [], "final": "done"}'
    
    agent = create_agent(max_steps=20)
    agent.set_provider(openai_provider)
    
    result = agent.run_coding("Read README.md and summarize it")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Steps: {result.steps}")


def example_with_config():
    """Use AgentConfig directly for more control"""
    config = AgentConfig(
        provider="openai_compatible",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        temperature=0.7,
        max_steps=15,
        timeout_s=120.0,
        history_window=5,
        workspace="/path/to/workspace",
    )
    config.validate()
    
    agent = LoopAgentAPI(config)
    result = agent.run("Your goal")
    print(result.to_dict())


if __name__ == "__main__":
    print("=== Quick Start ===")
    example_quick_start()
    
    print("\n=== Create Agent ===")
    example_create_agent()
    
    print("\n=== Custom Provider ===")
    example_custom_provider()
