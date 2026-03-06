"""
Example: Using prompt templates
"""

from loop_agent.prompts import (
    get_template,
    list_templates,
    register_template,
    PromptTemplate,
)


def main():
    # List built-in templates
    print("Built-in templates:", list_templates())
    
    # Get a template
    template = get_template("json_loop")
    if template:
        # Render with variables
        result = template.render(
            goal="Analyze data",
            state_summary="Initial state",
            last_steps="Step 1: loaded data",
        )
        print("\n--- JSON Loop Template ---")
        print(result)
    
    # Custom template
    custom = PromptTemplate(
        name="my_template",
        description="My custom template",
        template="Hello {{name}}, your task is: {{task}}",
        required_vars=["name", "task"],
    )
    register_template(custom)
    
    # Use custom template
    result = custom.render(name="Alice", task="analyze this")
    print("\n--- Custom Template ---")
    print(result)
    
    print("\nPrompt template example complete!")


if __name__ == "__main__":
    main()
