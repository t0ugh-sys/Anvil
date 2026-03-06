"""
Provider Examples for LoopAgent

Demonstrates how to use different LLM providers.
"""

from loop_agent.llm.providers import (
    build_invoke_from_args,
    get_provider,
    list_providers,
)
import argparse


def example_list_providers():
    """List all available providers"""
    providers = list_providers()
    print("Available providers:")
    for name, desc in providers.items():
        print(f"  - {name}: {desc}")


def example_mock_provider():
    """Use the mock provider (no API key needed)"""
    invoke_fn = get_provider('mock')
    if invoke_fn:
        response = invoke_fn("test prompt")
        print(f"Mock response: {response[:100]}...")


def example_build_from_args():
    """Build provider from command-line style args"""
    # Simulate argparse.Namespace
    args = argparse.Namespace(
        provider='openai_compatible',
        model='gpt-4o-mini',
        base_url='https://api.openai.com/v1',
        api_key_env='OPENAI_API_KEY',
        temperature=0.7,
        max_retries=3,
        retry_backoff_s=1.0,
        retry_http_codes=[502, 503],
        provider_header=[],
    )
    
    invoke_fn = build_invoke_from_args(args, mode='json')
    if invoke_fn:
        response = invoke_fn("Your prompt here")
        print(f"Response: {response[:200]}...")


def example_anthropic():
    """Use Anthropic Claude provider"""
    import os
    
    args = argparse.Namespace(
        provider='anthropic',
        model='claude-3-haiku-20240307',
        api_key_env='ANTHROPIC_API_KEY',
        temperature=0.7,
        max_retries=3,
        retry_backoff_s=1.0,
        retry_http_codes=[502, 503, 529],
        provider_header=[],
    )
    
    invoke_fn = build_invoke_from_args(args, mode='json')
    if invoke_fn:
        response = invoke_fn("Your prompt here")
        print(f"Claude response: {response[:200]}...")


def example_gemini():
    """Use Google Gemini provider"""
    args = argparse.Namespace(
        provider='gemini',
        model='gemini-1.5-flash',
        api_key_env='GEMINI_API_KEY',
        temperature=0.7,
        max_retries=3,
        retry_backoff_s=1.0,
        retry_http_codes=[502, 503],
        provider_header=[],
    )
    
    invoke_fn = build_invoke_from_args(args, mode='json')
    if invoke_fn:
        response = invoke_fn("Your prompt here")
        print(f"Gemini response: {response[:200]}...")


if __name__ == "__main__":
    print("=== List Providers ===")
    example_list_providers()
    
    print("\n=== Mock Provider ===")
    example_mock_provider()
    
    print("\n=== Build from Args ===")
    example_build_from_args()
