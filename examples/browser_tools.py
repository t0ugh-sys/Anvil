"""
Browser Automation Tool Extension for LoopAgent

This is an optional extension that provides browser automation capabilities
using Playwright. To use this, install the extra dependencies:

    pip install loopagent[browser]
    # or
    pip install playwright
    playwright install chromium

Usage:
    from examples.browser_tools import build_browser_tools, BrowserToolContext
    
    # Add to your agent's tools
    tools = build_browser_tools()
    
    # Use in agent:
    # tool_call: {name: "browser_navigate", arguments: {url: "https://example.com"}}
    # tool_call: {name: "browser_click", arguments: {selector: "#submit-button"}}
    # tool_call: {name: "browser_screenshot", arguments: {path: "screenshot.png"}}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from playwright.sync_api import sync_playwright, Page, Browser
except ImportError:
    raise ImportError(
        "Browser tools require playwright. Install with: pip install loopagent[browser]"
    )

from loop_agent.agent_protocol import ToolCall, ToolResult


@dataclass(frozen=True)
class BrowserToolContext:
    """Context for browser tools."""
    page: Page
    workspace_root: Path


BrowserToolFn = Callable[[BrowserToolContext, dict[str, object]], ToolResult]


def _navigate_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Navigate to a URL."""
    url = str(args.get('url', '')).strip()
    call_id = str(args.get('id', 'browser_navigate'))
    if not url:
        return ToolResult(id=call_id, ok=False, output='', error='url is required')
    
    try:
        context.page.goto(url, wait_until='domcontentloaded', timeout=30000)
        title = context.page.title()
        return ToolResult(id=call_id, ok=True, output=f'Navigated to {url}. Title: {title}', error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _click_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Click an element by selector."""
    selector = str(args.get('selector', '')).strip()
    call_id = str(args.get('id', 'browser_click'))
    if not selector:
        return ToolResult(id=call_id, ok=False, output='', error='selector is required')
    
    try:
        context.page.click(selector, timeout=10000)
        return ToolResult(id=call_id, ok=True, output=f'Clicked: {selector}', error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _fill_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Fill an input field."""
    selector = str(args.get('selector', '')).strip()
    value = str(args.get('value', ''))
    call_id = str(args.get('id', 'browser_fill'))
    
    if not selector:
        return ToolResult(id=call_id, ok=False, output='', error='selector is required')
    
    try:
        context.page.fill(selector, value, timeout=10000)
        return ToolResult(id=call_id, ok=True, output=f'Filled {selector} with: {value}', error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _screenshot_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Take a screenshot."""
    path = str(args.get('path', 'screenshot.png')).strip()
    full_page = args.get('full_page', False)
    call_id = str(args.get('id', 'browser_screenshot'))
    
    try:
        context.page.screenshot(path=path, full_page=bool(full_page))
        return ToolResult(id=call_id, ok=True, output=f'Screenshot saved to: {path}', error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _text_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Get text content from an element or page."""
    selector = str(args.get('selector', '')).strip()
    call_id = str(args.get('id', 'browser_text'))
    
    try:
        if selector:
            element = context.page.locator(selector).first
            text = element.inner_text(timeout=5000)
        else:
            text = context.page.content()
        return ToolResult(id=call_id, ok=True, output=text[:5000], error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def _evaluate_tool(context: BrowserToolContext, args: dict[str, object]) -> ToolResult:
    """Evaluate JavaScript in the browser context."""
    script = str(args.get('script', '')).strip()
    call_id = str(args.get('id', 'browser_evaluate'))
    
    if not script:
        return ToolResult(id=call_id, ok=False, output='', error='script is required')
    
    try:
        result = context.page.evaluate(script)
        return ToolResult(id=call_id, ok=True, output=str(result)[:5000], error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def build_browser_tools() -> dict[str, BrowserToolFn]:
    """Build a dictionary of browser automation tools."""
    return {
        'browser_navigate': _navigate_tool,
        'browser_click': _click_tool,
        'browser_fill': _fill_tool,
        'browser_screenshot': _screenshot_tool,
        'browser_text': _text_tool,
        'browser_evaluate': _evaluate_tool,
    }


class BrowserManager:
    """Manages browser lifecycle for LoopAgent."""
    
    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = workspace_root or Path.cwd()
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None
    
    def __enter__(self) -> 'BrowserManager':
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._page:
            self._page.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
    
    def get_page(self) -> Page:
        if not self._page:
            raise RuntimeError('Browser not started. Use within context manager.')
        return self._page
    
    def execute_tool(self, tool_name: str, args: dict[str, object]) -> ToolResult:
        """Execute a browser tool by name."""
        tools = build_browser_tools()
        tool_fn = tools.get(tool_name)
        if not tool_fn:
            return ToolResult(id=str(args.get('id', 'unknown')), ok=False, output='', error=f'unknown tool: {tool_name}')
        
        context = BrowserToolContext(page=self.get_page(), workspace_root=self.workspace_root)
        return tool_fn(context, args)


def demo() -> None:
    """Demo of browser automation tools."""
    print("Starting browser automation demo...")
    
    with BrowserManager() as bm:
        # Navigate to example.com
        result = bm.execute_tool('browser_navigate', {'url': 'https://example.com', 'id': 'nav1'})
        print(f"Navigate: {result.ok} - {result.output or result.error}")
        
        # Get page text
        result = bm.execute_tool('browser_text', {'id': 'text1'})
        print(f"Text: {result.ok} - {result.output[:100]}...")
        
        # Take screenshot
        result = bm.execute_tool('browser_screenshot', {'path': 'example.png', 'id': 'shot1'})
        print(f"Screenshot: {result.ok} - {result.output or result.error}")
    
    print("Demo complete!")


if __name__ == '__main__':
    demo()
