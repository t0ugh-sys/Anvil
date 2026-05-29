"""Search tools: file search, web search, URL fetching."""
from __future__ import annotations

import re
import subprocess
from typing import Dict, List

from ..agent_protocol import ToolResult
from ..policies import Capability
from ..tool_spec import ToolRisk, ToolSpec
from .base import ToolContext, iter_searchable_files

# Constants
MAX_FETCH_URL_CHARS = 5000
MAX_SEARCH_SNIPPET_CHARS = 200
MAX_WEB_RESULT_TITLE_CHARS = 120

# Search file extensions
_SEARCH_EXTENSIONS = (
    '*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml',
    '*.toml', '*.js', '*.ts', '*.jsx', '*.tsx', '*.rs', '*.go',
)


def _try_ripgrep(pattern: str, workspace_root: str) -> List[str] | None:
    """Try searching with ripgrep (rg) for best performance."""
    try:
        result = subprocess.run(
            ['rg', '-l', '--type-add', 'code:*.{py,js,ts,jsx,tsx,rs,go,toml,yaml,yml,json}',
             '-t', 'code', '-t', 'md', '-t', 'txt', pattern, '.'],
            cwd=workspace_root,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return sorted(
                f[2:] if f.startswith('./') else f
                for f in result.stdout.strip().split('\n')
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _try_grep(pattern: str, workspace_root: str) -> List[str] | None:
    """Try searching with grep as fallback."""
    try:
        cmd = ['grep', '-rl']
        for ext in _SEARCH_EXTENSIONS:
            cmd.extend(['--include', ext])
        cmd.extend([pattern, '.'])
        result = subprocess.run(
            cmd,
            cwd=workspace_root,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return sorted(
                f[2:] if f.startswith('./') else f
                for f in result.stdout.strip().split('\n')
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _python_search(pattern: str, workspace_root) -> List[str]:
    """Pure Python fallback search."""
    from pathlib import Path
    results: List[str] = []
    for path in iter_searchable_files(Path(workspace_root)):
        try:
            # Skip large files (>1MB)
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        if pattern in text:
            relative = path.relative_to(workspace_root).as_posix()
            results.append(relative)
    return sorted(results)


def search_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Search workspace files for a literal pattern.
    
    Uses ripgrep (rg) if available, falls back to grep, then Python.
    """
    pattern = str(args.get('pattern', '')).strip()
    call_id = str(args.get('id', 'search'))
    if not pattern:
        return ToolResult(id=call_id, ok=False, output='', error='pattern is required')

    try:
        workspace_str = str(context.workspace_root)
        
        # Try ripgrep first (fastest)
        files = _try_ripgrep(pattern, workspace_str)
        if files:
            return ToolResult(id=call_id, ok=True, output='\n'.join(files))
        
        # Try grep
        files = _try_grep(pattern, workspace_str)
        if files:
            return ToolResult(id=call_id, ok=True, output='\n'.join(files))
        
        # Python fallback
        files = _python_search(pattern, context.workspace_root)
        return ToolResult(id=call_id, ok=True, output='\n'.join(files))
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def web_search_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Search the web using DuckDuckGo HTML search (no API key required)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    query = str(args.get('query', '')).strip()
    call_id = str(args.get('id', 'web_search'))
    if not query:
        return ToolResult(id=call_id, ok=False, output='', error='query is required')

    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f'https://html.duckduckgo.com/html/?q={encoded_query}'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode('utf-8', errors='replace')
        
        # Parse results from HTML
        results: List[str] = []
        pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.+?)</a>.*?<a class="result__snippet"[^>]*>(.+?)</a>'
        matches = re.findall(pattern, html, re.DOTALL)
        
        for i, (link, title, snippet) in enumerate(matches[:10], 1):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = re.sub(r'<[^>]+>', '', snippet).strip()
            results.append(f'{i}. {title_clean}\n   URL: {link}\n   {snippet_clean[:MAX_SEARCH_SNIPPET_CHARS]}')
        
        if not results:
            return ToolResult(id=call_id, ok=True, output='No results found', error=None)
        
        output = f'Found {len(results)} results:\n\n' + '\n\n'.join(results)
        return ToolResult(id=call_id, ok=True, output=output, error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


_ALLOWED_URL_SCHEMES = frozenset({'http', 'https'})
_BLOCKED_URL_HOSTS = frozenset({
    '169.254.169.254',  # AWS/GCP/Azure metadata
    'metadata.google.internal',
    'localhost',
    '127.0.0.1',
    '::1',
})


def fetch_url_tool(context: ToolContext, args: Dict[str, object]) -> ToolResult:
    """Fetch content from a specific URL."""
    import urllib.error
    import urllib.parse
    import urllib.request
    from urllib.parse import urlparse

    url = str(args.get('url', '')).strip()
    call_id = str(args.get('id', 'fetch_url'))
    if not url:
        return ToolResult(id=call_id, ok=False, output='', error='url is required')

    try:
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_URL_SCHEMES:
            return ToolResult(id=call_id, ok=False, output='', error=f'blocked URL scheme: {parsed.scheme}')
        if parsed.hostname in _BLOCKED_URL_HOSTS:
            return ToolResult(id=call_id, ok=False, output='', error='blocked URL host')
    except Exception:
        return ToolResult(id=call_id, ok=False, output='', error='invalid URL')

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        request = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            html = response.read().decode('utf-8', errors='replace')
        
        # Simple HTML to text conversion
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        if len(text) > MAX_FETCH_URL_CHARS:
            text = text[:MAX_FETCH_URL_CHARS] + f'\n\n... (truncated, total {len(text)} chars)'
        
        return ToolResult(id=call_id, ok=True, output=text.strip(), error=None)
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


def search_tool_specs() -> List[ToolSpec]:
    """Return specs for search tools."""
    return [
        ToolSpec(
            name='search',
            description='Search workspace files for a literal pattern.',
            capabilities=(Capability.read,),
            risk_level=ToolRisk.low,
            requires_workspace=True,
            input_notes='',
        ),
        ToolSpec(
            name='web_search',
            description='Search the web through a simple HTTP search endpoint.',
            capabilities=(Capability.network,),
            risk_level=ToolRisk.high,
            requires_workspace=False,
            input_notes='Provide query text.',
        ),
        ToolSpec(
            name='fetch_url',
            description='Fetch and strip text content from one URL.',
            capabilities=(Capability.network,),
            risk_level=ToolRisk.high,
            requires_workspace=False,
            input_notes='Provide one URL string.',
        ),
    ]
