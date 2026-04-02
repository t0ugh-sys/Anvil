"""
LSPTool - Language Server Protocol Support

参考 Claude Code LSPTool 的实现，支持语言服务器协议。
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loop_agent.agent_protocol import ToolResult
from loop_agent.tool_def import ToolRegistration, ToolUseContext
from loop_agent.tool_builder import build_tool, build_search_tool


# ============== LSP Types ==============

@dataclass
class LSPDiagnostic:
    """LSP 诊断信息"""
    severity: int  # 1=Error, 2=Warning, 3=Info, 4=Hint
    message: str
    range: Dict[str, Any]


@dataclass
class LSPSymbol:
    """LSP 符号信息"""
    name: str
    kind: int
    location: Dict[str, Any]
    container_name: Optional[str] = None


# ============== LSP Client ==============

class LSPClient:
    """简化的 LSP 客户端"""
    
    def __init__(self, workspace_root: Path, server_command: List[str]):
        self.workspace_root = workspace_root
        self.server_command = server_command
        self._proc: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._pending: Dict[int, Any] = {}
    
    def start(self) -> bool:
        """启动 LSP 服务器"""
        try:
            self._proc = subprocess.Popen(
                self.server_command,
                cwd=str(self.workspace_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return True
        except Exception:
            return False
    
    def stop(self) -> None:
        """停止 LSP 服务器"""
        if self._proc:
            self._proc.terminate()
            self._proc = None
    
    def send_request(self, method: str, params: Dict[str, Any]) -> Optional[Any]:
        """发送 LSP 请求"""
        if not self._proc:
            return None
        
        self._request_id += 1
        request = {
            'jsonrpc': '2.0',
            'id': self._request_id,
            'method': method,
            'params': params,
        }
        
        try:
            # 发送请求
            request_json = json.dumps(request) + '\n'
            self._proc.stdin.write(request_json)
            self._proc.stdin.flush()
            
            # 读取响应
            response_line = self._proc.stdout.readline()
            if response_line:
                response = json.loads(response_line)
                return response.get('result')
            
        except Exception:
            pass
        
        return None
    
    def get_diagnostics(self, file_path: str) -> List[LSPDiagnostic]:
        """获取文件诊断"""
        # 这个需要服务器支持 textDocument/publishDiagnostics
        # 这里返回空列表作为基础实现
        return []
    
    def find_symbols(self, file_path: str, query: str) -> List[LSPSymbol]:
        """查找符号"""
        params = {
            'workspaceRoot': str(self.workspace_root),
            'query': query,
        }
        
        result = self.send_request('workspace/symbol', params)
        if not result:
            return []
        
        symbols: List[LSPSymbol] = []
        for item in result:
            symbols.append(LSPSymbol(
                name=item.get('name', ''),
                kind=item.get('kind', 0),
                location=item.get('location', {}),
                container_name=item.get('containerName'),
            ))
        
        return symbols
    
    def go_to_definition(self, file_path: str, line: int, column: int) -> Optional[Dict[str, Any]]:
        """跳转到定义"""
        params = {
            'textDocument': {
                'uri': self._path_to_uri(file_path),
            },
            'position': {
                'line': line - 1,  # LSP 使用 0-based
                'character': column,
            },
        }
        
        result = self.send_request('textDocument/definition', params)
        return result
    
    def _path_to_uri(self, path: str) -> str:
        """路径转 URI"""
        # Windows 路径处理
        path = path.replace('\\', '/')
        return f'file:///{path}'


# ============== LSP Tool Registry ==============

_LSP_SERVERS: Dict[str, List[str]] = {
    'python': ['python', '-m', 'pylsp'],
    'typescript': ['typescript-language-server', '--stdio'],
    'javascript': ['typescript-language-server', '--stdio'],
    'rust': ['rust-analyzer'],
    'go': ['gopls'],
    'java': ['java', '-jar', 'path/to/jdtls.jar'],
    'cpp': ['clangd'],
}


# ============== LSPTool Handler ==============

def _lsp_handler(
    context: ToolUseContext,
    args: Dict[str, Any],
) -> ToolResult:
    """LSP 工具处理函数"""
    call_id = str(args.get('id', 'lsp'))
    
    action = str(args.get('action', 'diagnostics')).strip().lower()
    file_path = str(args.get('path', '')).strip()
    
    workspace = Path(context.workspace_root)
    
    try:
        if action == 'diagnostics':
            # 获取诊断信息
            # 注意：完整的诊断需要 LSP 服务器运行
            # 这里提供基础实现
            return ToolResult(
                id=call_id,
                ok=True,
                output='LSP diagnostics not available (server not running)',
                error=None,
            )
        
        elif action == 'symbols':
            # 查找符号
            query = str(args.get('query', '')).strip()
            if not query:
                return ToolResult(id=call_id, ok=False, output='', error='query is required')
            
            return ToolResult(
                id=call_id,
                ok=True,
                output=f'Symbol search not available (server not running)',
                error=None,
            )
        
        elif action == 'definition':
            # 跳转到定义
            line = int(args.get('line', 1))
            column = int(args.get('column', 0))
            
            return ToolResult(
                id=call_id,
                ok=True,
                output=f'Go to definition not available (server not running)',
                error=None,
            )
        
        elif action == 'list_servers':
            # 列出支持的服务器
            lines = ['Supported LSP servers:']
            for lang, cmd in _LSP_SERVERS.items():
                lines.append(f'  {lang}: {" ".join(cmd)}')
            
            return ToolResult(id=call_id, ok=True, output='\n'.join(lines), error=None)
        
        else:
            return ToolResult(id=call_id, ok=False, output='', error=f'Unknown action: {action}')
    
    except Exception as exc:
        return ToolResult(id=call_id, ok=False, output='', error=str(exc))


# ============== Tool Registration ==============

LSP_TOOL = build_tool({
    'name': 'lsp',
    'description': 'Language Server Protocol operations',
    'handler': _lsp_handler,
    'input_schema': {
        'type': 'object',
        'properties': {
            'action': {
                'type': 'string',
                'description': 'Action: diagnostics, symbols, definition, list_servers',
                'enum': ['diagnostics', 'symbols', 'definition', 'list_servers'],
            },
            'path': {
                'type': 'string',
                'description': 'File path',
            },
            'query': {
                'type': 'string',
                'description': 'Search query (for symbols action)',
            },
            'line': {
                'type': 'number',
                'description': 'Line number (for definition action)',
            },
            'column': {
                'type': 'number',
                'description': 'Column number (for definition action)',
            },
        },
    },
    'search_hint': 'language server protocol',
})


# ============== Exports ==============

def get_lsp_tool() -> ToolRegistration:
    return LSP_TOOL
