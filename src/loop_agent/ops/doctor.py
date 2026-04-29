from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.providers import parse_provider_headers


@dataclass(frozen=True)
class HttpProbeResult:
    ok: bool
    status_code: Optional[int ]
    body_snippet: str
    error: Optional[str ] = None


def _http_probe(url: str, headers: Dict[str, str], timeout_s: float) -> HttpProbeResult:
    request = urllib.request.Request(url, headers=headers, method='GET')
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode('utf-8', errors='replace')
            return HttpProbeResult(ok=True, status_code=int(response.status), body_snippet=body[:300])
    except urllib.error.HTTPError as exc:
        body = ''
        try:
            body = exc.read().decode('utf-8', errors='replace')
        except Exception:
            body = ''
        return HttpProbeResult(ok=False, status_code=int(exc.code), body_snippet=body[:300], error=f'HTTP {exc.code}')
    except Exception as exc:
        return HttpProbeResult(ok=False, status_code=None, body_snippet='', error=str(exc))


def run_provider_doctor(
    *,
    base_url: str,
    model: str,
    wire_api: str,
    timeout_s: float,
    api_key_present: bool,
    extra_headers: List[str],
) -> Dict[str, Any]:
    host = ''
    try:
        host = urllib.parse.urlparse(base_url).hostname or ''
    except Exception:
        host = ''

    dns_ips: List[str] = []
    dns_error: Optional[str ] = None
    if host:
        try:
            infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
            seen = set()
            for info in infos:
                ip = info[4][0]
                if ip not in seen:
                    seen.add(ip)
                    dns_ips.append(ip)
        except Exception as exc:
            dns_error = str(exc)
    else:
        dns_error = 'invalid host from base_url'

    tcp_ok = False
    tcp_error: Optional[str ] = None
    if host:
        try:
            with socket.create_connection((host, 443), timeout=timeout_s):
                tcp_ok = True
        except Exception as exc:
            tcp_error = str(exc)

    headers = {
        'Accept': 'application/json',
        'User-Agent': 'Anvil/0.1 doctor',
    }
    headers.update(parse_provider_headers(extra_headers))

    base_probe = _http_probe(base_url.rstrip('/'), headers=headers, timeout_s=timeout_s)

    endpoint = '/responses' if wire_api == 'responses' else '/chat/completions'
    endpoint_headers = dict(headers)
    endpoint_headers['x-loopagent-model'] = model
    endpoint_probe = _http_probe(base_url.rstrip('/') + endpoint, headers=endpoint_headers, timeout_s=timeout_s)

    return {
        'base_url': base_url,
        'host': host,
        'model': model,
        'wire_api': wire_api,
        'api_key_present': api_key_present,
        'dns': {'ok': dns_error is None, 'ips': dns_ips, 'error': dns_error},
        'tcp_443': {'ok': tcp_ok, 'error': tcp_error},
        'probe_base': {
            'ok': base_probe.ok,
            'status_code': base_probe.status_code,
            'error': base_probe.error,
            'body_snippet': base_probe.body_snippet,
        },
        'probe_endpoint': {
            'ok': endpoint_probe.ok,
            'status_code': endpoint_probe.status_code,
            'error': endpoint_probe.error,
            'body_snippet': endpoint_probe.body_snippet,
        },
    }


def format_doctor_report(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
