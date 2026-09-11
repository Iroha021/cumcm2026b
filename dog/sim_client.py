"""唯一发 HTTP 的地方：4 条指令的封装。

协议要点（附件 2 第 5 节）：
    路径精确 /enter /measure /clear /exit，POST + Content-Type: application/json；
    未声明字段不会被忽略，而是返回 200 + accepted=false —— 故请求体必须白名单构造；
    必须同时检查 HTTP 状态与 accepted；
    每个新动作新 request_id；网络故障重试必须复用"原 request_id + 原请求体"，
    这样模拟器返回首次的完整响应，不会重复移动/检测/计费。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

Transport = Callable[[str, Dict[str, Any]], Tuple[int, Optional[Dict[str, Any]]]]

PATHS = ("/enter", "/measure", "/clear", "/exit")
RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)


class SimError(Exception):
    pass


class TransportError(SimError):
    """连不上、超时、连接被关闭（可能没有 JSON 体）。"""


class ProtocolError(SimError):
    """HTTP 非 200，或响应不是合法业务 JSON —— 属于我方 bug。"""


class RejectedError(SimError):
    """HTTP 200 且 accepted=false：JSON 合法但被拒（字段错误/状态拒绝/标识不匹配）。"""

    def __init__(self, path: str, payload: Dict[str, Any], body: Optional[Dict[str, Any]]):
        self.path = path
        self.payload = payload
        self.body = body or {}
        super(RejectedError, self).__init__(
            "%s rejected: %s" % (path, json.dumps(self.body, ensure_ascii=False)))


class Outcome(object):
    __slots__ = ("kind", "accepted", "virtual_time_s", "real_timestamp_ms", "raw", "extra")

    def __init__(self, kind: str, body: Dict[str, Any]):
        self.kind = kind
        self.accepted = bool(body.get("accepted"))
        self.virtual_time_s = _as_float(body.get("virtual_time_s"))
        self.real_timestamp_ms = _as_float(body.get("real_timestamp_ms"))
        self.raw = body
        self.extra: Dict[str, Any] = {}

    @property
    def result(self) -> Optional[str]:
        return self.extra.get("result")

    def __repr__(self) -> str:
        return "<%s accepted=%s vt=%s %s>" % (self.kind, self.accepted,
                                              self.virtual_time_s, self.extra)


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def http_transport(base_url: str, timeout_s: float = 8.0) -> Transport:
    url_base = base_url.rstrip("/")

    def _post(path: str, payload: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            url_base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                status = int(resp.status)
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            body = exc.read().decode("utf-8", "replace")
        except Exception as exc:                      # URLError / socket.timeout / ...
            raise TransportError("%s: %s" % (type(exc).__name__, exc))
        try:
            parsed = json.loads(body) if body.strip() else None
        except ValueError:
            parsed = None
        return status, parsed

    return _post


class SimClient(object):
    def __init__(self, cfg: Dict[str, Any], logger=None, transport: Optional[Transport] = None,
                 session_tag: str = ""):
        self.cfg = cfg
        self.logger = logger
        # request_id 按协议是"当前测试会话内的幂等键"。连续演练时给每次会话加一个短标记，
        # 万一模拟器的幂等表跨会话保留，也不会把上一局的响应当成新局的响应。
        self.session_tag = session_tag
        self.base_url = cfg.get("base_url", "http://127.0.0.1:2026")
        self.arena_id = cfg.get("arena_id", "default")
        self.robot_id = cfg["team_id"]
        self.timeout_s = float(cfg.get("http_timeout_s", 8.0))
        rt = cfg.get("retry", {}) or {}
        self.max_attempts = int(rt.get("max_attempts", 4))
        self.backoff_s = float(rt.get("backoff_s", 0.35))
        self.transport: Transport = transport or http_transport(self.base_url, self.timeout_s)
        self._counter = {"enter": 0, "measure": 0, "clear": 0, "exit": 0}
        self.entered = False
        self.exited = False

    # ------------------------------------------------------------- 请求构造
    def _new_request_id(self, kind: str) -> str:
        self._counter[kind] += 1
        tag = ("-" + self.session_tag) if self.session_tag else ""
        return "%s%s-%06d" % (kind[0], tag, self._counter[kind])

    def _base(self, kind: str, request_id: str) -> Dict[str, Any]:
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": request_id}

    # --------------------------------------------------------------- 调用
    def _call(self, path: str, payload: Dict[str, Any]) -> Outcome:
        kind = path.strip("/")
        status, body, attempts = self._call_raw(path, payload)
        if status != 200:
            raise ProtocolError("HTTP %s on %s: %s" % (status, path, json.dumps(body, ensure_ascii=False)
                                                       if body else "<no body>"))
        if body is None:
            raise ProtocolError("empty body on %s" % path)
        if body.get("accepted") is not True:
            raise RejectedError(path, payload, body)
        out = Outcome(kind, body)
        if kind == "enter":
            self.entered = True
            out.extra["remaining_real_duration_s"] = _as_float(body.get("remaining_real_duration_s"))
            out.extra["max_real_duration_s"] = _as_float(body.get("max_real_duration_s"))
            out.extra["max_virtual_duration_s"] = _as_float(body.get("max_virtual_duration_s"))
        elif kind == "measure":
            out.extra["result"] = body.get("measure_result")
            if "svd_deg" in body:
                out.extra["svd_deg"] = _as_float(body.get("svd_deg"))
        elif kind == "clear":
            out.extra["result"] = body.get("clear_result")
        elif kind == "exit":
            self.exited = True
            out.extra["exit_reason"] = body.get("exit_reason")
        self._log_exchange(path, payload, status, body, attempts, ok=True)
        return out

    def _call_raw(self, path: str, payload: Dict[str, Any]):
        """同 payload 幂等重试；返回 (status, body, attempts)。"""
        attempt = 0
        while True:
            attempt += 1
            try:
                status, body = self.transport(path, payload)
            except TransportError as exc:
                if attempt >= self.max_attempts:
                    self._log_exchange(path, payload, None, {"error": str(exc)}, attempt, ok=False)
                    raise
                time.sleep(self.backoff_s * attempt)
                continue
            if status in RETRYABLE_STATUS and attempt < self.max_attempts:
                time.sleep(self.backoff_s * attempt)
                continue
            return status, body, attempt

    def _log_exchange(self, path, payload, status, body, attempts, ok):
        if self.logger is None:
            return
        self.logger.event("http", path=path, request_id=payload.get("request_id"),
                          status=status, attempts=attempts, ok=ok,
                          body=body if isinstance(body, dict) else None)

    # ------------------------------------------------------------ 公开接口
    def enter(self) -> Outcome:
        payload = self._base("enter", self._new_request_id("enter"))
        return self._call("/enter", payload)

    def enter_wait(self, timeout_s: float = 240.0) -> Outcome:
        """等待接口就绪：倒计时期间/未开放时连接会直接失败，属正常现象。

        重试必须复用同一 payload（同一 request_id），这样即便首次请求其实已经执行、
        只是响应丢失，重试也会取回首次的完整响应，不会重复进入或重复计时。
        """
        payload = self._base("enter", self._new_request_id("enter"))
        t0 = time.monotonic()
        last_exc = None
        while True:
            try:
                return self._call("/enter", payload)
            except TransportError as exc:
                last_exc = exc
                if time.monotonic() - t0 >= timeout_s:
                    raise
                time.sleep(1.0)
            except RejectedError:
                raise

    def measure(self, position, channel: int) -> Outcome:
        payload = self._base("measure", self._new_request_id("measure"))
        payload["position"] = {"x": float(position[0]), "y": float(position[1])}
        payload["channel"] = int(channel)
        return self._call("/measure", payload)

    def clear(self, position, channel: int) -> Outcome:
        payload = self._base("clear", self._new_request_id("clear"))
        payload["position"] = {"x": float(position[0]), "y": float(position[1])}
        payload["channel"] = int(channel)
        return self._call("/clear", payload)

    def exit(self) -> Outcome:
        payload = self._base("exit", self._new_request_id("exit"))
        return self._call("/exit", payload)
