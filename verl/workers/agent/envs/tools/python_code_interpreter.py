"""
python_code_interpreter.py — Python 代码解释器工具

支持两种执行模式：
  - 无状态模式（默认）：每次在隔离子进程中执行，适合独立计算任务。
  - 有状态模式：通过 Jupyter Kernel 保持会话上下文，适合多轮交互计算。
    需要安装：pip install jupyter_client ipykernel

安全机制：
  - 禁止使用危险关键字（os、subprocess、socket 等）
  - 禁止危险文件写操作（open write 模式、pandas.to_csv 等）
  - 超时控制（默认 3 秒）

使用方式：
    from verl.workers.agent.envs.tools.python_code_interpreter import PythonCodeInterpreterTool

    tool = PythonCodeInterpreterTool()

    # 直接集成到 AgentEval
    from verl.workers.agent.envs.agent_eval import AgentEval
    agent = AgentEval(tools=[PythonCodeInterpreterTool()])
"""

import sys
import io
import traceback
import re
import json
import ast
import tokenize
import time
import os
import subprocess
import atexit
import multiprocessing
from queue import Empty
from typing import Any, Dict, Optional, Tuple

from verl.workers.agent.tool_envs import ToolBase

try:
    from jupyter_client import BlockingKernelClient
except ImportError:
    print(
        "[PythonCodeInterpreterTool] `jupyter_client` / `ipykernel` not found. "
        "Stateful mode unavailable. Run: pip install jupyter_client ipykernel"
    )
    BlockingKernelClient = None


# ==============================================================================
#  Safety checks
# ==============================================================================

_BANNED_KEYWORDS = {
    "exit", "quit", "yield", "requests", "urllib", "socket",
    "pip", "install", "conda", "subprocess", "os", "shutil",
}

_DANGEROUS_OPEN_MODES = {
    "w", "a", "x", "w+", "a+", "x+",
    "wb", "ab", "xb", "w+b", "a+b", "x+b",
    "r+", "rb+", "rt+",
}

_DANGEROUS_WRITE_METHODS = {
    "to_csv", "to_excel", "to_json", "to_parquet",
    "to_pickle", "to_sql", "to_hdf", "to_feather", "to_stata",
}


def _check_banned_keywords(code: str) -> Tuple[bool, str]:
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in tokens:
            if tok.type == tokenize.NAME and tok.string in _BANNED_KEYWORDS:
                return False, f"Safety Error: banned keyword '{tok.string}' is not allowed."
    except tokenize.TokenError as e:
        return False, f"Syntax Error: {e}"
    return True, ""


class _ASTSafetyChecker(ast.NodeVisitor):
    def __init__(self):
        self.violations: list = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = self._open_mode(node)
            if mode in _DANGEROUS_OPEN_MODES:
                self.violations.append(f"open() with dangerous mode '{mode}'")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _DANGEROUS_WRITE_METHODS:
                self.violations.append(f"file-writing method '.{node.func.attr}()'")
        self.generic_visit(node)

    @staticmethod
    def _open_mode(node) -> str:
        mode = "r"
        if len(node.args) > 1:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                mode = arg.value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        return mode


def _check_ast_safety(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    checker = _ASTSafetyChecker()
    checker.visit(tree)
    if checker.violations:
        return False, f"Safety Error: {', '.join(checker.violations)}"
    return True, ""


# ==============================================================================
#  Jupyter kernel manager (stateful mode)
# ==============================================================================

class JupyterKernelManager:
    """管理 Jupyter 内核的生命周期，支持多 session。"""

    _work_dir = os.path.join(os.getcwd(), "tmp_code_interpreter")

    def __init__(self):
        if BlockingKernelClient is None:
            raise ImportError(
                "Stateful mode requires `jupyter_client` and `ipykernel`. "
                "Run: pip install jupyter_client ipykernel"
            )
        os.makedirs(self._work_dir, exist_ok=True)
        self._kernels: Dict[str, Tuple[BlockingKernelClient, subprocess.Popen]] = {}
        atexit.register(self.shutdown_all)

    def get_or_create(self, session_id: str) -> BlockingKernelClient:
        if session_id in self._kernels:
            kc, proc = self._kernels[session_id]
            if proc.poll() is None:
                return kc
            self._remove(session_id)

        conn_file = os.path.join(self._work_dir, f"kernel-{session_id}.json")
        proc = subprocess.Popen(
            [sys.executable, "-m", "ipykernel_launcher", "-f", conn_file],
            cwd=self._work_dir,
        )
        while not os.path.exists(conn_file):
            time.sleep(0.1)
            if proc.poll() is not None:
                raise RuntimeError("Jupyter kernel process failed to start.")

        kc = BlockingKernelClient(connection_file=conn_file)
        kc.load_connection_file()
        kc.start_channels()
        try:
            kc.wait_for_ready(timeout=60)
        except RuntimeError:
            kc.shutdown()
            proc.terminate()
            raise RuntimeError("Could not connect to Jupyter kernel.")

        self._kernels[session_id] = (kc, proc)
        return kc

    def execute_code(self, session_id: str, code: str, timeout: int) -> Dict[str, Any]:
        kc = self.get_or_create(session_id)
        kc.execute(code)
        stdout, stderr, result, images = "", "", None, []

        while True:
            try:
                msg = kc.get_iopub_msg(timeout=timeout)
                mtype = msg["msg_type"]
                if mtype == "status" and msg["content"].get("execution_state") == "idle":
                    break
                elif mtype == "execute_result":
                    data = msg["content"]["data"]
                    result = data.get("text/plain")
                    if "image/png" in data:
                        images.append(data["image/png"])
                elif mtype == "display_data" and "image/png" in msg["content"]["data"]:
                    images.append(msg["content"]["data"]["image/png"])
                elif mtype == "stream":
                    if msg["content"]["name"] == "stdout":
                        stdout += msg["content"]["text"]
                    else:
                        stderr += msg["content"]["text"]
                elif mtype == "error":
                    stderr += "\n".join(msg["content"]["traceback"])
            except Empty:
                stderr += f"\nExecution timed out after {timeout}s."
                break

        image_paths = []
        for i, img_b64 in enumerate(images):
            import base64
            img_path = os.path.join(self._work_dir, f"{session_id}_fig_{i+1}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            image_paths.append(img_path)
        if image_paths:
            stdout += "\nGenerated images:\n" + "\n".join(image_paths)

        return {"stdout": stdout, "stderr": stderr, "result": result}

    def shutdown(self, session_id: str):
        if session_id in self._kernels:
            self._remove(session_id)

    def shutdown_all(self):
        for sid in list(self._kernels):
            self._remove(sid)

    def _remove(self, session_id: str):
        kc, proc = self._kernels.pop(session_id)
        try:
            kc.shutdown()
        except Exception:
            pass
        proc.terminate()
        proc.wait()
        conn_file = getattr(kc, "connection_file", None)
        if conn_file and os.path.exists(conn_file):
            os.remove(conn_file)


# ==============================================================================
#  Tool
# ==============================================================================

class PythonCodeInterpreterTool(ToolBase):
    """
    Python 代码解释器工具。

    无状态调用格式（default parser）：
        <tool_call>{"name": "python_interpreter", "arguments": {"code": "print(1+1)"}}</tool_call>

    有状态调用（跨轮共享变量），需传入 session_id：
        <tool_call>{"name": "python_interpreter", "arguments": {"code": "x = 42", "session_id": "s1"}}</tool_call>
    """

    name = "python_interpreter"
    TIMEOUT = 3  # seconds, override per instance if needed

    def __init__(self, timeout: int = TIMEOUT, **kwargs):
        super().__init__(
            name=self.name,
            description=(
                "Execute Python code and return stdout/stderr/result. "
                "Supports stateless (default) and stateful (Jupyter kernel) modes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Optional. Provide a session ID to enable stateful (Jupyter kernel) "
                            "execution where variables persist across calls with the same ID."
                        ),
                    },
                },
                "required": ["code"],
            },
        )
        self.TIMEOUT = timeout
        self._jupyter_manager: Optional[JupyterKernelManager] = None

    @property
    def jupyter_manager(self) -> JupyterKernelManager:
        if self._jupyter_manager is None:
            self._jupyter_manager = JupyterKernelManager()
        return self._jupyter_manager

    # ------------------------------------------------------------------
    # ToolBase interface
    # ------------------------------------------------------------------

    def execute(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        """同步执行，支持无状态和有状态两种模式。"""
        if re.search(r'<answer>.*?</answer>', action_string, re.DOTALL):
            return "", 0.0, True, {}

        action = self._extract_action(action_string)
        if not action:
            return "", 0.0, True, {}

        try:
            tool_call = self._parse_tool_call(action)
            args = tool_call.get("arguments", {})
            code = args.get("code", "")
            session_id = args.get("session_id")

            if not isinstance(code, str) or not code.strip():
                raise ValueError("'code' must be a non-empty string.")

            ok, msg = _check_banned_keywords(code)
            if not ok:
                raise PermissionError(msg)
            ok, msg = _check_ast_safety(code)
            if not ok:
                raise PermissionError(msg)

            if session_id:
                result = self.jupyter_manager.execute_code(session_id, code, self.TIMEOUT)
            else:
                result = self._run_stateless(code)

            result_str = json.dumps({"executed_code": code, **result}, ensure_ascii=False)
            obs = (
                "\n<|im_start|>user\n"
                f"<tool_response>{result_str}</tool_response>\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            return obs, 0.0, False, result

        except (ValueError, PermissionError, KeyError, json.JSONDecodeError) as e:
            obs = f"\n<|im_start|>user\nError: {e}<|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e), "status": "failed"}
        except Exception as e:
            obs = f"\n<|im_start|>user\nAn unexpected error occurred: {e}<|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e), "status": "failed"}

    async def execute_async(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        import asyncio
        return await asyncio.to_thread(self.execute, action_string, **kwargs)

    def reset(self, **kwargs):
        if self._jupyter_manager:
            self._jupyter_manager.shutdown_all()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_action(self, text: str) -> Optional[str]:
        matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        return matches[-1] if matches else None

    def _parse_tool_call(self, action: str) -> dict:
        try:
            return json.loads(action.strip())
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(action.strip())
            except Exception:
                fixed = re.sub(
                    r'"code"\s*:\s*"([^"]*)"',
                    lambda m: '"code": "{}"'.format(m.group(1).replace("\n", "\\n")),
                    action,
                )
                return json.loads(fixed)

    @staticmethod
    def _run_code_in_process(code: str, result_queue: multiprocessing.Queue):
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_val = None
        orig_out, orig_err = sys.stdout, sys.stderr
        try:
            builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(__builtins__.__dict__)
            for name in ("__import__", "eval", "exec", "compile", "input"):
                builtins.pop(name, None)
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            exec(code, {"__builtins__": builtins}, locals())  # noqa: S102
            result_val = locals().get("result")
        except Exception:
            stderr_buf.write(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err
            result_queue.put({
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
                "result": result_val,
            })

    def _run_stateless(self, code: str) -> Dict[str, Any]:
        q: multiprocessing.Queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=self._run_code_in_process, args=(code, q))
        proc.start()
        proc.join(timeout=self.TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"stdout": "", "stderr": f"Execution timed out after {self.TIMEOUT}s.", "result": None}
        try:
            return q.get(block=False)
        except Empty:
            return {"stdout": "", "stderr": "Process terminated unexpectedly.", "result": None}
