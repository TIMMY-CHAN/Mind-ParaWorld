"""
flashrag_search.py — FlashRAG 检索工具

通过 HTTP 请求调用本地 FlashRAG 服务（默认 http://localhost:6006），
将用户 query 检索为 top-k 文本段落后以 <tool_response> 格式返回。

使用方式：
    from verl.workers.agent.envs.tools.flashrag_search import FlashRAGSearchTool

    tool = FlashRAGSearchTool(server_url="http://localhost:6006")

    # 直接集成到 AgentEval
    from verl.workers.agent.envs.agent_eval import AgentEval
    agent = AgentEval(tools=[FlashRAGSearchTool()])
"""

import re
import json
import requests
from typing import Any, Dict, List, Optional, Tuple, Union

from verl.workers.agent.tool_envs import ToolBase


_DEFAULT_SERVER_URL = "http://localhost:6006"
_DEFAULT_TOPK = 3
_MIN_SEARCH_SCORE = 0.0


class FlashRAGSearchTool(ToolBase):
    """
    基于 FlashRAG 的检索工具。

    调用格式（default parser）：
        <tool_call>{"name": "flashrag_search", "arguments": {"query": "your query"}}</tool_call>

    也支持 query 为字符串列表，返回第一条结果。
    """

    name = "flashrag_search"

    def __init__(self, server_url: str = _DEFAULT_SERVER_URL, topk: int = _DEFAULT_TOPK, **kwargs):
        super().__init__(
            name=self.name,
            description="Retrieve relevant text passages from a FlashRAG knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    }
                },
                "required": ["query"],
            },
        )
        self.server_url = server_url.rstrip("/")
        self.topk = topk

    # ------------------------------------------------------------------
    # ToolBase interface
    # ------------------------------------------------------------------

    def execute(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        """同步执行检索，返回 (obs, reward, done, info)。"""
        # 答案短路
        if re.search(r'<answer>.*?</answer>', action_string, re.DOTALL):
            return "", 0.0, True, {}

        action = self._extract_action(action_string)
        if not action:
            return "", 0.0, True, {}

        try:
            tool_call = json.loads(action.strip())
            query = tool_call["arguments"]["query"]

            if isinstance(query, list):
                queries = query
            elif isinstance(query, str):
                queries = [query]
            else:
                raise ValueError(f"'query' must be str or list, got {type(query)}")

            docs, _ = self._batch_search(queries, top_n=self.topk)
            if not docs:
                raise ValueError("FlashRAG returned no results.")

            doc_content = docs[0].get("contents", "")
            obs = (
                "\n<|im_start|>user\n"
                f"<tool_response>{doc_content}</tool_response>\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            return obs, 0.0, False, {"contents": doc_content}

        except Exception as e:
            obs = f"\n<|im_start|>user\nError: {e}<|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e), "status": "failed"}

    async def execute_async(self, action_string: str, **kwargs) -> Tuple[str, float, bool, Dict]:
        """异步执行（通过线程池包装同步请求）。"""
        import asyncio
        return await asyncio.to_thread(self.execute, action_string, **kwargs)

    def reset(self, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_action(self, text: str) -> Optional[str]:
        matches = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        return matches[-1] if matches else None

    def _batch_search(
        self,
        questions: List[str],
        top_n: int = 3,
    ) -> Tuple[List[Dict], List[float]]:
        """
        调用 FlashRAG 服务的 /batch_search 接口。

        Returns:
            (documents_list, scores_list) — 每项对应一个 query 的 top 结果列表。
            失败时返回 ([], [])。
        """
        payload = {
            "query": questions,
            "top_n": top_n,
            "return_score": True,
        }
        try:
            response = requests.post(
                f"{self.server_url}/batch_search",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            documents_list: List[Dict] = []
            scores_list: List[float] = []
            for docs_data, score in zip(result[0][0], result[1][0]):
                if score >= _MIN_SEARCH_SCORE:
                    documents_list.append(docs_data)
                    scores_list.append(score)
            return documents_list, scores_list

        except requests.exceptions.RequestException as e:
            print(f"[FlashRAGSearchTool] HTTP error: {e}")
            return [], []
        except Exception as e:
            print(f"[FlashRAGSearchTool] Unexpected error: {e}")
            return [], []
