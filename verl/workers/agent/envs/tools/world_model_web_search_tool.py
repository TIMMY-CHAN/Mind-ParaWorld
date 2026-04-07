# world_model_web_search_tool.py
# 评测专用版本：支持命中日志记录，用于 Parallel World Benchmark

import json
import re
import os
import hashlib
from typing import Dict, Any, Tuple, Optional, List, Union
from openai import AsyncOpenAI
import httpx
import random
import asyncio

# ================= Prompt 清洗 =================

AGENT_INSTRUCTION_NOISE = """
请根据上下文选择你下一步的动作：思考，调用工具，或者给出你的答案；
请注意如果你没有足够的信息回答问题或仍不确认你的答案是否正确，你可以选择继续思考，或选择当前可用的工具解决问题；
请在你打算输出'<answer> 你给出的答案 </answer>'前，确认你是要给出最终答案，而不是还需要**进一步思考或调用工具**，一旦你输出答案，所有对话将结束；
在输出答案之前，请根据上下文验证你的推理流程是否正确
请记住遵循system prompt的格式要求，不要输出非法格式：
<think>
(这里输入你的思考过程）
</think>

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call> (如果需要调用工具,你的工具调用请求参考system prompt usage中的示例)

<answer>
(这里输入你的最终答案）
</answer>
"""

# ================= 世界模型 Prompt（增强版）=================

WORLD_MODEL_SYSTEM_PROMPT = """
# 角色与背景
你是一个在AI Agent离线训练环境中，扮演"动态信息编造者"的LLM。你的任务是充当一个真实但"有偏见"的搜索引擎。

**背景**: 我们正在评测一个ReAct范式的Agent。这个Agent很"懒惰"，倾向于直接搜索复杂问题，而不会将其分解。你的工作是创造一个信息环境：
- "懒惰"的搜索策略会碰壁（得到无用或误导的信息）
- 经过思考、分解后的"聪明"搜索策略则会得到奖励（获得准确信息）

你的【唯一】事实来源是我提供给你的【世界真相】JSON。你必须严格遵循以下规则。

---
# 核心规则

我将为你提供五部分输入：
1. 【**原始问题**】: Agent训练时收到的原始问题。
2. 【**世界真相**】: 一个JSON对象，定义了当前模拟世界中所有不可违背的原子事实。
3. 【**问题答案**】: 原始问题在设定的原子事实下的答案。
4. 【**Agent轨迹**】: Agent到目前为止的完整思考和行动历史。
5. 【**当前查询**】: Agent本次给你的具体查询字符串。

你必须首先判断【当前查询】是否是复合查询，然后执行对应的动作。

---
## 查询类型判断

**你需要判断查询是否是"复合查询"（Compound Query）或低质量查询**：

* **复合查询 (Compound Query)或低质量查询** - 是指：
    * 查询质量低
    * 一个包含**多个实体**的查询（如 "莱万和哈兰德谁进球多？"）
    * 一个包含**多个属性**的查询（如 "Canon EOS R6 Mark III 的重量和价格"）
    * 一个包含**比较、分析、推理**的查询（如 "哪个相机更好？"）
    * 一个极为简单不加定语，在真实环境中明显不可能直接得到所需搜索结果的查询

* **原子查询 (Atomic Query)** - 是指：
    * 查询字符串是一个**只包含单个实体或实体属性的短语**或**针对单个实体或单个实体属性的查询问题**
    * 这个查询**可以通过查找单个原子事实直接回答**（无论该事实是否存在于【世界真相】中）
    * **示例**: "罗伯特·莱万多夫斯基 欧冠总进球数", "小米SU7的官方售价是多少"

⚠️ **重要区分**：
- 原子查询在【世界真相】中**找不到对应事实** ≠ 复合查询
- 判断是否复合查询，只看**查询本身的结构**，不看是否能匹配到【世界真相】json中的条目

---
## 响应策略

### 动作A：响应原子查询

**步骤 1**: 尝试从【世界真相】中找到对应的事实
- **如果找到**: hit=1，在 matched_fact_keys 中返回匹配的键名
- **如果找不到**: hit=0，matched_fact_keys=[]

**步骤 2**: 生成 **4条** 搜索结果
- **如果 hit=1**: 必须有一条结果是"真相条目"，其`content`明确包含对应的真相值，其他3条是相关背景信息
- **如果 hit=0**: 4条结果都应该是相关但不包含具体事实的背景信息

**步骤 3**: 所有条目的`date`都应编造一个合理的未来日期（2026年之后）

**注意**: Agent有时会生成出人意料的原子查询，可能不完全匹配【世界真相】中的键名。此时你需要灵活判断，如果该查询确实是不可拆解的原子查询，但找不到对应事实，应该返回 hit=0。

### 动作B：响应复合查询和低质量查询

1. **必须**生成 **4条** 搜索结果
2. **3条"中等相关度、模棱两可"的结果**: 内容是分析、评论、比较或猜测性的，**严禁泄露【世界真相】中的任何具体数值或事实**
3. **1条"低等相关度、闲聊式"的结果**: 内容可以是实体的趣闻、历史等外围信息
4. 每个新闻必须足够完整（不少于50个token）
5. **强制**: hit=0，matched_fact_keys=[]

---
# 输出格式要求 【重要 - 用于评测统计】

**你必须在返回搜索结果的同时，返回命中日志（用于评估Agent性能）**

你的输出必须是一个JSON对象，严格包含以下字段：

```json
{
  "is_compound_query": true 或 false,
  "hit": 0 或 1,
  "matched_fact_keys": ["事实键名"],
  "reason": "一句话说明判断依据（可选）",
  "search_result": [
    {
      "id": 1,
      "title": "搜索结果标题",
      "content": "搜索结果的具体内容",
      "date": "YYYY-MM-DD"
    }
  ]
}
```

**关键规则**：
- **is_compound_query**:
  - true: 复合查询（多个实体/属性/比较）或低质量查询 → 强制 hit=0, matched_fact_keys=[]
  - false: 原子查询（单个实体/属性）→ hit 和 matched_fact_keys 取决于是否能匹配到【世界真相】

- **hit 和 matched_fact_keys** (仅当 is_compound_query=false 时有意义):
  - 如果在【世界真相】中找到对应事实：hit=1，matched_fact_keys 包含【一个】匹配的键名（精确复制）
  - 如果在【世界真相】中找不到对应事实：hit=0，matched_fact_keys=[]
  - ⚠️ 重要：matched_fact_keys 数组中最多只能有一个元素

---
# 开始执行

现在，请根据我给你的输入，严格执行你的任务。
"""

WORLD_MODEL_USER_TEMPLATE = """
【原始问题】:
{ORIGINAL_QUESTION}

【世界真相】:
{WORLD_TRUTH_JSON}

【问题答案】:
{GROUND_TRUTH}

【Agent轨迹】:
{AGENT_TRAJECTORY}

【当前查询】:
"{AGENT_QUERY}"
"""

# ================= 核心类 =================

class WorldModelWebSearchTool:
    """评测专用 RAG Engine，支持命中日志记录"""

    name = "web_search"
    user_prompt = ''

    def __init__(self, _name=None, _desc=None, _params=None,
                 endpoints: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        """
        初始化评测 RAG Engine。

        Args:
            endpoints: 世界模型节点地址。支持以下形式：
                - None（默认）: 从环境变量 WORLD_MODEL_ENDPOINTS 读取，
                               若未设置则使用 "http://localhost:8000/v1"
                - str: 单个地址，或逗号分隔的多个地址
                       例如 "http://host1:8000/v1,http://host2:8000/v1"
                - list: 地址列表
                       例如 ["http://host1:8000/v1", "http://host2:8000/v1"]
        """
        # ── 世界模型节点配置 ──
        if endpoints is not None:
            # 显式传参优先
            if isinstance(endpoints, str):
                endpoint_list = [url.strip() for url in endpoints.split(',') if url.strip()]
            else:
                endpoint_list = [url.strip() for url in endpoints if url.strip()]
        else:
            # 回退到环境变量
            endpoints_env = os.environ.get("WORLD_MODEL_ENDPOINTS", "http://localhost:8000/v1")
            endpoint_list = [url.strip() for url in endpoints_env.split(',') if url.strip()]

        self.world_model_endpoints = endpoint_list

        # 获取模型名称（从第一个可用节点）
        self.model_name = None
        for probe_url in self.world_model_endpoints:
            try:
                resp = httpx.get(f"{probe_url}/models", timeout=5)
                if resp.status_code == 200:
                    self.model_name = resp.json()["data"][0]["id"]
                    break
            except Exception:
                pass

        if self.model_name is None:
            self.model_name = "Qwen/Qwen3-VL-235B-A22B-Thinking"

        # 创建客户端池
        timeout_config = httpx.Timeout(connect=100.0, read=3000.0, write=100.0, pool=100.0)
        self.async_clients = []
        for url in self.world_model_endpoints:
            client = AsyncOpenAI(
                base_url=url,
                api_key="EMPTY",
                timeout=timeout_config,
                max_retries=0
            )
            self.async_clients.append(client)

        self.enable_thinking = True
        self.world_truth = {}  # 在 reset 时设置

        # 【性能优化】World Model 响应缓存
        self.world_model_cache: Dict[str, Tuple[str, float, bool, Dict]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def _clean_trajectory(self, trajectory: str) -> str:
        """清洗 Agent 历史中的冗余指令"""
        if not trajectory:
            return ""
        cleaned = trajectory.replace(AGENT_INSTRUCTION_NOISE.strip(), "")
        return cleaned

    def _remove_think_block(self, text: str) -> str:
        """移除 <think>...</think> 及其内容"""
        if not text:
            return ""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.replace('```json', '').replace('```', '')
        return cleaned.strip()

    def _return_error_observation(self, error_msg):
        return f"\n<|im_start|>user\n<tool_response>Error: {error_msg}</tool_response>\n{self.user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

    def _save_parse_failure_log(self, response_content: str, json_str: str, error: Exception, attempts: list):
        """
        保存 JSON 解析失败的详细日志到文件

        Args:
            response_content: World Model 原始返回内容
            json_str: 提取的 JSON 字符串
            error: 最后的错误信息
            attempts: 所有尝试的字符串列表
        """
        from datetime import datetime

        # 日志目录（相对于本文件位置）
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "json_parse_errors")
        os.makedirs(log_dir, exist_ok=True)

        # 生成日志文件名（按时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(log_dir, f"parse_error_{timestamp}.log")

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("JSON 解析失败详细日志\n")
                f.write("="*80 + "\n\n")

                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"错误类型: {type(error).__name__}\n")
                f.write(f"错误信息: {str(error)}\n")
                f.write(f"尝试次数: {len(attempts)}\n\n")

                f.write("-"*80 + "\n")
                f.write("原始 World Model 返回内容:\n")
                f.write("-"*80 + "\n")
                f.write(response_content)
                f.write("\n\n")

                f.write("-"*80 + "\n")
                f.write("提取的 JSON 字符串:\n")
                f.write("-"*80 + "\n")
                f.write(json_str)
                f.write("\n\n")

                f.write("-"*80 + "\n")
                f.write("所有尝试的解析字符串:\n")
                f.write("-"*80 + "\n")
                for idx, attempt in enumerate(attempts):
                    f.write(f"\n--- 尝试 #{idx+1} ---\n")
                    f.write(attempt[:500])  # 只保存前500字符
                    if len(attempt) > 500:
                        f.write(f"\n... (总长度: {len(attempt)} 字符)")
                    f.write("\n")

                f.write("\n" + "="*80 + "\n")

            print(f"[INFO Eval] 解析失败日志已保存: {log_file}")

        except Exception as e:
            print(f"[WARN Eval] 保存解析失败日志时出错: {e}")

    def _extract_json_from_text(self, text: str) -> str:
        """
        从混杂的文本中提取 JSON 内容（增强版）
        支持多种格式：
        1. 纯 JSON: {"key": "value"}
        2. Markdown 包裹: ```json\n{...}\n```
        3. 带前后文: "文字说明\n```json\n{...}\n```\n更多文字"
        """
        if not text:
            return ""

        # 策略 1: 尝试提取 markdown 代码块中的 JSON
        # 匹配 ```json ... ``` 或 ``` ... ```
        markdown_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json\n{...}\n```
            r'```\s*\n(.*?)\n```',       # ```\n{...}\n```
            r'```json(.*?)```',           # ```json{...}```（无换行）
            r'```(.*?)```'                # ```{...}```（无换行）
        ]

        for pattern in markdown_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_candidate = match.group(1).strip()
                # 验证是否是有效的 JSON（以 { 开头）
                if json_candidate.startswith('{'):
                    return json_candidate

        # 策略 2: 查找最外层的 {} 包裹的内容
        # 从第一个 { 到最后一个 } 配对
        stack = []
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start_idx != -1:
                        # 找到了完整的 JSON
                        json_candidate = text[start_idx:i+1]
                        return json_candidate

        # 策略 3: 如果以上都失败，尝试直接返回（可能本身就是 JSON）
        return text.strip()

    def _try_fix_truncated_json(self, json_str: str) -> str:
        """
        尝试修复被截断的 JSON

        常见截断模式：
        1. "date": "2027-01-15   <- 缺少右引号和后续结构
        2. 缺少数组/对象的闭合符号
        """
        if not json_str:
            return json_str

        # 检查是否可能被截断（没有正常结束）
        stripped = json_str.rstrip()

        # 如果最后没有 } 或 ]，可能被截断
        if stripped and stripped[-1] not in ['}', ']']:
            print(f"[INFO Eval] 检测到可能的截断 JSON，尝试修复...")

            # 策略 1: 查找未闭合的字符串（缺少右引号）
            # 例如：  "date": "2027-01-15
            if '"' in stripped:
                # 统计引号数量
                quote_count = stripped.count('"')
                if quote_count % 2 == 1:  # 奇数个引号，说明有未闭合的字符串
                    stripped += '"'
                    print(f"[INFO Eval] 补全未闭合的字符串引号")

            # 策略 2: 补全缺失的结构
            # 假设是 search_result 数组中的一个对象被截断
            # 需要补全：}（对象）、]（数组）、}（最外层）

            # 统计括号
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            open_brackets = stripped.count('[')
            close_brackets = stripped.count(']')

            # 补全缺失的闭合符号
            # 优先补全对象，然后数组，最后最外层对象
            missing_close_braces = open_braces - close_braces
            missing_close_brackets = open_brackets - close_brackets

            # 根据结构补全
            # 典型结构：{ ... "search_result": [ {...}, {...} ] }
            # 如果截断在 search_result 的某个对象中，需要：}, ], }

            if missing_close_braces > 0 or missing_close_brackets > 0:
                # 补全对象
                if missing_close_braces > 0:
                    stripped += '\n      }'  # 对象闭合
                    missing_close_braces -= 1
                    print(f"[INFO Eval] 补全对象闭合符号")

                # 补全数组
                if missing_close_brackets > 0:
                    stripped += '\n  ]'  # 数组闭合
                    missing_close_brackets -= 1
                    print(f"[INFO Eval] 补全数组闭合符号")

                # 补全最外层
                if missing_close_braces > 0:
                    stripped += '\n}'  # 最外层对象闭合
                    print(f"[INFO Eval] 补全最外层对象闭合符号")

        return stripped

    def _parse_world_model_response(self, response_content: str) -> Dict[str, Any]:
        """
        解析世界模型的返回，提取命中日志（增强版）
        支持从混杂文本中提取 JSON
        """
        # 1. 移除 think 块
        cleaned = self._remove_think_block(response_content)

        # 2. 尝试多种方式提取 JSON
        json_str = self._extract_json_from_text(cleaned)

        # 2.5 【新增】尝试修复截断的 JSON
        json_str = self._try_fix_truncated_json(json_str)

        # 3. 尝试解析 JSON（增强容错）
        parse_attempts = [
            json_str,  # 原始提取结果
            json_str.strip(),  # 去除首尾空白
            re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str),  # 移除控制字符
            re.sub(r',\s*}', '}', json_str),  # 移除对象尾随逗号
            re.sub(r',\s*]', ']', json_str),  # 移除数组尾随逗号
            re.sub(r',\s*}', '}', re.sub(r',\s*]', ']', json_str)),  # 同时移除两种尾随逗号
            json_str.replace("'", '"'),  # 单引号转双引号
        ]

        last_error = None
        for attempt_idx, attempt_str in enumerate(parse_attempts):
            try:
                result = json.loads(attempt_str)

                # 验证必需字段
                required_fields = ["is_compound_query", "hit", "matched_fact_keys", "search_result"]
                for field in required_fields:
                    if field not in result:
                        print(f"[WARN Eval] 世界模型返回缺少字段: {field}")
                        if field == "matched_fact_keys":
                            result[field] = []
                        elif field == "hit":
                            result[field] = 0
                        elif field == "is_compound_query":
                            result[field] = True  # 缺失时默认为复合查询（保守策略）
                        elif field == "search_result":
                            result[field] = []

                # 【防御性编程】验证 matched_fact_keys 长度
                matched_keys = result.get("matched_fact_keys", [])
                if not isinstance(matched_keys, list):
                    print(f"[WARN Eval] matched_fact_keys 不是列表: {type(matched_keys)}, 转换为列表")
                    result["matched_fact_keys"] = [matched_keys] if matched_keys else []
                    matched_keys = result["matched_fact_keys"]

                # 检查是否违反"单次只返回一个键"的规则
                if len(matched_keys) > 1:
                    print(f"[WARN Eval] matched_fact_keys 包含多个键 ({len(matched_keys)}): {matched_keys}")
                    print(f"[WARN Eval] 根据原子查询原则，只保留第一个键: {matched_keys[0]}")
                    result["matched_fact_keys"] = [matched_keys[0]]

                # 成功解析
                if attempt_idx > 0:
                    print(f"[INFO Eval] JSON 解析成功（尝试 #{attempt_idx+1}）")
                return result

            except json.JSONDecodeError as e:
                last_error = e
                continue

        # 4. 所有尝试都失败，保存详细日志并返回默认结构
        print(f"[ERROR Eval] 世界模型返回的 JSON 解析失败（尝试了 {len(parse_attempts)} 种方法）")
        print(f"[DEBUG Eval] 最后错误: {last_error}")
        print(f"[DEBUG Eval] 原始内容前300字符:\n{response_content[:300]}")
        print(f"[DEBUG Eval] 提取的 JSON 前200字符:\n{json_str[:200]}")

        # 【新增】保存完整的失败案例到日志文件
        self._save_parse_failure_log(response_content, json_str, last_error, parse_attempts)

        # 返回默认结构
        return {
            "is_compound_query": True,  # 解析失败默认为复合查询
            "hit": 0,
            "matched_fact_keys": [],
            "reason": f"JSON parse error after {len(parse_attempts)} attempts: {str(last_error)}",
            "search_result": []
        }

    def _get_cache_key(self, query: str, world_truth: Dict[str, Any]) -> str:
        """
        生成缓存键

        基于 query + atomic_facts 的哈希，确保相同查询+相同世界真相返回相同结果
        """
        facts_str = json.dumps(world_truth.get('atomic_facts', {}), sort_keys=True, ensure_ascii=False)
        cache_str = f"{query}|{facts_str}"
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()

    def _check_cache(self, query: str, world_truth: Dict[str, Any]) -> Optional[Tuple[str, float, bool, Dict]]:
        """检查缓存"""
        cache_key = self._get_cache_key(query, world_truth)
        if cache_key in self.world_model_cache:
            self.cache_hits += 1
            if self.cache_hits % 10 == 0:  # 每 10 次命中打印一次
                hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
                print(f"[Cache] Hits={self.cache_hits}, Misses={self.cache_misses}, Hit Rate={hit_rate:.1%}")
            return self.world_model_cache[cache_key]
        else:
            self.cache_misses += 1
            return None

    def _save_cache(self, query: str, world_truth: Dict[str, Any], result: Tuple[str, float, bool, Dict]):
        """保存到缓存"""
        cache_key = self._get_cache_key(query, world_truth)
        self.world_model_cache[cache_key] = result

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_calls = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_calls if total_calls > 0 else 0.0
        return {
            "cache_size": len(self.world_model_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }

    async def execute_async(self, action_string, **kwargs):
        """
        异步执行工具调用

        Returns:
            observation (str): 返回给 Agent 的观察文本
            reward (float): 奖励（固定为0，由外部评测脚本计算）
            done (bool): 是否结束
            info (dict): 包含命中日志的额外信息 {'hit_log': {...}}
        """
        # 1. 基础解析
        try:
            action_content = self.extract_action(action_string)
            if not action_content:
                return self._return_error_observation("No valid <tool_call> found."), 0.0, False, {}

            tool_call = json.loads(action_content.strip())
            query = tool_call.get("arguments", {}).get('query')
            if tool_call.get("name") != self.name or not query:
                raise ValueError("Tool name mismatch or query is missing.")
        except Exception as e:
            return self._return_error_observation(f"Invalid tool call format: {e}"), 0.0, False, {"error": str(e)}

        # 2. 核心逻辑：调用世界模型
        try:
            agent_trajectory = kwargs.get("agent_trajectory", "")
            world_truth_info = kwargs.get("world_truth", {}) or self.world_truth

            # 【性能优化】检查缓存
            cached_result = self._check_cache(query, world_truth_info)
            if cached_result is not None:
                obs, reward, done, info = cached_result
                # 返回缓存结果（需要深拷贝 info，避免修改缓存）
                return obs, reward, done, dict(info)

            # 缓存未命中，调用 World Model
            if not world_truth_info:
                original_question = "未知问题"
                world_truth_json = "{}"
                final_answer = ""
            else:
                original_question = world_truth_info.get("generated_question", "")
                atomic_facts = world_truth_info.get("atomic_facts", {})
                final_answer = world_truth_info.get("final_answer", "")

                if isinstance(atomic_facts, str):
                    try:
                        atomic_facts = json.loads(atomic_facts)
                    except:
                        pass

                world_truth_json = json.dumps(atomic_facts, ensure_ascii=False, indent=2)

            clean_trajectory = self._clean_trajectory(agent_trajectory)
            if len(clean_trajectory) > 30000:
                clean_trajectory = "...(context truncated)..." + clean_trajectory[-20000:]

            user_content = WORLD_MODEL_USER_TEMPLATE.format(
                ORIGINAL_QUESTION=original_question,
                WORLD_TRUTH_JSON=world_truth_json,
                GROUND_TRUTH=final_answer,
                AGENT_TRAJECTORY=clean_trajectory,
                AGENT_QUERY=query
            )

            messages = [
                {"role": "system", "content": WORLD_MODEL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]

            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 8192,  # 【修复】从 4096 增加到 8192，避免截断
            }
            if self.enable_thinking:
                api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

            # 3. 调用世界模型（带重试 + 负载均衡）
            # - infra 层默认允许重试
            # - 对临时性错误（timeout/5xx/429/连接失败）指数退避
            # - 对明显不可重试错误（如 400/invalid/参数问题）快速失败
            MAX_RETRIES = 5
            last_error: Optional[Exception] = None

            for attempt in range(MAX_RETRIES):
                try:
                    client = random.choice(self.async_clients)
                    chat_response = await client.chat.completions.create(**api_params)
                    response_content = (chat_response.choices[0].message.content or "").strip()

                    # 空响应按 infra 错误处理，触发重试
                    if not response_content:
                        raise RuntimeError("Empty world model response")

                    # 【调试】检查是否被截断
                    finish_reason = chat_response.choices[0].finish_reason
                    if finish_reason == "length":
                        print(f"[WARN Eval] World Model 输出被截断！(finish_reason=length)")

                    # 4. 解析返回
                    parsed_result = self._parse_world_model_response(response_content)

                    # 5. 构造给 Agent 的观察（只包含搜索结果）
                    search_results = parsed_result.get("search_result", [])
                    observation_json = {
                        "search_query": query,
                        "search_result": search_results
                    }
                    observation_str = json.dumps(observation_json, ensure_ascii=False)

                    # 6. 包装为 chat 格式
                    obs = f"\n<|im_start|>user\n<tool_response>{observation_str}</tool_response>\n{self.user_prompt}\n<|im_end|>\n<|im_start|>assistant\n"

                    # 7. 构造命中日志（用于统计）
                    hit_log = {
                        "query": query,
                        "is_compound_query": parsed_result.get("is_compound_query", True),
                        "hit": parsed_result.get("hit", 0),
                        "matched_fact_keys": parsed_result.get("matched_fact_keys", []),
                        "reason": parsed_result.get("reason", ""),
                    }

                    # 8. 【性能优化】保存到缓存
                    result = (obs, 0.0, False, {"hit_log": hit_log})
                    self._save_cache(query, world_truth_info, result)

                    return obs, 0.0, False, {"hit_log": hit_log}

                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    error_lower = error_str.lower()
                    print(f"[Warn Eval] RAG Request failed (Attempt {attempt+1}/{MAX_RETRIES}): {error_str}")

                    # 不可重试：明显的参数/请求错误
                    if "400" in error_str or "invalid" in error_lower:
                        break

                    # 其余按 infra 临时故障处理：指数退避
                    await asyncio.sleep(1 * (2 ** attempt))

            # 如果重试耗尽
            print(f"[Error Eval] RAG Request failed after retries. Last error: {last_error}")
            return self._return_error_observation(f"Internal Service Error: {str(last_error)}"), 0.0, False, {
                "error": str(last_error),
                "hit_log": {
                    "query": query,
                    "is_compound_query": True,  # API错误默认为复合查询
                    "hit": 0,
                    "matched_fact_keys": [],
                    "reason": f"API Error: {str(last_error)}"
                }
            }

        except Exception as e:
            error_type = type(e).__name__
            print(f'[ERROR Eval] Async Execute WRONG - {error_type}: {e}')
            return self._return_error_observation(f"Internal Error: {str(e)}"), 0.0, False, {
                "error": str(e),
                "hit_log": {
                    "query": query if 'query' in locals() else "unknown",
                    "is_compound_query": True,  # 内部错误默认为复合查询
                    "hit": 0,
                    "matched_fact_keys": [],
                    "reason": f"Internal Error: {str(e)}"
                }
            }

    def reset(self, *args, **kwargs):
        """在评测时接收 world_truth 信息"""
        self.world_truth = kwargs.get("world_truth", {})
        pass


# ================= 测试代码 =================

async def test_world_model_web_search_tool():
    print("\n" + "="*70)
    print("  测试评测版 RAG Engine")
    print("="*70 + "\n")

    engine = WorldModelWebSearchTool()

    # 模拟数据
    world_truth_info = {
        "generated_question": "在2026-27赛季的西班牙甲级联赛中，拉明·亚马尔和基利安·姆巴佩谁的联赛进球数更多？",
        "atomic_facts": {
            "拉明·亚马尔在2026-27赛季西甲联赛的进球数": "15",
            "基利安·姆巴佩在2026-27赛季西甲联赛的进球数": "20"
        },
        "final_answer": "在2026-27赛季的西班牙甲级联赛中，基利安·姆巴佩的联赛进球数（20球）比拉明·亚马尔（15球）更多。"
    }

    engine.reset(world_truth=world_truth_info)

    # Test Case 1: 低质量查询
    print(">>> Test Case 1: 低质量查询（复杂问题直接搜索）")
    action1 = '<tool_call>{"name": "web_search", "arguments": {"query": "在2026-27赛季的西班牙甲级联赛中，拉明·亚马尔和基利安·姆巴佩谁的联赛进球数更多？"}}</tool_call>'
    obs1, reward1, done1, info1 = await engine.execute_async(
        action_string=action1,
        agent_trajectory="",
        world_truth=world_truth_info
    )
    print(f"观察结果（前200字符）: {obs1[:200]}...")
    print(f"命中日志: {json.dumps(info1.get('hit_log', {}), ensure_ascii=False, indent=2)}")
    print("-" * 70)

    # Test Case 2: 高质量查询
    print("\n>>> Test Case 2: 高质量查询（原子化查询）")
    action2 = '<tool_call>{"name": "web_search", "arguments": {"query": "基利安·姆巴佩 2026-27赛季 西甲联赛进球数"}}</tool_call>'
    obs2, reward2, done2, info2 = await engine.execute_async(
        action_string=action2,
        agent_trajectory="",
        world_truth=world_truth_info
    )
    print(f"观察结果（前200字符）: {obs2[:200]}...")
    print(f"命中日志: {json.dumps(info2.get('hit_log', {}), ensure_ascii=False, indent=2)}")
    print("-" * 70)

    print("\n" + "="*70)
    print("  测试完成")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_world_model_web_search_tool())
