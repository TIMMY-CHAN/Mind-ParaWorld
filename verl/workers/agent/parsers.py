"""
parsers.py - 可插拔 tool call 解析器

每个 Parser 负责三件事：
  1. extract_action(text) -> Optional[str]
     从模型输出中提取工具调用（最后一个），并归一化为框架内部统一格式：
     {"name": "tool_name", "arguments": {...}}
     返回 JSON 字符串，解析失败返回 None。

  2. extract_all_actions(text) -> List[str]
     提取全部工具调用（支持一次输出多个 tool_call），返回 JSON 字符串列表。
     默认实现调用 extract_action 并包装为单元素列表；DefaultParser 覆盖以返回所有匹配。

  3. extract_answer(text) -> Optional[str]
     提取最终答案（<answer>...</answer>），与模型无关，统一处理。

内置 Parser：
  - "default"  : <tool_call>{"name":..., "arguments":{...}}</tool_call>
                 适用于 Hermes / NousResearch / Qwen3 / 本框架默认 prompt
  - "deepseek" : DeepSeek-V2 / V3 / R1 的专用 token 格式
  - "glm4"     : GLM-4 / GLM-Z1 的 XML 键值对格式（来源：GLM-4-7B-Flash chat_template.jinja）
  - "minimax"  : MiniMax-M2.5 的 <minimax:tool_call> XML invoke 格式
  - "qwen35"   : Qwen3.5 的 <function=name> / <parameter=key> 格式
  - "kimi_k2"  : Kimi-K2 的专用 token 格式（见类文档关于函数名限制的说明）

自定义 Parser 示例：

    from verl.workers.agent.parsers import ToolCallParser, PARSER_REGISTRY
    import re, json

    class MyParser(ToolCallParser):
        name = "my_parser"

        def extract_action(self, text):
            # 解析并归一化为 {"name": ..., "arguments": {...}}
            ...

        def extract_answer(self, text):
            ...

    PARSER_REGISTRY["my_parser"] = MyParser

注意：Parser 与 System Prompt 是绑定关系。
使用非默认 Parser 时，System Prompt 中的工具调用格式示例必须与所选 Parser 匹配。
"""

import re
import json
from typing import Optional, List


# ================= 基类 =================

class ToolCallParser:
    """tool call 解析器基类，子类需实现 extract_action 和 extract_answer。"""

    name: str

    def extract_action(self, text: str) -> Optional[str]:
        """
        从模型输出中提取工具调用。

        Returns:
            归一化的 JSON 字符串，格式为 {"name": "...", "arguments": {...}}；
            未找到或解析失败返回 None。
        """
        raise NotImplementedError

    def extract_all_actions(self, text: str) -> List[str]:
        """
        从模型输出中提取所有工具调用（支持一次输出多个 tool_call）。

        Returns:
            归一化的 JSON 字符串列表；未找到返回空列表。

        默认实现调用 extract_action，返回单元素列表（向后兼容）。
        子类可覆盖以支持真正的多工具并行调用。
        """
        action = self.extract_action(text)
        return [action] if action is not None else []

    def extract_answer(self, text: str) -> Optional[str]:
        """
        从模型输出中提取最终答案。

        Returns:
            答案字符串（已 strip）；未找到返回 None。
        """
        raise NotImplementedError


# ================= 内置 Parser =================

class DefaultParser(ToolCallParser):
    """
    默认 parser：<tool_call>{"name":..., "arguments":{...}}</tool_call>

    适用于：Hermes / NousResearch / Qwen3 / 本框架默认 prompt。

    兼容两种写法：
      1) 多行：<tool_call>\\n{...}\\n</tool_call>
      2) 单行：<tool_call>{...}</tool_call>
    """

    name = "default"

    def extract_action(self, text: str) -> Optional[str]:
        # 兼容两种 tool_call 写法：
        # 1) 多行：<tool_call>\n{...}\n</tool_call>
        # 2) 单行：<tool_call>{...}</tool_call>
        # 优先提取最后一个 tool_call（允许一次输出多个 tool_call）
        matches = re.findall(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL
        )
        return matches[-1] if matches else None

    def extract_all_actions(self, text: str) -> List[str]:
        """返回所有 tool_call 的 JSON 字符串列表，支持并行工具调用。"""
        return re.findall(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL
        )

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


class DeepSeekParser(ToolCallParser):
    """
    DeepSeek-V2 / V3 / R1 专用 token 格式：

        <|tool▁calls▁begin|><|tool▁call▁begin|>function<|tool▁sep|>tool_name
        ```json
        {"arg": "value"}
        ```<|tool▁call▁end|><|tool▁calls▁end|>

    注意：token 中的 ▁ 为 U+2581（LOWER ONE EIGHTH BLOCK），
    是 SentencePiece tokenizer 的词边界标记，解码后在文本中原样出现。
    """

    name = "deepseek"

    _PATTERN = re.compile(
        r'<\|tool▁call▁begin\|>function'
        r'<\|tool▁sep\|>([^\n]+)\n'
        r'```(?:json)?\n(.*?)\n```'
        r'<\|tool▁call▁end\|>',
        re.DOTALL,
    )

    def extract_action(self, text: str) -> Optional[str]:
        matches = self._PATTERN.findall(text)
        if not matches:
            return None
        tool_name, args_json = matches[-1]
        tool_name = tool_name.strip()
        try:
            args = json.loads(args_json.strip())
        except json.JSONDecodeError:
            return None
        return json.dumps({"name": tool_name, "arguments": args}, ensure_ascii=False)

    def extract_all_actions(self, text: str) -> List[str]:
        result = []
        for tool_name, args_json in self._PATTERN.findall(text):
            try:
                args = json.loads(args_json.strip())
                result.append(
                    json.dumps({"name": tool_name.strip(), "arguments": args}, ensure_ascii=False)
                )
            except json.JSONDecodeError:
                continue
        return result

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


class GLM4Parser(ToolCallParser):
    """
    GLM-4 / GLM-Z1 XML 键值对格式（来源：GLM-4-7B-Flash chat_template.jinja）：

        <tool_call>function_name
        <arg_key>key1</arg_key><arg_value>value1</arg_value>
        <arg_key>key2</arg_key><arg_value>value2</arg_value>
        </tool_call>

    与默认 parser 的区分点：tool_call 内容以函数名（非 {）开头。
    arg_value 内若为 JSON 序列化的非字符串值（如数字、布尔），会自动反序列化。
    """

    name = "glm4"

    def extract_action(self, text: str) -> Optional[str]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not matches:
            return None
        raw = matches[-1]

        # 函数名：<arg_key> 之前的内容
        name_match = re.match(r'^([^<{]+)', raw)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()

        keys = re.findall(r'<arg_key>(.*?)</arg_key>', raw)
        values = re.findall(r'<arg_value>(.*?)</arg_value>', raw)

        arguments = {}
        for k, v in zip(keys, values):
            # arg_value 可能是 tojson 序列化的非字符串（数字、布尔等）
            try:
                arguments[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                arguments[k] = v

        return json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)

    def extract_all_actions(self, text: str) -> List[str]:
        result = []
        for raw in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
            name_match = re.match(r'^([^<{]+)', raw)
            if not name_match:
                continue
            tool_name = name_match.group(1).strip()
            keys = re.findall(r'<arg_key>(.*?)</arg_key>', raw)
            values = re.findall(r'<arg_value>(.*?)</arg_value>', raw)
            arguments = {}
            for k, v in zip(keys, values):
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            result.append(json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False))
        return result

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


class MinimaxParser(ToolCallParser):
    """
    MiniMax-M2.5 XML invoke 格式（来源：Minimax-M2_5 chat-template.jinja）：

        <minimax:tool_call>
        <invoke name="tool-name">
        <parameter name="param-key">param-value</parameter>
        ...
        </invoke>
        </minimax:tool_call>

    parameter value 若为 JSON 序列化的非字符串值，会自动反序列化。
    """

    name = "minimax"

    def extract_action(self, text: str) -> Optional[str]:
        matches = re.findall(
            r'<minimax:tool_call>(.*?)</minimax:tool_call>', text, re.DOTALL
        )
        if not matches:
            return None
        raw = matches[-1]

        # 工具名
        name_match = re.search(r'<invoke\s+name=["\']([^"\']+)["\']>', raw)
        if not name_match:
            return None
        tool_name = name_match.group(1)

        # 参数：<parameter name="key">value</parameter>
        params = re.findall(
            r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
            raw, re.DOTALL
        )
        arguments = {}
        for k, v in params:
            v = v.strip()
            try:
                arguments[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                arguments[k] = v

        return json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)

    def extract_all_actions(self, text: str) -> List[str]:
        result = []
        for raw in re.findall(r'<minimax:tool_call>(.*?)</minimax:tool_call>', text, re.DOTALL):
            name_match = re.search(r'<invoke\s+name=["\']([^"\']+)["\']>', raw)
            if not name_match:
                continue
            tool_name = name_match.group(1)
            params = re.findall(
                r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
                raw, re.DOTALL
            )
            arguments = {}
            for k, v in params:
                v = v.strip()
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            result.append(json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False))
        return result

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


class Qwen35Parser(ToolCallParser):
    """
    Qwen3.5 <function=name> / <parameter=key> 格式（来源：Qwen3_5-0.8B chat_template.jinja）：

        <tool_call>
        <function=tool_name>
        <parameter=param_name>
        value
        </parameter>
        </function>
        </tool_call>

    注意：<tool_call> token 与 DefaultParser 相同，但内部结构完全不同；
    用户需通过 System Prompt 明确指定此格式，并配合 --parser qwen35 使用。
    parameter value 若为 JSON 序列化的非字符串值，会自动反序列化。
    """

    name = "qwen35"

    def extract_action(self, text: str) -> Optional[str]:
        matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not matches:
            return None
        raw = matches[-1]

        # 函数名：<function=name>
        name_match = re.search(r'<function=([^>]+)>', raw)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()

        # 参数：<parameter=key>value</parameter>
        params = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', raw, re.DOTALL)
        arguments = {}
        for k, v in params:
            k = k.strip()
            v = v.strip()
            try:
                arguments[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                arguments[k] = v

        return json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False)

    def extract_all_actions(self, text: str) -> List[str]:
        result = []
        for raw in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
            name_match = re.search(r'<function=([^>]+)>', raw)
            if not name_match:
                continue
            tool_name = name_match.group(1).strip()
            params = re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', raw, re.DOTALL)
            arguments = {}
            for k, v in params:
                k = k.strip()
                v = v.strip()
                try:
                    arguments[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    arguments[k] = v
            result.append(json.dumps({"name": tool_name, "arguments": arguments}, ensure_ascii=False))
        return result

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


class KimiK2Parser(ToolCallParser):
    """
    Kimi-K2 专用 token 格式（来源：kimi-k2 chat-template.jinja）：

        <|tool_calls_section_begin|>
        <|tool_call_begin|>call_id<|tool_call_argument_begin|>{"arg": "value"}<|tool_call_end|>
        <|tool_calls_section_end|>

    ⚠️ 函数名限制：
    Kimi-K2 的 chat_template 不在文本中输出函数名（只有 call_id 和参数 JSON），
    无法仅从文本可靠地恢复完整的 {"name": ..., "arguments": {...}} 结构。

    本 parser 通过 default_tool_name 参数作为兜底，默认为 "web_search"。
    适用于本框架的标准单工具场景（web_search）。

    若需要多工具支持，推荐以下替代方案：
      1. 在 System Prompt 中明确指示 Kimi 输出：
           <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
         然后使用 "default" parser。
      2. 修改推理循环以读取 API 结构化 tool_calls 字段（response.choices[0].message.tool_calls）。
    """

    name = "kimi_k2"

    _PATTERN = re.compile(
        r'<\|tool_call_begin\|>[^<]*<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>',
        re.DOTALL,
    )

    def __init__(self, default_tool_name: str = "web_search"):
        self._default_tool_name = default_tool_name

    def extract_action(self, text: str) -> Optional[str]:
        matches = self._PATTERN.findall(text)
        if not matches:
            return None
        args_json = matches[-1].strip()
        try:
            args = json.loads(args_json)
        except json.JSONDecodeError:
            return None
        return json.dumps(
            {"name": self._default_tool_name, "arguments": args},
            ensure_ascii=False
        )

    def extract_all_actions(self, text: str) -> List[str]:
        """
        ⚠️ 函数名限制：Kimi-K2 的文本中不含函数名，所有并行调用均以
        default_tool_name 兜底。仅适合单工具（web_search）并行查询场景。
        """
        result = []
        for args_json in self._PATTERN.findall(text):
            try:
                args = json.loads(args_json.strip())
                result.append(
                    json.dumps(
                        {"name": self._default_tool_name, "arguments": args},
                        ensure_ascii=False
                    )
                )
            except json.JSONDecodeError:
                continue
        return result

    def extract_answer(self, text: str) -> Optional[str]:
        answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return answers[-1].strip() if answers else None


# ================= 注册表与工厂函数 =================

PARSER_REGISTRY: dict = {
    "default":  DefaultParser,
    "deepseek": DeepSeekParser,
    "glm4":     GLM4Parser,
    "minimax":  MinimaxParser,
    "qwen35":   Qwen35Parser,
    "kimi_k2":  KimiK2Parser,
}


def get_parser(name_or_instance) -> ToolCallParser:
    """
    按名称返回 Parser 实例，或直接透传已有实例。

    Args:
        name_or_instance: Parser 名称字符串，或 ToolCallParser 实例。

    Returns:
        ToolCallParser 实例。

    Raises:
        ValueError: 名称未在注册表中找到。

    示例::
        parser = get_parser("default")
        parser = get_parser("glm4")
        parser = get_parser(KimiK2Parser(default_tool_name="web_search"))  # 指定兜底工具名
        parser = get_parser(MyCustomParser())   # 直接传实例
    """
    if isinstance(name_or_instance, ToolCallParser):
        return name_or_instance
    parser_cls = PARSER_REGISTRY.get(name_or_instance)
    if parser_cls is None:
        available = list(PARSER_REGISTRY.keys())
        raise ValueError(
            f"Unknown parser '{name_or_instance}'. "
            f"Available: {available}"
        )
    return parser_cls()
