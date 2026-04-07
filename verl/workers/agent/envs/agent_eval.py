# agent_eval.py
# 评测专用版本：集成 WorldModelWebSearchTool，支持命中日志收集

from typing import Optional, List, Dict, Any, Union
import re
import json
from copy import deepcopy
from verl.workers.agent.tool_envs import ToolBase
from verl.workers.agent.envs.tools.world_model_web_search_tool import WorldModelWebSearchTool



class AgentEval(ToolBase):
    """评测专用多轮工具调用 Agent，支持轨迹收集和命中日志记录"""

    name = "agent_eval"
    user_prompt = """
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

    def __init__(self, _name=None, _desc=None, _params=None, save_full_history: bool = True,
                 tools: Optional[List] = None,
                 parser: Union[str, 'ToolCallParser'] = "default",
                 **kwargs):
        """
        初始化评测专用多轮工具调用 Agent

        Args:
            _name: Agent 名称（兼容旧接口）
            _desc: Agent 描述（兼容旧接口）
            _params: Agent 参数（兼容旧接口）
            save_full_history: 是否保存完整对话历史（大规模评测时建议设为 False 以节省内存）
            tools: 可插拔工具列表（每个工具需有 .name 属性）。
                   为 None 时默认注册 web_search（WorldModelWebSearchTool）。
                   示例：tools=[MyCustomTool(), WorldModelWebSearchTool()]
            parser: tool call 解析器，支持名称字符串或 ToolCallParser 实例。
                    内置选项："default"（默认）、"deepseek"、"glm4"。
                    自定义：继承 ToolCallParser 后直接传入实例。
            **kwargs: 其他参数
        """
        super().__init__(name=self.name)
        self.chatml_history: str = ""  # 【修复】明确类型为字符串
        self.multi_modal_data = None
        self.crop_image = None
        self.world_truth: Dict[str, Any] = {}  # 存储世界真相
        self.save_full_history = save_full_history  # 【新增】控制是否保存完整历史
        self._warned: set = set()  # 已打印过的 WARN 类型，每类只打印一次

        # 【评测专用】轨迹收集器
        self.trajectory_log: Dict[str, Any] = {
            "tool_calls": [],  # 每次工具调用的详细信息
            "hit_logs": [],    # 每次搜索的命中日志
            "full_history": []  # 完整的对话历史（可选）
        }

        # 子工具注册表（支持可插拔工具注册）
        # tools 参数：传入工具实例列表即可替换默认工具集
        if tools is not None:
            self.sub_tool_registry = {tool.name: tool for tool in tools}
        else:
            self.sub_tool_registry = {
                'web_search': WorldModelWebSearchTool('web_search', 'Tool for web search', {}),
            }

        # tool call 解析器（支持按模型格式插拔）
        from verl.workers.agent.parsers import get_parser
        self.parser = get_parser(parser)

    def extract_answer(self, action_string: str) -> Optional[str]:
        return self.parser.extract_answer(action_string)

    def extract_action(self, action_string: str) -> Optional[str]:
        return self.parser.extract_action(action_string)

    def register_tool(self, tool) -> None:
        """
        注册一个工具到 sub_tool_registry。

        工具需要有 .name 属性，以及 execute() 或 execute_async() 方法。
        如果同名工具已存在，会打印警告并覆盖。

        Args:
            tool: 工具实例，需有 .name 属性

        示例::
            agent = AgentEval()
            agent.register_tool(MyCustomTool())
        """
        if not hasattr(tool, 'name'):
            raise AttributeError(f"Tool {tool} must have a 'name' attribute.")
        if tool.name in self.sub_tool_registry:
            print(f"[WARNING] Tool '{tool.name}' already registered, overwriting.")
        self.sub_tool_registry[tool.name] = tool

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        同步执行入口（封装 execute_async）。

        推荐在 async 上下文中直接调用 execute_async。
        同步场景可使用此方法，但需注意不能在已有 event loop 的上下文中调用。
        """
        import asyncio
        return asyncio.run(self.execute_async(action_string, **kwargs))

    def _save_agent_parse_failure_log(self, full_action: str, extracted_json: str, error: Exception):
        """
        保存 Agent 生成的错误 tool_call JSON 到日志文件

        Args:
            full_action: Agent 完整输出（包含 <tool_call> 标签）
            extracted_json: 提取的 JSON 字符串
            error: JSON 解析错误
        """
        import os
        from datetime import datetime

        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "agent_parse_errors")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(log_dir, f"agent_parse_error_{timestamp}.log")

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("Agent Tool Call JSON 解析失败日志\n")
                f.write("="*80 + "\n\n")

                f.write(f"时间: {datetime.now().isoformat()}\n")
                f.write(f"错误类型: {type(error).__name__}\n")
                f.write(f"错误信息: {str(error)}\n\n")

                f.write("-"*80 + "\n")
                f.write("Agent 完整输出:\n")
                f.write("-"*80 + "\n")
                f.write(full_action)
                f.write("\n\n")

                f.write("-"*80 + "\n")
                f.write("提取的 tool_call JSON:\n")
                f.write("-"*80 + "\n")
                f.write(extracted_json)
                f.write("\n\n")

                f.write("="*80 + "\n")

            print(f"[INFO Eval] Agent 解析失败日志已保存: {log_file}")
        except Exception as e:
            print(f"[WARN Eval] 保存 Agent 解析失败日志时出错: {e}")

    async def execute_async(self, action_string: str, **kwargs) -> tuple:
        """
        异步执行工具调用，并收集轨迹

        Returns:
            obs (str): 观察文本
            reward (float): 奖励（评测模式固定为0）
            done (bool): 是否结束
            info (dict): 包含命中日志等信息
        """
        import asyncio

        # 1. 检查是否是最终答案
        answer = self.extract_answer(action_string)
        if answer:
            self.trajectory_log["final_answer"] = answer
            return "", 0.0, True, {"final_answer": answer}

        # 2. 提取工具调用
        action = self.extract_action(action_string)
        if not action:
            # 【修复】没有工具调用，应该结束
            return "", 0.0, True, {"error": "No valid tool call found"}

        try:
            tool_call = json.loads(action.strip())
        except json.JSONDecodeError as e:
            # 【增强】尝试修复常见的 JSON 格式问题
            original = action.strip()

            # 辅助函数：提取第一个完整的 JSON 对象（处理多余的括号）
            def extract_first_json(s):
                depth = 0
                for i, char in enumerate(s):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            return s[:i+1]
                return s

            fixed_attempts = [
                original,  # 原始
                re.sub(r',(\s*)}', r'\1}', original),  # 移除对象尾随逗号（支持换行）
                re.sub(r',(\s*)]', r'\1]', original),  # 移除数组尾随逗号（支持换行）
                re.sub(r',(\s*)}', r'\1}', re.sub(r',(\s*)]', r'\1]', original)),  # 同时移除
                extract_first_json(original),  # 提取第一个完整 JSON（处理多余右括号）
            ]

            last_error = e
            for attempt in fixed_attempts[1:]:  # 跳过第一个（已经尝试过）
                try:
                    tool_call = json.loads(attempt)
                    print(f"[INFO Eval] JSON 修复成功！")
                    break  # 成功，继续执行
                except json.JSONDecodeError as retry_error:
                    last_error = retry_error
                    continue
            else:
                # 所有尝试都失败
                error_msg = f"Invalid tool call format: {action.strip()}. Error: {last_error}"
                obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
                print(f"[ERROR Eval] JSON parse failed after all attempts: {last_error}")

                # 【新增】保存 Agent 生成的错误 tool_call JSON 到日志
                self._save_agent_parse_failure_log(action_string, action.strip(), last_error)

                return obs, 0.0, True, {"error": str(last_error), "error_type": "parse_error"}

        try:
            tool_name = tool_call["name"]

            if tool_name in self.sub_tool_registry:
                sub_tool = self.sub_tool_registry[tool_name]

                # 3. 准备通用参数（各工具通过 **kwargs 按需取用，未用到的参数会被忽略）
                execute_kwargs = {
                    'agent_trajectory': self.chatml_history + action_string,
                    'world_truth': self.world_truth,
                }

                # 4. 异步调用
                if hasattr(sub_tool, 'execute_async') and callable(sub_tool.execute_async):
                    obs, reward, done, info = await sub_tool.execute_async(action_string=action_string, **execute_kwargs)
                else:
                    def _sync_wrapper():
                        return sub_tool.execute(action_string=action_string, **execute_kwargs)
                    obs, reward, done, info = await asyncio.to_thread(_sync_wrapper)

                # 5. 【关键】收集轨迹日志
                if "hit_log" in info:
                    self.trajectory_log["hit_logs"].append(info["hit_log"])

                # 记录工具调用
                self.trajectory_log["tool_calls"].append({
                    "tool_name": tool_name,
                    "query": tool_call.get("arguments", {}).get("query", ""),
                    "hit": info.get("hit_log", {}).get("hit", 0),
                })

                # 6. 更新历史
                if not done:
                    self.chatml_history += action_string
                    self.chatml_history += obs

                # 7. 记录完整历史（【修复】支持可选保存）
                if self.save_full_history:
                    self.trajectory_log["full_history"].append({
                        "role": "assistant",
                        "content": action_string
                    })
                    self.trajectory_log["full_history"].append({
                        "role": "user",
                        "content": obs
                    })

            else:
                raise ValueError(f"Unknown tool name: {tool_name}")

            reward = 0.0
            done = False
            info = {**info, "status": "success", "tool_used": tool_name}
            print(f'[DEBUG Eval] SUCCESS ACTION: {tool_name} - Query: {tool_call.get("arguments", {}).get("query", "")[:50]}...')
            return obs, reward, done, info

        except Exception as e:
            # 【修复】区分错误类型
            error_type = type(e).__name__
            print(f'[ERROR Eval] Execute WRONG - {error_type}: {str(e)} | action: {action_string[:100]}...')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"

            # 工具执行错误（如网络超时），可能可以重试，返回 done=False
            # 但如果是严重错误（如 KeyError），应该结束
            is_recoverable = error_type in ["TimeoutError", "ConnectionError", "HTTPError"]
            done_flag = not is_recoverable

            return obs, 0.0, done_flag, {"error": str(e), "status": "failed", "error_type": error_type}

    async def execute_parallel_async(self, action_string: str, **kwargs) -> tuple:
        """
        并行执行多个工具调用，结果合并为单条观察消息。

        若只提取到 ≤1 个工具调用，退化为 execute_async（保持原有行为）。

        Returns:
            obs (str): 合并后的观察文本（ChatML 格式）
            reward (float): 奖励（固定为0）
            done (bool): 任一工具调用触发结束则为 True
            info (dict): 包含并行调用数量等信息
        """
        import asyncio

        # 检查是否是最终答案
        answer = self.extract_answer(action_string)
        if answer:
            self.trajectory_log["final_answer"] = answer
            return "", 0.0, True, {"final_answer": answer}

        # 提取所有工具调用
        actions = self.parser.extract_all_actions(action_string)

        # ≤1 个工具调用时退化为普通调用
        if len(actions) <= 1:
            return await self.execute_async(action_string, **kwargs)

        async def _run_one(action_json: str):
            """执行单个工具调用，返回 (obs, done, info, tool_call_dict)。"""
            try:
                tool_call = json.loads(action_json.strip())
            except json.JSONDecodeError:
                return None, False, {"error": "json_parse_error"}, {}

            tool_name = tool_call.get("name", "")
            sub_tool = self.sub_tool_registry.get(tool_name)
            if sub_tool is None:
                return None, False, {"error": f"Unknown tool: {tool_name}"}, tool_call

            execute_kwargs = {
                "agent_trajectory": self.chatml_history + action_string,
                "world_truth": self.world_truth,
            }
            # 构造单个 tool_call 的 action_string
            single_action_str = f"<tool_call>\n{action_json}\n</tool_call>"

            if hasattr(sub_tool, "execute_async") and callable(sub_tool.execute_async):
                obs, _, done, info = await sub_tool.execute_async(
                    action_string=single_action_str, **execute_kwargs
                )
            else:
                def _sync():
                    return sub_tool.execute(action_string=single_action_str, **execute_kwargs)
                obs, _, done, info = await asyncio.to_thread(_sync)

            return obs, done, info, tool_call

        raw_results = await asyncio.gather(*[_run_one(a) for a in actions])

        obs_contents = []
        any_done = False
        for obs, done, info, tool_call in raw_results:
            if obs is None:
                obs_contents.append(f"Error: {info.get('error', 'unknown')}")
                continue

            # 收集轨迹日志
            if "hit_log" in info:
                self.trajectory_log["hit_logs"].append(info["hit_log"])
            self.trajectory_log["tool_calls"].append({
                "tool_name": tool_call.get("name", "unknown"),
                "query": tool_call.get("arguments", {}).get("query", ""),
                "hit": info.get("hit_log", {}).get("hit", 0),
            })

            # 从 ChatML 格式中提取纯内容
            content_match = re.search(
                r'<\|im_start\|>user\n(.*?)<\|im_end\|>', obs, re.DOTALL
            )
            obs_contents.append(content_match.group(1) if content_match else obs.strip())

            if done:
                any_done = True

        # 合并为单条 ChatML 消息
        combined_content = "\n\n".join(obs_contents)
        combined_obs = f"\n<|im_start|>user\n{combined_content}<|im_end|>\n<|im_start|>assistant\n"

        if not any_done:
            self.chatml_history += action_string
            self.chatml_history += combined_obs

        if self.save_full_history:
            self.trajectory_log["full_history"].append(
                {"role": "assistant", "content": action_string}
            )
            self.trajectory_log["full_history"].append(
                {"role": "user", "content": combined_obs}
            )

        return combined_obs, 0.0, any_done, {
            "status": "success",
            "parallel_calls": len(actions),
        }

    def reset(self, raw_prompt=None, multi_modal_data=None, origin_multi_modal_data=None, **kwargs):
        """
        重置 Agent 状态，准备新一轮评测

        Args:
            raw_prompt: 初始提示词（字符串）
            multi_modal_data: 处理后的多模态数据
            origin_multi_modal_data: 原始多模态数据
            **kwargs: 必须包含 'world_truth' 字段（Dict，包含 generated_question, atomic_facts, final_answer）

        Raises:
            Warning: 如果 world_truth 缺失或格式不正确
        """
        self.world_truth = kwargs.get("world_truth", {})

        # 每类警告只打印一次，避免大批量评测时刷屏
        if not self.world_truth:
            if "empty_world_truth" not in self._warned:
                print("[WARN Eval] world_truth is empty! Evaluation may not work correctly.")
                self._warned.add("empty_world_truth")
        elif 'atomic_facts' not in self.world_truth:
            if "missing_atomic_facts" not in self._warned:
                print("[WARN Eval] world_truth missing 'atomic_facts' field! FCR calculation will fail.")
                self._warned.add("missing_atomic_facts")
        elif not self.world_truth.get('atomic_facts'):
            if "empty_atomic_facts" not in self._warned:
                print("[WARN Eval] atomic_facts is empty! No facts to match.")
                self._warned.add("empty_atomic_facts")

        # 【修复】确保 chatml_history 是字符串类型
        # 2. 初始化历史
        if raw_prompt is None:
            self.chatml_history = ""
        elif isinstance(raw_prompt, str):
            self.chatml_history = raw_prompt
        else:
            # 如果是 List，通常在 async_inference 中由外部维护 messages，这里保持为空即可
            # 或者根据需求将其转换为 string，但通常不需要
            self.chatml_history = "" 

        self.multi_modal_data = origin_multi_modal_data

        # 重置轨迹日志
        self.trajectory_log = {
            "tool_calls": [],
            "hit_logs": [],
            "full_history": [] if self.save_full_history else None,  # 根据配置决定是否初始化
            "world_truth_info": self.world_truth  # 记录本题的世界真相
        }

        # 是否是多模态任务
        is_multimodal_task = bool(self.multi_modal_data)

        if is_multimodal_task:
            assert 'image' in self.multi_modal_data.keys(), f'[ERROR] Multi-modal data provided but "image" key is missing.'
            assert len(self.multi_modal_data['image']) > 0, f'[ERROR] "image" list is empty.'
            self.height = self.multi_modal_data['image'][0].height
            self.width = self.multi_modal_data['image'][0].width
        else:
            self.height = None
            self.width = None

        # 重置子工具（通用化：统一传入 world_truth，工具按需取用；user_prompt 按需设置）
        for tool_name, tool in self.sub_tool_registry.items():
            if hasattr(tool, 'user_prompt'):
                tool.user_prompt = self.user_prompt
            tool.reset(world_truth=self.world_truth)

    def get_trajectory_log(self) -> Dict[str, Any]:
        """获取完整的轨迹日志（用于外部统计）"""
        return self.trajectory_log

    def reset_crop_image(self, multi_modal_data):
        self.crop_image = multi_modal_data


# ================= 测试代码 =================

async def test_agent_eval():
    import asyncio
    print("\n" + "="*70)
    print("  测试评测版 AgentEval")
    print("="*70 + "\n")

    agent = AgentEval()

    # 模拟数据
    world_truth_info = {
        "generated_question": "2026年世界杯冠军是哪个国家？",
        "atomic_facts": {"2026年世界杯冠军": "中国队", "决赛比分": "3:0战胜巴西"},
        "final_answer": "中国队获得了2026年世界杯冠军。"
    }

    agent.reset(raw_prompt="", world_truth=world_truth_info)

    # 模拟两轮交互
    print(">>> 第1轮：低质量查询")
    action1 = '<tool_call>{"name": "web_search", "arguments": {"query": "2026年世界杯冠军是哪个国家？"}}</tool_call>'
    obs1, r1, d1, info1 = await agent.execute_async(action1)
    print(f"Hit: {info1.get('hit_log', {}).get('hit', 0)}")
    print(f"观察（前100字符）: {obs1[:100]}...")
    print()

    print(">>> 第2轮：高质量查询")
    action2 = '<tool_call>{"name": "web_search", "arguments": {"query": "2026年世界杯冠军"}}</tool_call>'
    obs2, r2, d2, info2 = await agent.execute_async(action2)
    print(f"Hit: {info2.get('hit_log', {}).get('hit', 0)}")
    print(f"观察（前100字符）: {obs2[:100]}...")
    print()

    # 获取轨迹日志
    traj_log = agent.get_trajectory_log()
    print(">>> 轨迹统计")
    print(f"工具调用次数: {len(traj_log['tool_calls'])}")
    print(f"命中次数: {sum([log['hit'] for log in traj_log['hit_logs']])}")
    print(f"命中的事实键: {[log['matched_fact_keys'] for log in traj_log['hit_logs']]}")

    print("\n" + "="*70)
    print("  测试完成")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_agent_eval())
