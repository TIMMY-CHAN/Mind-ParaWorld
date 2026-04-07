# Data Format

框架接受 JSONL 格式数据，每行一个 JSON 对象。

```json
{
  "index": 0,
  "prompt": [{"role": "user", "content": "你的问题或任务"}],
  "answer": "参考答案（供 LLM-as-Judge 评分使用）",
  "extra_info": {}
}
```

## 字段说明

| 字段 | 类型 | 是否必填 | 说明 |
|---|---|---|---|
| `index` | int | 是 | 样本序号 |
| `prompt` | list | 是 | 对话消息列表，格式与 OpenAI messages 一致 |
| `answer` | str | 是 | 参考答案，Judge 评分时与模型输出对比 |
| `extra_info` | dict | 是 | 附加元数据，不需要时传 `{}` |

## 示例

`mpw_bench.jsonl` 包含三条示例，可直接作为格式参考，也可用于框架冒烟测试：

```bash
# 用示例数据快速验证框架配置
head -n 3 data/mpw_bench.jsonl > data/smoke_test.jsonl
```
