# Data Format

The framework accepts JSONL format data, one JSON object per line.

```json
{
  "index": 0,
  "prompt": [{"role": "user", "content": "Your question or task"}],
  "answer": "Reference answer (for LLM-as-Judge scoring)",
  "extra_info": {}
}
```

## Field Descriptions

| Field | Type | Required | Description |
|---|---|---|---|
| `index` | int | Yes | Sample number |
| `prompt` | list | Yes | Dialogue message list, format consistent with OpenAI messages |
| `answer` | str | Yes | Reference answer, compared with model output during Judge scoring |
| `extra_info` | dict | Yes | Additional metadata, pass `{}` if not needed |

## Example

`mpw_bench.jsonl` contains three examples that can be used as format reference or for framework smoke testing:

```bash
# Quickly verify framework configuration with example data
head -n 3 data/mpw_bench.jsonl > data/smoke_test.jsonl
```
