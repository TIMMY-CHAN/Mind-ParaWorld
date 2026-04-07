#!/usr/bin/env python3
"""
将 Markdown 主表转换为 Excel 文件。

用法：
    python analysis/utils/convert_to_excel.py input.md output.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = ['Pass@1 (%)', 'FCR', 'Hit Rate (%)', 'Avg Turns', 'Samples']
EMOJI_PREFIXES = ['🟢 ', '🟡 ', '🔴 ']


def extract_markdown_table(markdown_text: str) -> pd.DataFrame:
    lines = markdown_text.splitlines()

    header_line = None
    data_lines = []
    in_table = False

    for line in lines:
        if line.startswith('| Model'):
            header_line = line
            in_table = True
            continue
        if in_table and line.startswith('|') and not line.startswith('|----'):
            data_lines.append(line)

    if header_line is None:
        raise ValueError('未找到 Markdown 表格表头（以 `| Model` 开头）')

    headers = [h.strip() for h in header_line.split('|')[1:-1]]
    data = [[c.strip() for c in line.split('|')[1:-1]] for line in data_lines]

    if not data:
        raise ValueError('未找到 Markdown 表格数据行')

    return pd.DataFrame(data, columns=headers)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if 'Difficulty' in df.columns:
        for prefix in EMOJI_PREFIXES:
            df['Difficulty'] = df['Difficulty'].str.replace(prefix, '', regex=False)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def convert_markdown_to_excel(input_path: Path, output_path: Path) -> None:
    markdown_text = input_path.read_text(encoding='utf-8')
    df = extract_markdown_table(markdown_text)
    df = clean_dataframe(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name='Results by Difficulty')

    print(f'Excel file created successfully: {output_path}')
    print(f'Total rows: {len(df)}')


def main() -> None:
    parser = argparse.ArgumentParser(description='将 Markdown 主表转换为 Excel 文件')
    parser.add_argument('input', help='输入 Markdown 文件路径')
    parser.add_argument('output', help='输出 Excel 文件路径')
    args = parser.parse_args()

    convert_markdown_to_excel(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
