import re
from typing import Iterable, List


def clear_text(input: str) -> str:
    result = re.sub(r'@[\S]*', '', input.lower()) # убрать упоминания
    result = re.sub(r'https?:\/\/[\S]*', '', result) # убрать ссылки
    result = re.sub(r"[^a-z0-9 ']+", " ", result)  # оставить только буквы и цифры и апострофы
    result = re.sub(r"\s+", " ", result)  # убрать дублирующиеся пробелы
    return result.strip()

def store_data(data: Iterable[str], filename: str):
    with open(filename, 'w') as file:
        file.write('\n'.join(data))

def load_data(filename: str) -> List:
    with open(filename, 'r') as file:
        content = [line.strip() for line in file]
    return content