"""读取并校验合成经营资料。

校验规则与 tools/check_materials.py 一致：内容指纹、行数、主键唯一、
字段集合与类型、直接外键。此处提供可被应用其他模块复用的载入接口。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Materials:
    """一次载入并通过校验的全部资料。"""

    root: Path
    fingerprint: str
    tables: dict[str, list[dict]]

    def table(self, name: str) -> list[dict]:
        return self.tables[name]

    def counts(self) -> dict[str, int]:
        return {name: len(rows) for name, rows in self.tables.items()}


def _check(catalog: dict, raw: dict[str, bytes]) -> dict[str, list[dict]]:
    loaded: dict[str, list[dict]] = {}
    for name, spec in catalog["tables"].items():
        data = raw[name]
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError("资料指纹不符 " + name)
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != spec["row_count"]:
            raise ValueError("资料行数不符 " + name)
        identifiers = [r[spec["key"]] for r in rows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("资料主键重复 " + name)
        for row in rows:
            if set(row) != set(spec["fields"]):
                raise ValueError("字段不符 " + name)
            for field, definition in spec["fields"].items():
                value = row[field]
                if value is None and definition["nullable"]:
                    continue
                if type(value).__name__ != definition["type"]:
                    raise ValueError("字段类型不符 " + name + "." + field)
        loaded[name] = rows
    for name, spec in catalog["tables"].items():
        for field, (target, key) in spec["references"].items():
            keys = {r[key] for r in loaded[target]}
            for row in loaded[name]:
                if row[field] is not None and row[field] not in keys:
                    raise ValueError("直接引用不存在 " + name + "." + field)
    return loaded


def load(root: str | Path, *, validate: bool = True) -> Materials:
    """载入 root/data 下的全部资料；validate 为真时执行完整校验。"""
    root = Path(root)
    catalog_text = (root / "data/catalog.json").read_text(encoding="utf-8")
    catalog = json.loads(catalog_text)
    raw = {}
    for name in catalog["tables"]:
        raw[name] = (root / "data" / (name + ".jsonl")).read_bytes()
    if validate:
        tables = _check(catalog, raw)
    else:
        tables = {
            name: [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
            for name, data in raw.items()
        }
    fingerprint = hashlib.sha256(catalog_text.encode("utf-8")).hexdigest()
    return Materials(root=root, fingerprint=fingerprint, tables=tables)
