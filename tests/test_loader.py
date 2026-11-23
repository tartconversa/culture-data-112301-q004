"""tourflow.materials 载入器：校验规则与损坏检测。"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tourflow.materials import load

ROOT = Path(__file__).resolve().parents[1]


class LoaderTests(unittest.TestCase):
    def test_load_counts(self):
        materials = load(ROOT)
        counts = materials.counts()
        self.assertEqual(counts["daily_panel"], 8640)
        self.assertEqual(counts["group_requests"], 24)
        self.assertTrue(materials.fingerprint)

    def test_corruption_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "data", target / "data")
            with (target / "data" / "sites.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"id":"S9","daily_capacity":1,"protected_limit":1,"region":"R9"}\n')
            with self.assertRaisesRegex(ValueError, "资料指纹不符"):
                load(target)

    def test_type_violation_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "data", target / "data")
            # 改动内容后同步伪造指纹，验证类型校验仍在指纹之后兜底
            rows = [json.loads(line) for line in (target / "data" / "sites.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["daily_capacity"] = "八十"
            payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
            (target / "data" / "sites.jsonl").write_text(payload, encoding="utf-8")
            import hashlib
            catalog = json.loads((target / "data" / "catalog.json").read_text(encoding="utf-8"))
            catalog["tables"]["sites"]["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            (target / "data" / "catalog.json").write_text(
                json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "字段类型不符"):
                load(target)


if __name__ == "__main__":
    unittest.main()
