import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _class_node(path: str, class_name: str) -> ast.ClassDef:
    tree = ast.parse(_source(path), filename=path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {path}")


def _init_node(class_node: ast.ClassDef) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    raise AssertionError(f"__init__ not found in {class_node.name}")


def _super_init_calls(init_node: ast.FunctionDef) -> list[ast.Call]:
    calls = []
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "__init__" or not isinstance(node.func.value, ast.Call):
            continue
        if isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == "super":
            calls.append(node)
    return calls


class TransformersCompatTests(unittest.TestCase):
    def test_processors_require_only_tokenizer_and_keep_codec_local(self):
        processors = (
            ("moss_tts_delay/processing_moss_tts.py", "MossTTSDelayProcessor"),
            ("moss_tts_local/processing_moss_tts.py", "MossTTSDelayProcessor"),
            ("moss_tts_local_v1.5/processing_moss_tts.py", "MossTTSLocalProcessor"),
        )

        for path, class_name in processors:
            with self.subTest(path=path):
                source = _source(path)
                self.assertNotIn("MODALITY_TO_BASE_CLASS_MAPPING", source)
                self.assertNotIn("AUTO_TO_BASE_CLASS_MAPPING", source)

                class_node = _class_node(path, class_name)
                attribute_assignments = [
                    node
                    for node in class_node.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "attributes" for target in node.targets)
                ]
                self.assertEqual(len(attribute_assignments), 1)
                value = attribute_assignments[0].value
                self.assertIsInstance(value, ast.List)
                self.assertEqual(
                    [item.value for item in value.elts if isinstance(item, ast.Constant)],
                    ["tokenizer"],
                )

                calls = _super_init_calls(_init_node(class_node))
                self.assertEqual(len(calls), 1)
                self.assertNotIn("audio_tokenizer", {kw.arg for kw in calls[0].keywords})

    def test_delay_configs_forward_pad_token_id_to_pretrained_config(self):
        configs = (
            ("moss_tts_delay/configuration_moss_tts.py", "MossTTSDelayConfig"),
            ("moss_tts_local/configuration_moss_tts.py", "MossTTSDelayConfig"),
            ("moss_tts_local_v1.5/configuration_moss_tts.py", "MossTTSLocalConfig"),
        )

        for path, class_name in configs:
            with self.subTest(path=path):
                class_node = _class_node(path, class_name)
                calls = _super_init_calls(_init_node(class_node))
                self.assertEqual(len(calls), 1)
                self.assertIn("pad_token_id", {kw.arg for kw in calls[0].keywords})


if __name__ == "__main__":
    unittest.main()
