import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _module_tree(relative_path: str) -> ast.Module:
    return ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name} not found")


def _assigned_tuple(class_node: ast.ClassDef, name: str) -> tuple[str, ...]:
    for node in class_node.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and isinstance(node.value, ast.Tuple)
        ):
            return tuple(
                item.value
                for item in node.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    raise AssertionError(f"{name} tuple not found on {class_node.name}")


class SmartSeedOutputTests(unittest.TestCase):
    def test_easy_cade_exposes_selected_seed_without_reordering_existing_outputs(self):
        tree = _module_tree("mod/easy/mg_cade25_easy.py")
        cade_easy = _class_node(tree, "CADEEasyUI")

        self.assertEqual(
            _assigned_tuple(cade_easy, "RETURN_NAMES"),
            ("LATENT", "IMAGE", "mask_preview", "selected_seed"),
        )
        self.assertEqual(
            _assigned_tuple(cade_easy, "RETURN_TYPES"),
            ("LATENT", "IMAGE", "IMAGE", "INT"),
        )

    def test_base_cade_returns_selected_seed_for_advanced_wiring(self):
        tree = _module_tree("mod/easy/mg_cade25_easy.py")
        base_cade = _class_node(tree, "ComfyAdaptiveDetailEnhancer25")

        self.assertEqual(
            _assigned_tuple(base_cade, "RETURN_NAMES"),
            ("LATENT", "IMAGE", "steps", "cfg", "denoise", "mask_preview", "selected_seed"),
        )
        self.assertEqual(
            _assigned_tuple(base_cade, "RETURN_TYPES"),
            ("LATENT", "IMAGE", "INT", "FLOAT", "FLOAT", "IMAGE", "INT"),
        )

    def test_supersimple_exposes_final_step_selected_seed(self):
        tree = _module_tree("mod/easy/mg_supersimple_easy.py")
        super_simple = _class_node(tree, "MG_SuperSimple")

        self.assertEqual(
            _assigned_tuple(super_simple, "RETURN_NAMES"),
            ("LATENT", "IMAGE", "selected_seed"),
        )
        self.assertEqual(
            _assigned_tuple(super_simple, "RETURN_TYPES"),
            ("LATENT", "IMAGE", "INT"),
        )


if __name__ == "__main__":
    unittest.main()
