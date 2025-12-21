import pathlib
import unittest


class ReadmeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.readme_path = self.project_root / "README.md"

    def test_readme_exists(self):
        self.assertTrue(
            self.readme_path.exists(), "Das Projekt sollte eine README.md enthalten."
        )

    def test_readme_contains_project_title(self):
        content = self.readme_path.read_text(encoding="utf-8")
        self.assertIn("Uni-Projekt-2025", content)


if __name__ == "__main__":
    unittest.main()
