from src.utils.dialogue_parser import clear_annotations


def test_remove_annotations():
    print(clear_annotations("Hello (World) dawg"))
