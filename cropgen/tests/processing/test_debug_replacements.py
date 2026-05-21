from cropgen.processing.AnnotatedPage import AnnotatedPage


def test_debug_replacements(text: str):
    print(AnnotatedPage._correct_text(text, 0))
