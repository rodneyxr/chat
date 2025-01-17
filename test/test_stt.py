import pytest
from src.sttpy import process_with_ollama


def test_basic_commands():
    """Test basic commands and their translations"""
    test_cases = [
        ("hello", "hello"),
        ("how are you", "How are you?"),
        ("space", " "),
        ("open the door", "Open the door."),
        ("close the window", "Close the window."),
    ]

    for input_text, expected in test_cases:
        result = process_with_ollama(input_text)
        print(result)
        assert expected in result


def test_programming_commands():
    """Test various programming commands and their translations"""
    test_cases = [
        ("print hello world", 'print("hello world")'),
        ("python for loop from zero to five", "for i in range(0, 5):"),
        ("define function calculate sum", "def calculate_sum"),
    ]

    for input_text, expected in test_cases:
        result = process_with_ollama(input_text)
        assert expected.lower() in result.lower()


def test_natural_language_programming():
    """Test natural language for programming"""
    result = process_with_ollama("create a variable counter and set it to 5")
    # show the output in pytest output
    print(result)
    assert "counter = 5" in result


def test_natural_language_text():
    """Test natural language improvements"""
    result = process_with_ollama("hello how are you question mark.")
    # show the output in pytest output
    print(result)
    assert "how are you?" in result.lower()


def test_error_correction():
    """Test the error correction capabilities"""
    result = process_with_ollama("prnt hello world")  # Common typo
    assert "print" in result.lower()


def test_special_characters():
    """Test handling of special programming characters"""
    test_cases = [
        ("open curly brace", "{"),
        ("close parenthesis", ")"),
        ("period", "."),
        ("question mark", "?"),
        ("dash", "-"),
        ("underscore", "_"),
    ]

    for input_text, expected_char in test_cases:
        result = process_with_ollama(input_text)
        assert expected_char in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
