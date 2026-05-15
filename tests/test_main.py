from src.main import hello


def test_hello_default() -> None:
    assert hello() == "hello, world!"


def test_hello_custom() -> None:
    assert hello("ml") == "hello, ml!"
