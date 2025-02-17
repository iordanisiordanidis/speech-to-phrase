"""Tests for Home Assistant API."""

from speech_to_phrase.hass_api import Entity, Things


def test_template_syntax_removed() -> None:
    """Test that entity names have template syntax removed."""
    things = Things(
        entities=[
            Entity(names=["<test> {entity} [with] (template) syntax>]("], domain="test")
        ],
        areas=[],
        floors=[],
    )
    assert things.to_lists_dict() == {
        "area": {"values": []},
        "floor": {"values": []},
        "name": {
            "values": [
                {
                    "in": "test entity with template syntax",
                    "out": "<test> {entity} [with] (template) syntax>](",
                    "context": {"domain": "test"},
                }
            ]
        },
    }
