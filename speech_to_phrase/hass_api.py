"""Home Assistant API."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import aiohttp

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Home Assistant entity."""

    names: List[str]
    domain: str
    _hash: str = ""

    def get_hash(self) -> str:
        """Get a stable hash for this entity."""
        if not self._hash:
            hasher = hashlib.sha256()

            hasher.update(self.domain.encode("utf-8"))
            for name in sorted(self.names):
                hasher.update(name.encode("utf-8"))

            self._hash = hasher.hexdigest()

        return self._hash


@dataclass
class Area:
    """Home Assistant area."""

    names: List[str]
    _hash: str = ""

    def get_hash(self) -> str:
        """Get a stable hash for this area."""
        if not self._hash:
            hasher = hashlib.sha256()

            for name in sorted(self.names):
                hasher.update(name.encode("utf-8"))

            self._hash = hasher.hexdigest()

        return self._hash


@dataclass
class Floor:
    """Home Assistant floor."""

    names: List[str]
    _hash: str = ""

    def get_hash(self) -> str:
        """Get a stable hash for this floor."""
        if not self._hash:
            hasher = hashlib.sha256()

            for name in sorted(self.names):
                hasher.update(name.encode("utf-8"))

            self._hash = hasher.hexdigest()

        return self._hash


@dataclass
class Things:
    """Exposed things in Home Assistant."""

    entities: List[Entity] = field(default_factory=list)
    areas: List[Area] = field(default_factory=list)
    floors: List[Floor] = field(default_factory=list)
    trigger_sentences: List[str] = field(default_factory=list)
    _hash: str = ""

    def get_hash(self) -> str:
        """Get a stable hash for all the things."""
        if not self._hash:
            hasher = hashlib.sha256()

            for entity_hash in sorted(e.get_hash() for e in self.entities):
                hasher.update(entity_hash.encode("utf-8"))

            for area_hash in sorted(e.get_hash() for e in self.entities):
                hasher.update(area_hash.encode("utf-8"))

            for floor_hash in sorted(e.get_hash() for e in self.entities):
                hasher.update(floor_hash.encode("utf-8"))

            for trigger_sentence in sorted(self.trigger_sentences):
                hasher.update(trigger_sentence.encode("utf-8"))

            self._hash = hasher.hexdigest()

        return self._hash

    def to_lists_dict(self) -> Dict[str, Any]:
        """Get lists dictionary for hassil intents."""
        lists_dict: Dict[str, Any] = {}
        lists_dict["name"] = {
            "values": [
                {
                    "in": _remove_template_syntax(e_name),
                    "out": e_name,
                    "context": {"domain": e.domain},
                }
                for e in self.entities
                for e_name in e.names
            ]
        }

        lists_dict["area"] = {
            "values": [a_name for a in self.areas for a_name in a.names]
        }

        lists_dict["floor"] = {
            "values": [f_name for f in self.floors for f_name in f.names]
        }

        return lists_dict


def _remove_template_syntax(name: str) -> str:
    """Remove template syntax from a name."""
    return re.sub(r"[{}\[\]<>()]", "", name)


@dataclass
class HomeAssistantInfo:
    """Information loaded from Home Assistant websocket API."""

    system_language: str
    things: Things
    pipeline_languages: Set[str] = field(default_factory=set)


async def get_hass_info(token: str, uri: str) -> HomeAssistantInfo:
    """Use HA websocket API to get exposed entities/areas/floors."""
    things = Things()
    pipeline_languages: Set[str] = set()

    current_id = 0

    def next_id() -> int:
        nonlocal current_id
        current_id += 1
        return current_id

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(uri, max_msg_size=0) as websocket:
            # Authenticate
            msg = await websocket.receive_json()
            assert msg["type"] == "auth_required", msg

            await websocket.send_json(
                {
                    "type": "auth",
                    "access_token": token,
                }
            )

            msg = await websocket.receive_json()
            assert msg["type"] == "auth_ok", msg

            # Get system language
            await websocket.send_json({"id": next_id(), "type": "get_config"})

            msg = await websocket.receive_json()
            assert msg["success"], msg

            system_language: str = msg["result"]["language"]

            # Get pipeline STT languages
            await websocket.send_json(
                {"id": next_id(), "type": "assist_pipeline/pipeline/list"}
            )
            msg = await websocket.receive_json()
            assert msg["success"], msg

            for pipeline in msg["result"]["pipelines"]:
                stt_language = pipeline.get("stt_language")
                if stt_language:
                    pipeline_languages.add(stt_language)

            # Get exposed entities
            await websocket.send_json(
                {"id": next_id(), "type": "homeassistant/expose_entity/list"}
            )

            msg = await websocket.receive_json()
            assert msg["success"], msg

            entity_ids = []
            for entity_id, exposed_info in msg["result"]["exposed_entities"].items():
                if exposed_info.get("conversation"):
                    entity_ids.append(entity_id)

            await websocket.send_json(
                {
                    "id": next_id(),
                    "type": "get_states",
                }
            )
            msg = await websocket.receive_json()
            assert msg["success"], msg
            states = {s["entity_id"]: s for s in msg["result"]}

            # Get device info
            await websocket.send_json(
                {"id": next_id(), "type": "config/device_registry/list"}
            )
            msg = await websocket.receive_json()
            assert msg["success"], msg
            # devices = {device_info["id"]: device_info for device_info in msg["result"]}

            # Floors
            await websocket.send_json(
                {"id": next_id(), "type": "config/floor_registry/list"}
            )
            msg = await websocket.receive_json()
            assert msg["success"], msg
            floors = {
                floor_info["floor_id"]: floor_info for floor_info in msg["result"]
            }
            for floor_info in floors.values():
                names = [floor_info["name"]]
                names.extend(floor_info.get("aliases", []))
                things.floors.append(Floor(names=[name.strip() for name in names]))

            # Areas
            await websocket.send_json(
                {"id": next_id(), "type": "config/area_registry/list"}
            )
            msg = await websocket.receive_json()
            assert msg["success"], msg
            areas = {area_info["area_id"]: area_info for area_info in msg["result"]}
            for area_info in areas.values():
                names = [area_info["name"]]
                names.extend(area_info.get("aliases", []))
                things.areas.append(Area(names=[name.strip() for name in names]))

            # Contains aliases
            # Check area_id as well as area of device_id
            # Use original_device_class
            await websocket.send_json(
                {
                    "id": next_id(),
                    "type": "config/entity_registry/get_entries",
                    "entity_ids": entity_ids,
                }
            )

            msg = await websocket.receive_json()
            assert msg["success"], msg
            for entity_id, entity_info in msg["result"].items():
                domain = entity_id.split(".")[0]
                name = None
                names = []

                if entity_info:
                    if entity_info.get("disabled_by") is not None:
                        # Skip disabled entities
                        continue

                    name = entity_info.get("name") or entity_info["original_name"]
                    names.extend(entity_info.get("aliases", []))

                if (not name) and (entity_id in states):
                    name = states[entity_id]["attributes"].get("friendly_name")

                if name:
                    names.append(name)

                things.entities.append(
                    Entity(names=[name.strip() for name in names], domain=domain)
                )

            # Get sentences from sentence triggers
            await websocket.send_json(
                {"id": next_id(), "type": "conversation/sentences/list"}
            )
            msg = await websocket.receive_json()
            if msg["success"]:
                things.trigger_sentences.extend(set(msg["result"]["trigger_sentences"]))

    return HomeAssistantInfo(
        system_language=system_language,
        things=things,
        pipeline_languages=pipeline_languages,
    )
