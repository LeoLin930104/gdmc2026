from __future__ import annotations

from prefab_housing.interior import derive_room_signature, plan_room
from prefab_housing.types import RoomRequest, RoomSpatialConstraints


def test_signature_classifies_room_size_and_exposure() -> None:
    request = RoomRequest(
        room_type="bedroom",
        utility_type="residential",
        role="habitable",
        constraints=RoomSpatialConstraints(
            voxel_size=(8, 6, 8),
            door_faces=("south",),
            window_faces=("north", "east"),
            privacy_depth=3,
            occupancy_capacity=2,
        ),
    )
    signature = derive_room_signature(request)
    assert signature.size_class == "standard"
    assert signature.exposure == "broad"
    assert signature.privacy_band == "deep_private"
    assert signature.lighting_tier == "central_plus_edges"


def test_room_plan_assigns_core_and_lighting_keywords() -> None:
    request = RoomRequest(
        room_type="bathroom",
        utility_type="residential",
        role="service",
        constraints=RoomSpatialConstraints(
            voxel_size=(8, 6, 8),
            door_faces=("west",),
            window_faces=(),
            privacy_depth=2,
            occupancy_capacity=1,
        ),
    )
    plan = plan_room(request)
    assert "toilet" in plan.core_keywords
    assert "sink" in plan.core_keywords
    assert "shower" in plan.core_keywords
    assert "ceiling_light" in plan.lighting_keywords
    assert "wall_light" in plan.lighting_keywords
