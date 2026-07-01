from __future__ import annotations

import hashlib
import random

TOWN_SQUARE = "town_square"
NON_CENTRAL_ROLES = ("farm", "residential", "barracks")
VALID_ROLES = (TOWN_SQUARE, *NON_CENTRAL_ROLES)


def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256((name or "").encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 32)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(p[0] for p in points)
    sz = sum(p[1] for p in points)
    return (sx / n, sz / n)


def assign_district_roles(
    zone_seed_points,
    settlement_name: str,
    centroid: tuple[float, float] | None = None,
) -> dict[int, str]:
    points = [(float(p[0]), float(p[1])) for p in zone_seed_points]
    n = len(points)
    if n == 0:
        return {}

    cx, cz = centroid if centroid is not None else _centroid(points)

    def dist_sq(i: int) -> float:
        px, pz = points[i]
        return (px - cx) ** 2 + (pz - cz) ** 2

    # min() returns the first index on a tie -> deterministic.
    central = min(range(n), key=dist_sq)
    roles: dict[int, str] = {central: TOWN_SQUARE}

    others = [i for i in range(n) if i != central]
    pool = list(NON_CENTRAL_ROLES)
    random.Random(_seed_from_name(settlement_name)).shuffle(pool)
    for k, i in enumerate(others):
        roles[i] = pool[k % len(pool)]

    # Safety: with fewer than 3 non-central zones the shuffle can omit "farm";
    # force it on, since the fields must live somewhere. (No-op in the standard
    # 4-zone case, where the 3 others always get one of each.)
    if others and "farm" not in roles.values():
        roles[others[0]] = "farm"

    return roles


def role_keeps_fields(role: str) -> bool:
    return role == "farm"


def farm_zone(roles: dict[int, str]) -> int | None:
    for zone_index, role in sorted(roles.items()):
        if role == "farm":
            return zone_index
    return None


if __name__ == "__main__":
    # Synthetic 4-zone layout: zone 2 is dead center -> should be town_square.
    pts = [(0.0, 0.0), (20.0, 0.0), (10.0, 10.0), (10.0, 30.0)]
    a = assign_district_roles(pts, "Karrowdeep")
    b = assign_district_roles(pts, "Karrowdeep")
    c = assign_district_roles(pts, "Thornwick")
    print("Karrowdeep:", a)
    print("repeat    :", b, "(stable)" if a == b else "(NON-DETERMINISTIC!)")
    print("Thornwick :", c)
    print("central is town_square:", a[2] == TOWN_SQUARE)
    print("farm present:", "farm" in a.values(), "-> farm zone:", farm_zone(a))
    # 3-zone and 2-zone edge cases still yield a farm district.
    print("3-zone:", assign_district_roles(pts[:3], "Karrowdeep"))
    print("2-zone:", assign_district_roles(pts[:2], "Karrowdeep"))
