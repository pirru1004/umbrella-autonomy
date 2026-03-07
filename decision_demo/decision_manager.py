import numpy as np


class DecisionManager:
    def __init__(self, w_science=1.0, w_hazard=1.2, w_energy=0.8, low_energy_threshold=0.2):
        self.w_science = w_science
        self.w_hazard = w_hazard
        self.w_energy = w_energy
        self.low_energy_threshold = low_energy_threshold

    def get_neighbors(self, x, y, size):
        neighbors = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                neighbors.append((nx, ny))
        return neighbors

    def decide(self, position, science_map, hazard_map, resources):
        x, y = position
        size = science_map.shape[0]

        # Guardrails
        if resources["comms_state"] == "LOST":
            return {
                "mode": "SAFE_STOP",
                "target": position,
                "reason": "Communications lost"
            }

        if resources["energy_state"] < self.low_energy_threshold:
            return {
                "mode": "RETURN",
                "target": (0, 0),
                "reason": "Low energy"
            }

        candidates = self.get_neighbors(x, y, size)

        best_score = -999
        best_target = position
        best_reason = "No better option"

        for nx, ny in candidates:
            science_score = science_map[nx, ny]
            hazard_score = hazard_map[nx, ny]

            utility = (
                self.w_science * science_score
                - self.w_hazard * hazard_score
                - self.w_energy * (1.0 - resources["energy_state"])
            )

            if utility > best_score:
                best_score = utility
                best_target = (nx, ny)
                best_reason = f"science={science_score:.2f}, hazard={hazard_score:.2f}, utility={utility:.2f}"

        return {
            "mode": "NORMAL",
            "target": best_target,
            "reason": best_reason
        }
