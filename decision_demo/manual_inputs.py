def get_manual_inputs():
    science_candidates = [
        {"id": 1, "x": 14, "y": 16, "science_score": 0.91, "confidence": 0.82, "mineral_type": "clay"},
        {"id": 2, "x": 6, "y": 9,  "science_score": 0.78, "confidence": 0.90, "mineral_type": "basalt"},
        {"id": 3, "x": 17, "y": 4, "science_score": 0.69, "confidence": 0.75, "mineral_type": "sulfate"}
    ]

    resource_state = {
        "energy_state": 0.42,
        "comms_state": "OK",       # OK / DEGRADED / LOST
        "compute_margin": 0.78,
        "risk_level": 0.33,
        "wind_state": 0.20
    }

    hazard_overrides = {
        1: {"hazard_score": 0.65, "distance_cost": 0.50},
        2: {"hazard_score": 0.25, "distance_cost": 0.20},
        3: {"hazard_score": 0.40, "distance_cost": 0.60}
    }

    rover_position = (2, 2)

    return science_candidates, resource_state, hazard_overrides, rover_position
