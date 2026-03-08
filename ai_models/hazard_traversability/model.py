class HazardTraversabilityModel:
    def __init__(
        self,
        w_slope=0.30,
        w_rock=0.25,
        w_soft_soil=0.20,
        w_obstacle=0.15,
        w_uncertainty=0.10
    ):
        self.w_slope = w_slope
        self.w_rock = w_rock
        self.w_soft_soil = w_soft_soil
        self.w_obstacle = w_obstacle
        self.w_uncertainty = w_uncertainty

    def classify_hazard_type(self, slope, rock_density, soft_soil_risk, obstacle_presence):
        if slope > 0.7 and rock_density > 0.5:
            return "rocky_steep"
        if soft_soil_risk > 0.7:
            return "soft_soil"
        if obstacle_presence > 0.7:
            return "obstacle_heavy"
        if slope < 0.3 and rock_density < 0.3 and soft_soil_risk < 0.3:
            return "safe_flat"
        return "moderate"

    def evaluate_candidate(self, candidate_features):
        slope = candidate_features["slope"]
        rock_density = candidate_features["rock_density"]
        soft_soil_risk = candidate_features["soft_soil_risk"]
        obstacle_presence = candidate_features["obstacle_presence"]
        terrain_uncertainty = candidate_features["terrain_uncertainty"]

        hazard_score = (
            self.w_slope * slope +
            self.w_rock * rock_density +
            self.w_soft_soil * soft_soil_risk +
            self.w_obstacle * obstacle_presence +
            self.w_uncertainty * terrain_uncertainty
        )

        hazard_score = max(0.0, min(1.0, hazard_score))
        traversability_score = 1.0 - hazard_score

        hazard_type = self.classify_hazard_type(
            slope, rock_density, soft_soil_risk, obstacle_presence
        )

        return {
            "hazard_score": hazard_score,
            "traversability_score": traversability_score,
            "hazard_type": hazard_type
        }

    def evaluate_all_candidates(self, terrain_inputs):
        results = {}
        for cid, features in terrain_inputs.items():
            results[cid] = self.evaluate_candidate(features)
        return results
