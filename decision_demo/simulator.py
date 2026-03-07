import matplotlib.pyplot as plt
from decision_demo.models_mock import create_mock_maps, create_mock_resources
from decision_demo.decision_manager import DecisionManager


def run_simulation(steps=25, size=20):
    science_map, hazard_map = create_mock_maps(size=size)
    resources = create_mock_resources()
    manager = DecisionManager()

    position = (2, 2)
    previous_position = None
    visited_positions = [position]

    path = [position]
    decisions = []

    for step in range(steps):
        decision = manager.decide(
            position,
            science_map,
            hazard_map,
            resources,
            previous_position=previous_position,
            visited_positions=visited_positions
        )
        decisions.append(decision)

        if decision["mode"] in ["SAFE_STOP", "RETURN"]:
            path.append(decision["target"])
            break

        previous_position = position
        position = decision["target"]
        path.append(position)
        visited_positions.append(position)

        # Fake energy drain
        resources["energy_state"] -= 0.01
        resources["energy_state"] = max(resources["energy_state"], 0.0)

    return science_map, hazard_map, path, decisions


def plot_simulation(science_map, hazard_map, path):
    plt.figure(figsize=(8, 8))
    combined = science_map - hazard_map
    plt.imshow(combined, cmap="viridis", origin="lower")

    xs = [p[1] for p in path]
    ys = [p[0] for p in path]
    plt.plot(xs, ys, color="red", marker="o", label="Path")

    plt.scatter(xs[0], ys[0], color="white", s=100, label="Start")
    plt.scatter(xs[-1], ys[-1], color="yellow", s=100, label="End")

    plt.title("Umbrella Decision Demo")
    plt.legend()
    plt.colorbar(label="Science - Hazard")
    plt.show()


if __name__ == "__main__":
    science_map, hazard_map, path, decisions = run_simulation()
    plot_simulation(science_map, hazard_map, path)

    for i, d in enumerate(decisions[:15]):
        print(f"Step {i}: mode={d['mode']} target={d['target']} reason={d['reason']}")
