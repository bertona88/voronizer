import copy
import importlib
import userInput as u
import main


BASE_KEYS = [
    "MODEL",
    "SUPPORT",
    "NET",
    "NET_CONNECT",
    "SMOOTH",
    "MODEL_THRESH",
    "MODEL_CELL_MM",
    "MODEL_SHELL_MM",
    "NET_THICKNESS_MM",
    "RESOLUTION",
    "FILE_NAME",
    "SHOW_PLOTS",
    "AUTO_EXPORT",
    "RUN_LABEL",
]

BASELINE_SETTINGS = {k: copy.copy(getattr(u, k)) for k in BASE_KEYS}

SWEEPS = [
    (
        "net_surface_thin",
        {
            "NET": True,
            "NET_CONNECT": False,
            "MODEL_THRESH": 0.22,
            "MODEL_CELL_MM": 0.25,
            "NET_THICKNESS_MM": 1.0,
            "SMOOTH": False,
        },
    ),
    (
        "net_hybrid_medium",
        {
            "NET": True,
            "NET_CONNECT": True,
            "MODEL_THRESH": 0.30,
            "MODEL_CELL_MM": 0.35,
            "NET_THICKNESS_MM": 2.0,
            "SMOOTH": False,
        },
    ),
    (
        "net_hybrid_dense",
        {
            "NET": True,
            "NET_CONNECT": True,
            "MODEL_THRESH": 0.40,
            "MODEL_CELL_MM": 0.50,
            "NET_THICKNESS_MM": 3.0,
            "SMOOTH": False,
        },
    ),
]


def apply_settings(overrides):
    for key, value in BASELINE_SETTINGS.items():
        setattr(u, key, copy.copy(value))
    for key, value in overrides.items():
        setattr(u, key, value)
    u.RESOLUTION = 200
    u.SHOW_PLOTS = False
    u.AUTO_EXPORT = True
    u.MODEL = True
    u.SUPPORT = False
    u.FILE_NAME = "sphere.stl"


def main_sweeps():
    for label, overrides in SWEEPS:
        apply_settings(overrides)
        u.RUN_LABEL = label
        print(f"\n=== Running sweep: {label} ===")
        main.main()
    for key, value in BASELINE_SETTINGS.items():
        setattr(u, key, value)


if __name__ == "__main__":
    main_sweeps()
