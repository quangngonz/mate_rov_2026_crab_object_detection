from __future__ import annotations

from typing import List, Tuple

from cv2_enumerate_cameras import enumerate_cameras


def get_camera_list() -> List[Tuple[int, str]]:
    """Return available camera devices as (index, name)."""
    cameras: List[Tuple[int, str]] = []
    try:
        for idx, camera_info in enumerate(enumerate_cameras()):
            cam_index = int(getattr(camera_info, "index", idx))
            cam_name = str(getattr(camera_info, "name", f"Camera {cam_index}"))
            cameras.append((cam_index, cam_name))
    except Exception as exc:
        print(f"Warning: camera enumeration failed: {exc}")

    if not cameras:
        cameras = [(0, "Camera 0")]
    return cameras


def prompt_user_selection(available_cams: List[Tuple[int, str]]) -> str:
    """Prompt user to select camera index from terminal list."""
    print("\nSelect Camera Source:")
    print("-" * 60)
    print(f"{'Index':<8} | Device Name")
    print("-" * 60)
    for idx, name in available_cams:
        print(f"{idx:<8} | {name}")
    print("-" * 60)

    valid_indices = {idx for idx, _ in available_cams}
    while True:
        choice = input("Enter camera index (default: 0): ").strip()
        if choice == "":
            return "0"
        try:
            selected = int(choice)
            if selected in valid_indices:
                return str(selected)
            print(f"Invalid index '{selected}'. Choose one from the list.")
        except ValueError:
            print("Invalid input. Please enter a numeric camera index.")
