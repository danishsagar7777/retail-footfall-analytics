from src.utils.config import load_config


def main():
    config = load_config()

    print("Configuration loaded successfully.")
    print(f"Camera source: {config['camera']['source']}")
    print(f"Target FPS: {config['processing']['target_fps']}")
    print(f"Detection model: {config['detection']['model']}")


if __name__ == "__main__":
    main()
