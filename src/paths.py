from pathlib import Path


class ProjectPaths:
    """
    Resolves paths relative to the project root to ensure consistency across scripts
    """

    # Anchor: src/paths.py -> src -> ROOT
    _ROOT = Path(__file__).resolve().parent.parent

    @classmethod
    def get_root(cls) -> Path:
        return cls._ROOT

    @classmethod
    def get_raw_data_dir(cls) -> Path:
        """
        Returns: datasets/raw/
        """
        return cls._ROOT / 'datasets'


if __name__ == '__main__':
    print(ProjectPaths.get_root())
