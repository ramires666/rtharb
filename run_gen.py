import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from standalone_report.generate_report import main

if __name__ == "__main__":
    main()
