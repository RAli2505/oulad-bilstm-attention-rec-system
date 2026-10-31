"""Run improved OULAD experiments; all settings and artifacts are saved separately."""
import sys
from run_risk_article import main

if __name__ == '__main__':
    if '--improved' not in sys.argv:
        sys.argv.append('--improved')
    main()
