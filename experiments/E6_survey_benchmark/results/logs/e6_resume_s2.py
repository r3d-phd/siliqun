
import subprocess, sys
sys.exit(sum(subprocess.call([sys.executable, 'C:\\Users\\raada\\Documents\\PhD-Audit\\repos\\siliqun\\experiments\\E6_survey_benchmark\\run_e6.py',
    "--algo", a, "--seed", "2"]) for a in ['sac', 'ddpg']) != 0)
