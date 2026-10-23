#!/usr/bin/env python3
"""Runner script for Replication 1: Daraeizadeh et al. (2020)"""
import json
import argparse
from siliqun.api import SiliqunEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_e1.json")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = json.load(f)
        
    print(f"Running {config['experiment']} on {config['hardware_profile']}")
    # SiliQun API execution logic goes here

if __name__ == "__main__":
    main()
