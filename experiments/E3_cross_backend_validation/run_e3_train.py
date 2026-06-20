#!/usr/bin/env python3
"""Runner script for Replication 3: Training on SiliQun"""
import json
import argparse
from siliqun.api import SiliqunEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_e3.json")
    args = parser.parse_args()
    print("Running E3 training phase")

if __name__ == "__main__":
    main()
