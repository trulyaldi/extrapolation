#!/usr/bin/env python3

import argparse

from database.results import find_results, list_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system")
    parser.add_argument("--expectation")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.system and args.expectation:
        rows = find_results(args.system, args.expectation)
    else:
        rows = list_results(args.limit)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
