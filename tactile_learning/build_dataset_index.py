"""Validate saved tactile episodes and create a reusable training index."""
import argparse
from collections import Counter

from tactile.data.index import build_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Dataset root containing episode directories")
    parser.add_argument("--output", required=True, help="Output index JSON")
    parser.add_argument("--include-invalid", action="store_true", help="Diagnostic use: allow metadata.valid=false")
    parser.add_argument("--allow-partial-contact", action="store_true", help="Include one/two-contact episodes for robustness studies")
    args = parser.parse_args()
    result = build_index(args.root, args.output, require_valid=not args.include_invalid,
                         require_three_contacts=not args.allow_partial_contact)
    print(f"Accepted {len(result['episodes'])}, rejected {len(result['rejected'])}, classes {len(result['classes'])}")
    for label, count in sorted(Counter(e['label'] for e in result['episodes']).items()):
        print(f"  {label}: {count}")
    print(f"Index: {args.output}")
    if not result["episodes"]:
        parser.exit(2, "No usable episodes. Inspect the rejected records in the output index.\n")


if __name__ == "__main__":
    main()
