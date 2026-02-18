"""Migrate models from the typo'd HF collection to the correctly-spelled one.

Copies all models from "the-obfuscation-altas" (typo) to "the-obfuscation-atlas" (correct).
The old collection is kept intact since it's referenced in the paper.

Usage:
    python migrate_hf_collection.py --dry-run   # Preview what would be added
    python migrate_hf_collection.py              # Actually migrate
"""

import argparse
import time

from huggingface_hub import add_collection_item, get_collection

ORG = "AlignmentResearch"
OLD_COLLECTION_SLUG = f"{ORG}/the-obfuscation-altas"
NEW_COLLECTION_SLUG = f"{ORG}/the-obfuscation-atlas"


def get_collection_items(collection_slug: str) -> list[dict]:
    """Fetch all items from a HuggingFace collection.

    Args:
        collection_slug: Full collection slug like "org/collection-name-hash"

    Returns:
        List of dicts with 'item_id' and 'item_type' keys.
    """
    collection = get_collection(collection_slug)
    items = []
    for item in collection.items:
        items.append({"item_id": item.item_id, "item_type": item.item_type})
    return items


def migrate(dry_run: bool = False) -> None:
    """Copy all models from old collection to new collection.

    Args:
        dry_run: If True, just print what would happen without making changes.
    """
    print(f"Source collection (typo):   {OLD_COLLECTION_SLUG}")
    print(f"Target collection (correct): {NEW_COLLECTION_SLUG}")
    print()

    # Fetch items from old collection
    print("Fetching items from source collection...")
    old_items = get_collection_items(OLD_COLLECTION_SLUG)
    print(f"Found {len(old_items)} items in source collection")

    if not old_items:
        print("Nothing to migrate.")
        return

    # Fetch items already in new collection to avoid duplicates
    print("Fetching items from target collection...")
    try:
        new_items = get_collection_items(NEW_COLLECTION_SLUG)
        existing_ids = {item["item_id"] for item in new_items}
        print(f"Found {len(new_items)} items already in target collection")
    except Exception as e:
        print(f"Could not fetch target collection (may not exist yet): {e}")
        existing_ids = set()

    to_add = [item for item in old_items if item["item_id"] not in existing_ids]
    already_present = len(old_items) - len(to_add)

    print(f"\nItems to add: {len(to_add)}")
    if already_present:
        print(f"Already in target: {already_present}")
    print()

    if not to_add:
        print("All items already present in target collection. Nothing to do.")
        return

    succeeded = 0
    failed = 0

    for item in to_add:
        item_id = item["item_id"]
        item_type = item["item_type"]

        if dry_run:
            print(f"[DRY-RUN] Would add {item_type} {item_id}")
            succeeded += 1
            continue

        try:
            add_collection_item(
                collection_slug=NEW_COLLECTION_SLUG,
                item_id=item_id,
                item_type=item_type,
            )
            print(f"Added {item_type} {item_id}")
            succeeded += 1
            # Small delay to be polite to the API
            time.sleep(0.2)
        except Exception as e:
            print(f"Failed to add {item_id}: {e}")
            failed += 1

    print(f"\nDone. Added: {succeeded}, Failed: {failed}")
    if not dry_run:
        print(f"\nOld collection kept at: https://huggingface.co/collections/{OLD_COLLECTION_SLUG}")
        print(f"New collection at:      https://huggingface.co/collections/{NEW_COLLECTION_SLUG}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
