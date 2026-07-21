"""
Check if add_chunks.json matches the dataset.
"""
import json
import os
from collections import defaultdict


def load_dataset(dataset_path):
    """Load the original dataset and extract all messages."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_messages = []
    for item in dataset:
        sample_id = item.get('sample_id', 'unknown')
        conversation = item.get('conversation', {})

        # Collect all session messages
        sessions = [k for k in conversation.keys() if k.startswith('session_')]
        sessions.sort()

        for session_key in sessions:
            session_data = conversation[session_key]
            if isinstance(session_data, list):
                for msg in session_data:
                    all_messages.append({
                        'sample_id': sample_id,
                        'session': session_key,
                        'speaker': msg.get('speaker', ''),
                        'text': msg.get('text', ''),
                        'dia_id': msg.get('dia_id', ''),
                    })

    return all_messages


def load_chunks(chunks_path):
    """Load the saved chunks and extract all messages."""
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    all_messages = []
    for chunk_entry in chunks_data:
        chunks = chunk_entry.get('chunks', [])
        for chunk in chunks:
            messages = chunk.get('messages', [])
            conversation_id = chunk.get('conversation_id', '')
            for msg in messages:
                all_messages.append({
                    'conversation_id': conversation_id,
                    'speaker_name': msg.get('speaker_name', ''),
                    'content': msg.get('content', ''),
                    'dia_id': msg.get('dia_id', ''),
                })

    return all_messages


def normalize_message(msg):
    """Create a normalized key for message comparison."""
    # For dataset messages
    if 'text' in msg:
        return (
            msg.get('dia_id', ''),
            msg.get('speaker', ''),
            msg.get('text', ''),
        )
    # For chunk messages
    elif 'content' in msg:
        return (
            msg.get('dia_id', ''),
            msg.get('speaker_name', ''),
            msg.get('content', ''),
        )
    return None


def check_consistency(dataset_path, chunks_path):
    """Check if chunks match dataset."""
    dataset_msgs = load_dataset(dataset_path)
    chunks_msgs = load_chunks(chunks_path)

    print(f"=" * 60)
    print(f"Dataset messages: {len(dataset_msgs)}")
    print(f"Chunks messages:  {len(chunks_msgs)}")
    print(f"=" * 60)

    # Create normalized keys
    dataset_keys = set()
    for msg in dataset_msgs:
        key = normalize_message(msg)
        if key:
            dataset_keys.add(key)

    chunks_keys = set()
    for msg in chunks_msgs:
        key = normalize_message(msg)
        if key:
            chunks_keys.add(key)

    # Find missing and extra
    missing_in_chunks = dataset_keys - chunks_keys
    extra_in_chunks = chunks_keys - dataset_keys

    print(f"\nChunks: {len(chunks_msgs)} entries across {len(chunks_msgs[0]) if chunks_msgs else 0} chunk calls")

    if missing_in_chunks:
        print(f"\n❌ MISSING {len(missing_in_chunks)} messages in chunks:")
        for key in list(missing_in_chunks)[:10]:
            print(f"   dia_id={key[0]}, speaker={key[1]}, text={key[2][:50]}...")
        if len(missing_in_chunks) > 10:
            print(f"   ... and {len(missing_in_chunks) - 10} more")
    else:
        print("\n✅ No missing messages in chunks")

    if extra_in_chunks:
        print(f"\n⚠️  EXTRA {len(extra_in_chunks)} messages in chunks (not in dataset):")
        for key in list(extra_in_chunks)[:10]:
            print(f"   dia_id={key[0]}, speaker={key[1]}, text={key[2][:50]}...")
        if len(extra_in_chunks) > 10:
            print(f"   ... and {len(extra_in_chunks) - 10} more")
    else:
        print("✅ No extra messages in chunks")

    # Check by conversation
    print("\n" + "=" * 60)
    print("Per-conversation breakdown:")
    print("=" * 60)

    # Count per sample_id in dataset
    dataset_by_sample = defaultdict(list)
    for msg in dataset_msgs:
        dataset_by_sample[msg['sample_id']].append(msg)

    # Count per conversation_id in chunks
    chunks_by_conv = defaultdict(list)
    for chunk_entry in chunks_msgs:
        chunks_by_conv[chunk_entry['conversation_id']].append(chunk_entry)

    all_ids = sorted(set(list(dataset_by_sample.keys()) + list(chunks_by_conv.keys())))
    for cid in all_ids:
        ds_count = len(dataset_by_sample.get(cid, []))
        ck_count = len(chunks_by_conv.get(cid, []))
        status = "✅" if ds_count == ck_count else "❌"
        print(f"  {cid}: dataset={ds_count}, chunks={ck_count} {status}")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)  # LifeBench_eval
    dataset_path = os.path.join(base_dir, "datasets", "smoke", "locomo_smoke.json")
    chunks_path = os.path.join(base_dir, "results", "data_test", "add_chunks.json")

    print(f"Checking:")
    print(f"  Dataset: {dataset_path}")
    print(f"  Chunks:  {chunks_path}")
    print()

    check_consistency(dataset_path, chunks_path)
