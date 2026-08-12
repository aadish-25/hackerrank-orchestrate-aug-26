import csv

messages = list(csv.DictReader(open('dataset/messages.csv', encoding='utf-8')))
outputs = {row['message_id']: row for row in csv.DictReader(open('dataset/output.csv', encoding='utf-8'))}

print('=== Suspicious Messages ===')
for m in messages:
    text = m['message_text'].lower()
    if 'ignore' in text or 'override' in text or 'system instruction' in text or 'prompt' in text or 'bypass' in text:
        msg_id = m['message_id']
        out = outputs.get(msg_id, {})
        print(f'\nID: {msg_id}')
        print(f'Text: {m["message_text"]}')
        print(f'Media: {m.get("media_type", "None")} -> {m.get("media_file_path", "")}')
        print(f'Pipeline Action: {out.get("action")}')
        print(f'Pipeline Reason: {out.get("reason")}')
        print(f'Pipeline Confidence: {out.get("confidence")}')
