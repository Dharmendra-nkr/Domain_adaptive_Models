from datasets import load_dataset

d = load_dataset('bigbio/bc5cdr', 'bc5cdr_bigbio_kb', trust_remote_code=True)
print('Splits:', list(d.keys()))
print('Columns:', d['train'].column_names)
print('First doc keys:', d['train'][0].keys())

first = d['train'][0]
print('First passages length:', len(first['passages']))
print('First entities length:', len(first['entities']))
print('Passages structure:', first['passages'][0].keys() if first['passages'] else None)
print('Entities structure:', first['entities'][0].keys() if first['entities'] else None)
