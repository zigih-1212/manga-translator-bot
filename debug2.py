with open('translator/pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

line = lines[386]  # 0-indexed = line 387
print('Line 387:', repr(line[:120]))
print()
# Find all non-ASCII chars
for i, c in enumerate(line):
    if ord(c) > 127:
        print(f'  pos {i}: U+{ord(c):04X} = {c}')
