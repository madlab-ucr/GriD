import zstandard as zstd
import json

import numpy as np

import sys

# Get the file name from command line argument
if len(sys.argv) < 2:
    print('Please provide a file name as an argument')
    sys.exit(1)
file_name = sys.argv[1]

# Open the compressed data file
with open(file_name, 'rb') as f:
    compressed_data = f.read()

# Decompress the data
decompressed_data = zstd.decompress(compressed_data)

# Load the JSON data
data = json.loads(decompressed_data)

print('NUMBER OF POSTS:', len(data))
print()

# Pretty print the content of the posts
for post in data:
    print('Title:', post['title'])
    print('Subreddit:', post['subreddit'])
    print('Score:', post['score'])
    print('Random Comment:', post['comments'][np.random.randint(0, len(post['comments']))]['body'])

    print('GPT:', post['gpt'])

    for comment in post['comments']:
        print('  Score:', comment['score'])
        print('  Body:', comment['body'])
    print()
