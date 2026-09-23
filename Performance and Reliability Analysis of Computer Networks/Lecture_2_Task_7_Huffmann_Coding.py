

# Huffman Coding
text = "living is aalborg is fun"


# Frequency table
freq_table = {}

for char in text:
    freq_table[char] = freq_table.get(char, 0) + 1

print("Frequency Table:")
print(freq_table)


# Build Huffman Tree
nodes = []

for char, freq in freq_table.items():
    node = {
        "char": char,
        "freq": freq,
        "left": None,
        "right": None
    }
    nodes.append(node)

# Sort by frequency
nodes = sorted(nodes, key=lambda node: node["freq"])

while len(nodes) > 1:
    # Remove two smallest nodes
    left = nodes.pop(0)
    right = nodes.pop(0)

    print(
        f"Merging {left['char']}:{left['freq']} "
        f"and {right['char']}:{right['freq']}"
    )

    # Create new internal node
    merged_node = {
        "char": None,
        "freq": left["freq"] + right["freq"],
        "left": left,
        "right": right
    }

    # Add it back
    nodes.append(merged_node)

    # Sort again
    nodes = sorted(nodes, key=lambda node: node["freq"])


# Last remaining node is the root
root = nodes[0]

print("\nRoot frequency:", root["freq"])


# 3. Generate Huffman codes
codes = {}


def generate_codes(node, code=""):

    # If we reach a character
    if node["char"] is not None:
        codes[node["char"]] = code
        return

    # Left = 0
    generate_codes(node["left"], code + "0")

    # Right = 1
    generate_codes(node["right"], code + "1")


generate_codes(root)

print("\nHuffman Codes:")

for char, code in codes.items():

    if char == " ":
        print(f"space: {code}")
    else:
        print(f"{char}: {code}")

# 4. Encode message

message = "fun aalborg"

encoded = ""

for char in message:
    encoded += codes[char]

print("\nOriginal message:")
print(message)

print("Encoded message:")
print(encoded)

print("Number of bits:", len(encoded))

# 5. Decode message

def decode(encoded, root):

    decoded = ""
    current = root

    for bit in encoded:

        if bit == "0":
            current = current["left"]
        else:
            current = current["right"]

        # Found a character
        if current["char"] is not None:
            decoded += current["char"]

            # Go back to root
            current = root

    return decoded


decoded = decode(encoded, root)

print("\nDecoded message:")
print(decoded)

# 6. Compression comparison

message = "aalborg is fun"

# ASCII
ascii_bits = len(message) * 8

# Huffman
huffman_encoded = ""

for char in message:
    huffman_encoded += codes[char]

huffman_bits = len(huffman_encoded)

print("\nCompression:")

print("Message:", message)
print("ASCII bits:", ascii_bits)
print("Huffman bits:", huffman_bits)

compression = (1 - huffman_bits / ascii_bits) * 100

print(f"Compression: {compression:.2f}%")