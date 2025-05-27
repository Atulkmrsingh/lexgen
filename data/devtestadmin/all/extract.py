# Open the source file for reading
with open("en-ta/outtest.ta", "r", encoding="utf-8") as source_file:
    # Read the content from the source file
    content = source_file.read()

# Split the content by newline character
lines = content.split('\n')

# Extract Hindi words after the '-' character
hindi_words = [line.split('(')[0].strip() for line in lines]

# Create a new file for writing the extracted Hindi words
with open("hindi_words.txt", "w", encoding="utf-8") as output_file:
    # Write the extracted Hindi words to the output file
    for word in hindi_words:
        output_file.write(word + '\n')
