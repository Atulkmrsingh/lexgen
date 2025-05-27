import sacrebleu

def calculate_bleu(ref_file, candidate_file):
    # Read the contents of the reference file
    with open(ref_file, 'r') as ref:
        ref_lines = ref.readlines()

    # Read the contents of the candidate file
    with open(candidate_file, 'r') as candidate:
        candidate_lines = candidate.readlines()

    # Convert the lines into lists of sentences
    ref_sentences = [line.strip() for line in ref_lines]
    candidate_sentences = [line.strip() for line in candidate_lines]

    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(candidate_sentences, [ref_sentences])

    return bleu.score

# Example usage
ref_file_path = 'ref_fname.txt'
candidate_file_path = 'output.txt'
bleu_score = calculate_bleu(ref_file_path, candidate_file_path)
print(f"BLEU score: {bleu_score}")
