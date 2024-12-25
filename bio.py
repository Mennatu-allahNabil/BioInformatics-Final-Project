import streamlit as st
import numpy as np
import pandas as pd
from itertools import permutations
import bisect

# Functions for DNA processing
def GC_Content(seq):
    l = len(seq)
    num_G = seq.count("G")
    num_C = seq.count("C")
    total = num_C + num_G
    return total / l

def Complement(seq):
    dic = {"G": "C", "C": "G", "A": "T", "T": "A"}
    s = list(seq)
    for i in range(len(seq)):
        s[i] = dic[s[i]]
    return "".join(s)

def Reverse(seq):
    return "".join(reversed(seq))

def Reverse_Complement(seq):
    return Complement(Reverse(seq))

def Translation_Table(seq):
    dic = {
        "TTT": "F", "CTT": "L", "ATT": "I", "GTT": "V",
        "TTC": "F", "CTC": "L", "ATC": "I", "GTC": "V",
        "TTA": "L", "CTA": "L", "ATA": "I", "GTA": "V",
        "TTG": "L", "CTG": "L", "ATG": "M", "GTG": "V",
        "TCT": "S", "CCT": "P", "ACT": "T", "GCT": "A",
        "TCC": "S", "CCC": "P", "ACC": "T", "GCC": "A",
        "TCA": "S", "CCA": "P", "ACA": "T", "GCA": "A",
        "TCG": "S", "CCG": "P", "ACG": "T", "GCG": "A",
        "TAT": "Y", "CAT": "H", "AAT": "N", "GAT": "D",
        "TAC": "Y", "CAC": "H", "AAC": "N", "GAC": "D",
        "TAA": "*", "CAA": "Q", "AAA": "K", "GAA": "E",
        "TAG": "*", "CAG": "Q", "AAG": "K", "GAG": "E",
        "TGT": "C", "CGT": "R", "AGT": "S", "GGT": "G",
        "TGC": "C", "CGC": "R", "AGC": "S", "GGC": "G",
        "TGA": "*", "CGA": "R", "AGA": "R", "GGA": "G",
        "TGG": "W", "CGG": "R", "AGG": "R", "GGG": "G",
    }
    s = ""
    sf = ""
    flag = 0
    for i in range(0, len(seq) - 2, 3):
        if dic[seq[i:i+3]] == "M":
            flag = 1
        elif dic[seq[i:i+3]] == "*":
            flag = 0
        if flag == 1:
            s += dic[seq[i:i+3]]
        sf += dic[seq[i:i+3]]
    return sf, s

# Naive Pattern Matching Algorithm
def naive_matching(t, p):
    found = []
    pattern_length = len(p)
    text_length = len(t)
    num_of_alignments = text_length - pattern_length + 1
    
    for i in range(num_of_alignments):
        if t[i:i+pattern_length] == p:
            found.append(i)

    return found

def bad_character(t, p):
    text_length = len(t)
    pattern_length = len(p)
    table = np.zeros([4, pattern_length])

    row = ["A", "C", "G", "T"]
    for i in range(4):
        num = -1
        for j in range(pattern_length):
            if row[i] == p[j]:
                table[i, j] = -1
                num = -1
            else:
                num += 1
                table[i, j] = num

    col_names = [f"{n}_{i}" for i, n in enumerate(p)]

    shift = []
    i = 0
    while i < text_length - pattern_length + 1:
        if p == t[i:i + pattern_length]:
            shift.append(i)
        else:
            for j in range(i + pattern_length - 1, i - 1, -1):
                if t[j] != p[int(j - i)]:
                    k = row.index(t[j])
                    i += int(table[k, j - i])  
                    break
        i = int(i + 1)

    table_df = pd.DataFrame(table, columns=col_names, index=row)
    return table_df, shift

def IndexSorted(seq, ln):
    index = []
    for i in range(len(seq) - ln + 1):
        index.append((seq[i:i+ln], i))
    index.sort() 
    return index

def query(t, p, index):
    keys = [r[0] for r in index]
    st = bisect.bisect_left(keys, p[:len(keys[0])])
    en = bisect.bisect(keys, p[:len(keys[0])])
    hits = index[st:en] 
    l = [h[1] for h in hits]
    offsets = []
    for i in l:
        if t[i:i+len(p)] == p:
            offsets.append(i)
    return offsets

def compute_suffix_array(T):
    # Dictionary for character ordering
    dec = {
        '$': 0,
        'A': 1,
        'C': 2,
        'G': 3,
        'T': 4
    }
    
    table = []
    i = 2**0
    n = 0
    
    while True:    
        l = []
        dec2 = {}
        
        if i > 1:
            for j in range(len(T)):
                if not(table[n-1][j:j+i] in l):
                    l.append(table[n-1][j:j+i])
            l.sort()
            for j in range(len(l)):
                dec2[tuple(l[j])] = j
                
        row = []
        for j in range(len(T)):
            if i == 1:
                row.append(dec[T[j]])
            else:
                row.append(dec2[tuple(table[n-1][j:j+i])])
        table.append(row)
        
        flag = 0
        for j in range(len(row)):
            c = row.count(j)
            if c > 1:
                flag = 1
                break
                
        if flag == 0:
            break
        n += 1
        i = 2**n
    
    # Convert table to DataFrame for better visualization
    suffixes = [T[i:] for i in range(len(T))]
    final_order = [i for i, _ in sorted(enumerate(table[-1]), key=lambda x: x[1])]
    suffix_array_df = pd.DataFrame({
        'Index': final_order,
        'Suffix': [suffixes[i] for i in final_order]
    })
    
    return table, suffix_array_df

# Overlap Functions
def overlap(a, b, min_length=3):
    start = 0
    while True:
        start = a.find(b[:min_length], start)
        if start == -1:
            return 0
        if b.startswith(a[start:]):
            return len(a) - start
        
        start += 1

def native_overlap(reads, k):
    olap = {}
    for a, b in permutations(reads, 2):
        olen = overlap(a, b, k)
        if olen > 0:
            olap[(a, b)] = olen
    return olap

# Streamlit App
st.title("DNA Sequence Analysis")

# Input Options
st.sidebar.header("Input Options")
input_method = st.sidebar.selectbox("Select input method", ["Enter Text", "Upload FASTA File"])

def validate_sequence(sequence):
    if not set(sequence.upper()).issubset({"A", "C", "G", "T"}):
        return False
    return True

sequence = ""
if input_method == "Enter Text":
    sequence = st.sidebar.text_area("Enter DNA sequence:")
    if sequence:
        sequence = sequence.upper()  # Convert to uppercase
        if not validate_sequence(sequence):
            st.sidebar.error("Invalid DNA sequence. Only A, C, G, T are allowed. Please re-enter.")
            sequence = ""

elif input_method == "Upload FASTA File":
    uploaded_file = st.sidebar.file_uploader("Upload a FASTA file", type=["fasta", "txt"])
    if uploaded_file:
        sequence = uploaded_file.read().decode("utf-8").split("\n", 1)[1].replace("\n", "")
        sequence = sequence.upper()  # Convert to uppercase
        if not validate_sequence(sequence):
            st.sidebar.error("Invalid DNA sequence in file. Only A, C, G, T are allowed.")
            sequence = ""

if sequence:
    st.write(f"Input DNA Sequence: {sequence}")

    # Buttons for functions
    if st.button("Complement"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        st.write(f"Complement: {Complement(sequence)}")
    if st.button("Reverse"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        st.write(f"Reverse: {Reverse(sequence)}")
    if st.button("Reverse Complement"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        st.write(f"Reverse Complement: {Reverse_Complement(sequence)}")
    
    if st.button("GC Content"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        gc_content = GC_Content(sequence)
        st.write(f"GC Content: {gc_content * 100:.2f}%")

    if st.button("Translation"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        if len(sequence) < 3:
            st.warning("Translation is not allowed for sequences shorter than 3 bases.")
        else:
            full_translation, partial_translation = Translation_Table(sequence)
            st.write(f"Partial Translation: {partial_translation}")

    # Pattern Matching
    st.subheader("Pattern Matching")
    pattern = st.text_input("Enter pattern to search:")
    if pattern:
        pattern = pattern.upper()  # Ensure uppercase for safety
        selected_method = st.selectbox("Select Matching Algorithm", ["naive", "bad character"])
        
        if selected_method == "naive":
            naive_result = naive_matching(sequence, pattern)
            for i in naive_result:
                st.write(f'Pattern found at shift = {i}')

        elif selected_method == "bad character":
            table, bad_result = bad_character(sequence, pattern)
            col1, col2 = st.columns(2)

            with col1:
                st.write(table)

            with col2:
                st.write("### Pattern Occurrences")
                for i in bad_result:
                    st.write(f'Pattern occurs at shift = {i}')

    # Index-Sorted Query
    st.subheader("Index-Sorted Query")
    pattern_for_query = st.text_input("Enter pattern for Index-Sorted Query:")
    k = st.number_input("Enter k (length of k-mers):", min_value=1, value=3, step=1)

    if pattern_for_query and k > 0:
        pattern_for_query = pattern_for_query.upper()  # Ensure uppercase for safety
        index = IndexSorted(sequence, k)
        query_result = query(sequence, pattern_for_query, index)
        st.write(f"Sorted Index: {index}")
        st.write("Pattern found at positions:")
        for i in query_result:
            st.write(f"- Position: {i}")

    # New Suffix Array Section
    st.subheader("Suffix Array Analysis")
    if st.button("Generate Suffix Array"):
        sequence = sequence.upper()  # Ensure uppercase for safety
        # Ensure sequence ends with $
        if not sequence.endswith('$'):
            sequence_with_terminal = sequence + '$'
        else:
            sequence_with_terminal = sequence
            
        # Compute suffix array
        table, suffix_array_df = compute_suffix_array(sequence_with_terminal)
        
        # Display results
        st.write("### Suffix Array Table")
        st.dataframe(pd.DataFrame(table))
        
        st.write("### Sorted Suffixes")
        st.dataframe(suffix_array_df)

    # Overlap Section
    st.subheader("Overlap Analysis")
    reads_input = st.text_area("Enter DNA reads (one per line):")
    min_overlap = st.number_input("Minimum overlap length (k):", min_value=1, value=3, step=1)

    if reads_input:
        reads = [read.upper() for read in reads_input.strip().split("\n")]  # Ensure uppercase for all reads
        st.write("### Input Reads")
        for read in reads:
            st.write(f"- {read}")

        # Compute overlaps
        overlaps = native_overlap(reads, min_overlap)

        if overlaps:
            st.write("### Overlap Results")

            # Create a DataFrame for visualization
            overlap_data = []
            for (a, b), olen in overlaps.items():
                overlap_data.append((a, b, olen))

            overlap_df = pd.DataFrame(overlap_data, columns=["Read A", "Read B", "Overlap Length"])
            st.dataframe(overlap_df)

        else:
            st.write("No overlaps found with the given minimum overlap length.")

else:
    st.warning("Please input a DNA sequence or upload a FASTA file.")

