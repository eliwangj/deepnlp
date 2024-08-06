def get_prf(eval_filepath):
    with open(eval_filepath, 'r') as eval_file:
        lines = eval_file.readlines()
        line_count = 0
        title_index = -1
        for line in lines:
            if len(line) == 0:
                continue
            splits = line.split()
            if splits == ['SEMANTIC', 'SCORES:']:
                title_index = line_count # recored where the heading 'SEMANTIC SCORES:' is
            line_count += 1

        p = float(lines[title_index + 1].split()[-2])  # Labeled precision
        r = float(lines[title_index + 2].split()[-2])  # Labeled recall
        f = float(lines[title_index + 3].split()[-1])  # Labeled F1
    
    return p, r, f 

# if __name__ == '__main__':
#     eval_filepath = '/Users/eliwang/NLP_Work/NLP_Lab/DeepnlpToolkit/dnlptk-main-sep2/DeepNLP/model/dev_eval.txt'
#     p, r, f = get_prf(eval_filepath)
#     print("p: ", p, "r: ", r, "f: ", f)