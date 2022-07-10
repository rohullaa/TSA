def BIO2BIOUL(input_file, output_file):
    print("Convert BIO -> BIOUL for file:", input_file)
    with open(input_file,'r', encoding='utf8') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    # label_type = labels[idx].split('-')[-1]
                    label_type = labels[idx][2:]
                    if "B-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" U-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" B-"+label_type+"\n")
                    elif "I-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" L-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" I-"+label_type+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIOUL file generated:", output_file)

BIO2BIOUL(input_file = './data/dev.conll', output_file='dev_bioul.conll')