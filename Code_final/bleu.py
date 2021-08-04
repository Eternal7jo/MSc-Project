#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import math
import numpy as np
import re
import os
import subprocess
import tempfile

def truncate(f, n):
    if math.isnan(f):
        return f
    return math.floor(f * 10 ** n) / 10 ** n



def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES multi-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)


    # Dump hypotheses and references to tempfiles
    # hypothesis_file = tempfile.NamedTemporaryFile()
    # hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    # hypothesis_file.write(b"\n")
    # hypothesis_file.flush()
    # reference_file = tempfile.NamedTemporaryFile()
    # reference_file.write("\n".join(references).encode("utf-8"))
    # reference_file.write(b"\n")
    # reference_file.flush()

    hypothesis_file = open('./hypothesis_file.txt', 'w', encoding='utf-8')
    hypothesis_file.write("\n".join(hypotheses))
    hypothesis_file.write("\n")
    hypothesis_file.flush()

    reference_file = open('./reference_file.txt', 'w', encoding='utf-8')
    reference_file.write("\n".join(hypotheses))
    reference_file.write("\n")
    reference_file.flush()

    hypothesis_file.close()
    reference_file.close()

    # Calculate BLEU using multi-bleu script
    with open('hypothesis_file.txt', "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += ['reference_file.txt']
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


if __name__ == '__main__':

    text = 'Today is a good day'
    gold = 'Today a good day'
    bleu = truncate(moses_multi_bleu(np.array([text]), np.array([gold]))*100, 2)
    print(bleu)