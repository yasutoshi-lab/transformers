import os
W = os.path.expanduser("~/freesia-work/v2")
p = f"{W}/freesia-pt/README.md"; s = open(p).read()
s = s.replace("| ja | test | 10,000 | 10,103,894 | 6,667 / 3,333 |", "| ja | test | 10,000 | 10,113,318 | 6,667 / 3,333 |")
s = s.replace("| en | test | 10,000 | 10,285,579 | 6,667 / 3,333 |", "| en | test | 10,000 | 10,289,627 | 6,667 / 3,333 |")
s = s.replace("- `validation` is used only to log loss during training. `test` is held out for the final evaluation.",
              "- `validation` is used only to log loss during training. `test` is held out for the final evaluation. The test split was filtered with a stricter rule: no document has a shingle Jaccard of 0.5 or higher with any train or validation document. This is checked with MinHash LSH using 20 bands × 5 rows. Rejected documents were replaced with unused pool documents.")
assert "10,113,318" in s and "stricter rule" in s
open(p, "w").write(s)
p = f"{W}/tokenizer-pt/README.md"; s = open(p).read()
old = "The `validation` and `test` splits are the whole pool validation and test splits."
assert old in s
s = s.replace(old, old + " The test split was additionally filtered: no document has a shingle Jaccard of 0.5 or higher with the pretraining train documents or with validation.")
open(p, "w").write(s)
print("README_FIXED")
