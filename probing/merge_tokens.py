import torch

train_st = torch.load("../speechtokenizer_decode/speechtokenizer_tokens/frozen/train_utt2speech_token.pt")
test_st = torch.load("../speechtokenizer_decode/speechtokenizer_tokens/frozen/test_utt2speech_token.pt")
val_st = torch.load("../speechtokenizer_decode/speechtokenizer_tokens/frozen/val_utt2speech_token.pt")

train_wt = torch.load("../wavtokenizer_decode/data/wavtokenizer_tokens/train_utt2speech_token.pt")
test_wt = torch.load("../wavtokenizer_decode/data/wavtokenizer_tokens/test_utt2speech_token.pt")
val_wt = torch.load("../wavtokenizer_decode/data/wavtokenizer_tokens/val_utt2speech_token.pt")

all_tokens_st = {**train_st, **test_st, **val_st}
all_tokens_wt = {**train_wt, **test_wt, **val_wt}

print(f"Total: {len(all_tokens_st)} (train={len(train_st)}, test={len(test_st)}, val={len(val_st)})")
print(f"Total: {len(all_tokens_wt)} (train={len(train_wt)}, test={len(test_wt)}, val={len(val_wt)})")

torch.save(all_tokens_st, "extracted_tokens/speechtokenizer/utt2speech_token_all.pt")
torch.save(all_tokens_wt, "extracted_tokens/wavtokenizer/utt2speech_token_all.pt")