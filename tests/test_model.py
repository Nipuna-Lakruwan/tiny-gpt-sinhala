import torch

from model import GPT, GPTConfig


def test_forward_and_causality():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab=50, ctx=16, n_layer=2, n_head=2, d=32, dropout=0.0)).eval()
    x = torch.randint(0, 50, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 50) and loss.item() > 0
    # changing a future token must not change earlier predictions
    x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 50
    l2, _ = m(x2)
    assert torch.allclose(logits[:, :-1], l2[:, :-1], atol=1e-5)


def test_explicit_attention_matches_fused():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab=50, ctx=16, n_layer=2, n_head=2, d=32, dropout=0.0)).eval()
    x = torch.randint(0, 50, (2, 10))
    a, _ = m(x)
    b, _, atts = m(x, return_attn=True)
    assert torch.allclose(a, b, atol=1e-5)
    assert atts[0].shape == (2, 2, 10, 10)
    assert torch.allclose(atts[0].sum(-1), torch.ones(2, 2, 10), atol=1e-5)
