
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================================
#  Modello: Conditional GAN per sequenze temporali multivariate basata su Transformer
#  - Generator (G): produce direttamente l'intera sequenza [B, L, C] in un solo passaggio,
#    (B batch size, L lunghezza seq, C nfeatures) guidato da rumore latente z e condizione s. 

#  - Discriminator (D): Projection Discriminator con spectral normalization. Riassume la
#    sequenza tramte un token CLS, poi combina uno score "non condizionale" con un termine
#    di proiezione che misura la coerenza con la condizione (s).
#
#  NOTE PRATICHE:
#  - L'output del G usa tanh
#  - seq_len di G e D deve coincidere; anche cond_dim deve essere identico su G e D.
#  - time_tokens sono parametri appresi che fungono da base "posizionale" per ogni timestep;
#    il positional encoding sinusoidale aggiunge una componente deterministica di posizione.
# =========================================================================================


# ---------- Positional Encoding (sinusoidale) ----------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Encoding posizionale sinusoidale in stile Transformer.
    Non ha parametri allenabili: dipende solo da (max_len, d_model)
    
    Args:
        d_model: dimensione dei token (embedding dim).
        max_len: lunghezza massima supportata (deve essere >= seq_len effettiva).

    Forward:
        x: Tensor [B, L, d] usato solo per leggere L; il contenuto non viene usato.
        return: Tensor [1, L, d] pronto per essere broadcast-ato su B.
    """
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dimensioni pari
        pe[:, 1::2] = torch.cos(position * div_term)  # dimensioni dispari
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, d] (no grad)

    def forward(self, x):  # x: [B, L, d]
        return self.pe[:, :x.size(1), :]


# ---------- Conditional LayerNorm (FiLM) ----------
class ConditionalLayerNorm(nn.Module):
    """
    LayerNorm condizionale (stile FiLM): applica LN e poi modula (affine) tramite gamma/beta
    calcolati da un vettore di condizione c (es. concatenazione [z, s]).

    Args:
        normalized_shape: dimensione del canale/embedding (d_model).
        cond_dim: dimensione del vettore di condizione c.
        eps: termine di stabilizzazione numerica per LayerNorm.

    Forward:
        x: Tensor [B, L, d]
        c: Tensor [B, cond_dim]
        return: Tensor [B, L, d] modulato da c
    """
    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, elementwise_affine=False, eps=eps)  # affine via FiLM
        self.gamma = nn.Linear(cond_dim, normalized_shape)
        self.beta = nn.Linear(cond_dim, normalized_shape)

    def forward(self, x, c):  # x: [B,L,d], c: [B,cond]
        y = self.ln(x)
        g = self.gamma(c).unsqueeze(1)  # [B,1,d]
        b = self.beta(c).unsqueeze(1)   # [B,1,d]
        return y * (1 + g) + b          # shift+scale dipendenti da c


# ---------- Transformer Blocks ----------
class CondTransformerBlock(nn.Module):
    """
    Blocco Transformer pre-norm con modulazione condizionale:
    - LN condizionale (FiLM) prima dell'attenzione
    - self-attention multi-testa (batch_first=True → input [B,L,d])
    - LN condizionale + feed-forward (GELU) con dropout
    - residual connections su entrambe le sottostrutture

    Args:
        d_model, nhead, dim_ff, cond_dim, dropout: iperparametri standard Transformer.
    """
    def __init__(self, d_model, nhead, dim_ff, cond_dim, dropout=0.1):
        super().__init__()
        self.ln1 = ConditionalLayerNorm(d_model, cond_dim)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = ConditionalLayerNorm(d_model, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, c):
        # Attenzione condizionata: la normalizzazione è modulata da c
        h = self.ln1(x, c)
        attn_out, _ = self.attn(h, h, h, need_weights=False)  # self-attention bidirezionale
        x = x + attn_out
        # Feed-forward condizionato
        h2 = self.ln2(x, c)
        x = x + self.ff(h2)
        return x


class TransformerBlock(nn.Module):
    """
    Blocco Transformer standard (pre-norm) usato nel Discriminatore.
    """
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h2 = self.ln2(x)
        x = x + self.ff(h2)
        return x


class Generator(nn.Module):
    """
    Generatore: produce sequenze multivariate coerenti con la condizione s.

    Idea chiave:
    - Si costruisce una "tela temporale" di lunghezza L con token appresi (time_tokens) + PE sinusoidale.
    - Si inietta un bias globale g (proiezione di z ed s) su tutti i timestep.
    - Si attraversa una pila di CondTransformerBlock modulati dalla condizione c=[z,s].
    - Proiezione finale in C canali e tanh → [-1, 1].

    Args:
        seq_len: L, lunghezza della sequenza temporale generata.
        channels: C, numero di feature per timestep.
        cond_dim: dim(s), dimensione del vettore condizionale esterno.
        latent_dim: dim(z), dimensione del rumore latente.
        d_model, depth, nhead, dim_ff, dropout: iperparametri dei Transformer.

    Forward:
        z: [B, latent_dim]  (variabilità)
        s: [B, cond_dim]    (condizione esterna)
        return: [B, L, C] in [-1,1]
    """
    def __init__(self,
                 seq_len=120,
                 channels=5,
                 cond_dim=4,
                 latent_dim=128,
                 d_model=256,
                 depth=6,
                 nhead=8,
                 dim_ff=512,
                 dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.total_cond = latent_dim + cond_dim

        # Base temporale: token appresi per ogni posizione + pe sinusoidale deterministica
        self.time_tokens = nn.Parameter(torch.randn(seq_len, d_model))          # [L,d]
        self.pos_enc = SinusoidalPositionalEncoding(d_model, seq_len)           # [1,L,d]

        # Proiezioni per creare:
        # - un bias globale g da sommare a tutti i timestep
        # - un vettore di condizione c = [z,s] per FiLM nei blocchi
        self.noise_proj = nn.Linear(latent_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        self.blocks = nn.ModuleList([
            CondTransformerBlock(d_model, nhead, dim_ff, self.total_cond, dropout)
            for _ in range(depth)
        ])
        self.to_data = nn.Linear(d_model, channels)

    def forward(self, z, s):  # z:[B,latent], s:[B,cond]
        B = z.size(0)
        c = torch.cat([z, s], dim=1)  # [B, total_cond] → condizione per FiLM nei blocchi

        # Tela temporale: [B, L, d] = (token appresi + PE) replicati per il batch
        base = self.time_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, L, d]
        pos = self.pos_enc(base)                                # [1, L, d]
        h = base + pos                                          # [B, L, d]

        # Iniezione globale di z e s come bias condiviso su tutti i timestep
        g = self.noise_proj(z) + self.cond_proj(s)  # [B,d]
        h = h + g.unsqueeze(1)                      # broadcast su L

        # Trasformazioni condizionate lungo la profondità
        for blk in self.blocks:
            h = blk(h, c)

        # Proiezione nei C canali e squash in [-1,1]
        x = self.to_data(h)               # [B, L, C]
        return torch.tanh(x)              # normalizzazione coerente coi target


# ---------- Discriminator (Projection) ----------
class Discriminator(nn.Module):
    """
    Discriminatore con Projection Head (Miyato & Koyama, 2018):
    - Proietta la sequenza nei token d_model.
    - Preprende un token CLS appreso che "riassume" la sequenza.
    - Aggiunge positional encoding sinusoidale (sul CLS e sui token dati).
    - Passa attraverso blocchi Transformer standard.
    - Dal token CLS finale calcola:
        * uno score scalare via Linear (SN)
        * un termine di proiezione: <W_c s, h_cls> che valuta coerenza con la condizione.
      Lo score finale è la somma dei due.

    Args:
        seq_len, channels, cond_dim, d_model, depth, nhead, dim_ff, dropout: come da G.

    Forward:
        x: [B, L, C] (reale o generato)
        s: [B, cond_dim]
        return: [B] (score per sequenza)
    """
    def __init__(self,
                 seq_len=120,
                 channels=5,
                 cond_dim=4,
                 d_model=256,
                 depth=6,
                 nhead=8,
                 dim_ff=512,
                 dropout=0.1):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.seq_len = seq_len

        # Ingresso: per-timestep C → d_model (SN per stabilità)
        self.from_data = sn(nn.Linear(channels, d_model))
        # Token CLS appreso per riassumere l'intera sequenza
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # PE su (1 + L) posizioni (CLS + L timestep)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, seq_len + 1)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.out = sn(nn.Linear(d_model, 1))         # score "non condizionale"
        self.cond_proj = sn(nn.Linear(cond_dim, d_model))  # embedding condizione per proiezione

    def forward(self, x, s):  # x:[B,L,C], s:[B,cond]
        B = x.size(0)
        h = self.from_data(x)                     # [B,L,d]
        cls = self.cls_token.expand(B, -1, -1)    # [B,1,d]
        h = torch.cat([cls, h], dim=1)            # [B, 1+L, d]
        h = h + self.pos_enc(h)                   # PE su CLS e token

        # Modellazione contestuale lungo la sequenza
        for blk in self.blocks:
            h = blk(h)

        # Riassunto dal token CLS (posizione 0)
        h_cls = self.ln(h[:, 0, :])               # [B,d]

        # Score base + termine di proiezione condizionale
        score = self.out(h_cls)                   # [B,1]
        proj = torch.sum(self.cond_proj(s) * h_cls, dim=1, keepdim=True)  # [B,1]
        return (score + proj).squeeze(1)          # [B]


def build_models(seq_len=120, channels=5, cond_dim=4,
                 latent_dim=128, d_model=256, depth=6, nhead=8, dim_ff=512, dropout=0.1):
    """
    Costruisce coppia (Generator, Discriminator) con iperparametri condivisi.

    Returns:
        G: Generator
        D: Discriminator

    Esempio d'uso:
        G, D = build_models(seq_len=120, channels=5, cond_dim=4)
        z = torch.randn(B, 128)
        s = torch.randn(B, 4)
        x_fake = G(z, s)           # [B,120,5] in [-1,1]
        score = D(x_fake, s)       # [B]
    """
    G = Generator(seq_len, channels, cond_dim, latent_dim, d_model, depth, nhead, dim_ff, dropout)
    D = Discriminator(seq_len, channels, cond_dim, d_model, depth, nhead, dim_ff, dropout)
    return G, D
